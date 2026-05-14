import json
import os
import time
import traceback
from pathlib import Path
from typing import Optional
from uuid import UUID

from bittensor.core.subtensor import Subtensor
from bittensor.utils.btlogging import logging
from bittensor_wallet import Wallet
from oro_sdk.models.terminal_status import TerminalStatus
from oro_sdk.models.claim_work_response import ClaimWorkResponse
from oro_sdk.types import Unset
from src.agent.scoring import blend_final_score
from src.agent.types import ProblemDict

from . import auto_update
from .backend_client import BackendClient, BackendError
from .backoff import ExponentialBackoff
from .completion import CompletionReporter
from .config import METRICS_PORT, build_config, configure_logging
from .heartbeat_manager import HeartbeatManager
from .inference_token import validate_inference_token, validation_model_for
from .metrics import (
    ACTIVE_RUNS,
    CLAIM_WORK_SECONDS,
    CLAIM_WORK_TOTAL,
)
from .progress_reporter import ProgressReporter
from .resource_collector import collect_resource_metrics
from .retry_queue import LocalRetryQueue
from .sandbox_runner import SandboxRunner
from .version_collector import collect_service_versions
from .weight_setter import WeightSetterThread


class Validator:
    def __init__(self):
        self.config = build_config()
        configure_logging(self.config)
        self.setup_bittensor_objects()

        # Backend API client
        self.backend_client = BackendClient(
            base_url=self.config.backend_url,
            wallet=self.wallet,
        )

        # Retry queue for failed completions
        self.retry_queue = LocalRetryQueue(self.backend_client)

        # Backoff for transient errors
        self.backoff = ExponentialBackoff()

        # Collect Docker image digests for version tracking
        self.service_versions = collect_service_versions()

        # Per-eval helpers wired with shared deps
        self.sandbox_runner = SandboxRunner(self.config, self._eval_dir)
        self.completion = CompletionReporter(self.backend_client, self.retry_queue)

    def setup_bittensor_objects(self):
        # Build Bittensor validator objects.
        logging.info("Setting up Bittensor objects.")

        # Initialize wallet.
        self.wallet = Wallet(config=self.config)
        logging.info(f"Wallet: {self.wallet}")

        # Initialize subtensor.
        self.subtensor = Subtensor(config=self.config)
        logging.info(f"Subtensor: {self.subtensor}")

        # Initialize metagraph.
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        logging.info(f"Metagraph: {self.metagraph}")

        # Connect the validator to the network.
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            logging.error(
                f"Your validator: {self.wallet} is not registered to chain connection: {self.subtensor} \nRun 'btcli register' and try again."
            )
            exit()
        else:
            # Each validator gets a unique identity (UID) in the network.
            self.my_subnet_uid = self.metagraph.hotkeys.index(
                self.wallet.hotkey.ss58_address
            )
            logging.info(f"Running validator on uid: {self.my_subnet_uid}")

    def _eval_dir(self, eval_run_id_str: str) -> Path:
        """Return per-evaluation subdirectory under logs/, creating it if needed.

        Each evaluation gets its own directory so the sandbox only sees its own
        files — it cannot read other agents' code, problem files, or output.
        """
        d = Path(self.config.workspace_dir) / "logs" / f"eval_{eval_run_id_str}"
        d.mkdir(parents=True, exist_ok=True)
        # Sandbox runs as --user 1000:1000, ensure it can write to this directory
        os.chmod(d, 0o777)
        return d

    def run(self):
        """Main validation loop - claims work from Backend and executes evaluations."""
        logging.info("Starting validator loop.")

        from prometheus_client import start_http_server

        start_http_server(METRICS_PORT)
        logging.info(f"Prometheus /metrics server listening on :{METRICS_PORT}")

        self._current_eval_run_id = None

        UPDATE_CHECK_INTERVAL = 300
        self._last_update_check = 0

        weight_setter = WeightSetterThread(
            backend_client=self.backend_client,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
            wallet=self.wallet,
            netuid=self.config.netuid,
            interval_seconds=self.config.weight_update_interval,
        )
        weight_setter.start()
        logging.info(
            f"Weight setter started (interval: {self.config.weight_update_interval}s)"
        )

        try:
            while True:
                try:
                    if time.time() - self._last_update_check >= UPDATE_CHECK_INTERVAL:
                        if auto_update.check_for_updates():
                            self.service_versions = collect_service_versions()
                        self._last_update_check = time.time()

                    # Log current state
                    if self._current_eval_run_id:
                        logging.warning(
                            f"Still tracking eval run {self._current_eval_run_id} - this should not happen!"
                        )

                    # Claim work from Backend
                    logging.info("Claiming work from Backend...")
                    with CLAIM_WORK_SECONDS.time():
                        try:
                            work = self.backend_client.claim_work(
                                service_versions=self.service_versions,
                                resource_metrics=collect_resource_metrics(),
                            )
                        except Exception:
                            CLAIM_WORK_TOTAL.labels(result="error").inc()
                            raise

                    if work is None:
                        CLAIM_WORK_TOTAL.labels(result="empty").inc()
                        logging.info(
                            f"No work available, sleeping {self.config.poll_interval}s"
                        )
                        time.sleep(self.config.poll_interval)
                        self.backoff.reset()
                        continue

                    CLAIM_WORK_TOTAL.labels(result="success").inc()
                    logging.info(
                        f"Claimed work: {work.eval_run_id} "
                        f"(agent_version={work.agent_version_id}, suite={work.suite_id})"
                    )
                    self.backoff.reset()

                    # Track the current run
                    self._current_eval_run_id = work.eval_run_id
                    ACTIVE_RUNS.inc()

                    # Execute evaluation cycle
                    logging.info(f"Starting evaluation cycle for {work.eval_run_id}")
                    try:
                        self.run_evaluation_cycle(work)
                    finally:
                        ACTIVE_RUNS.dec()
                    logging.info(f"Completed evaluation cycle for {work.eval_run_id}")

                    # Clear the tracking
                    self._current_eval_run_id = None

                    # Process any pending retries
                    if self.retry_queue.get_pending_count() > 0:
                        logging.info("Processing retry queue...")
                        self.retry_queue.process_pending()

                except BackendError as e:
                    if e.is_banned:
                        # Banned validators should poll infrequently — the ban won't
                        # be lifted for a while and frequent polling wastes resources.
                        ban_sleep = 300  # 5 minutes
                        logging.warning(
                            f"Validator is banned: {e}. "
                            f"Sleeping {ban_sleep}s before retrying."
                        )
                        time.sleep(ban_sleep)
                    elif e.is_transient:
                        sleep_time = self.backoff.next()
                        logging.warning(f"Backend unavailable: {e}")
                        logging.info(f"Backing off for {sleep_time:.1f}s")
                        time.sleep(sleep_time)
                    elif e.is_at_capacity:
                        # AT_CAPACITY should back off with jitter - either we have a stuck
                        # run or there's a race condition we need to wait out
                        sleep_time = self.backoff.next()
                        logging.warning(
                            f"At capacity (current tracked run: {self._current_eval_run_id}): {e}"
                        )
                        logging.info(f"Backing off for {sleep_time:.1f}s")
                        time.sleep(sleep_time)
                    else:
                        logging.error(f"Non-transient backend error: {e}")
                        time.sleep(5)

                except (TimeoutError, ConnectionError, OSError) as e:
                    # Network/system errors - log and retry
                    logging.warning(
                        f"Network/system error in main loop: {type(e).__name__}: {e}"
                    )
                    self._current_eval_run_id = None
                    time.sleep(5)

                except (TypeError, AttributeError, KeyError, ValueError) as e:
                    # Programming errors - these indicate bugs, log with full traceback
                    logging.error(
                        f"Programming error in main loop: {type(e).__name__}: {e}"
                    )
                    traceback.print_exc()
                    self._current_eval_run_id = None
                    time.sleep(5)

                except Exception as e:
                    # Catch-all for unexpected errors - log type for debugging
                    logging.error(
                        f"Unexpected error in main loop ({type(e).__name__}): {e}"
                    )
                    traceback.print_exc()
                    self._current_eval_run_id = None
                    time.sleep(5)

        except KeyboardInterrupt:
            logging.info("Keyboard interrupt detected, shutting down...")
        finally:
            weight_setter.stop()
            logging.info("Validator stopped.")

    def fetch_problems(
        self, suite_id: int, eval_run_id_str: str
    ) -> tuple[Optional[Path], list[UUID], list[ProblemDict]]:
        """Fetch problems from Backend API and save sanitized version to temp file.

        The full problem data (with reward/voucher) is returned in memory for
        the ProgressReporter to use during scoring. The file written to disk
        has reward/voucher stripped so the sandbox cannot read ground-truth answers.

        Returns:
            Tuple of (problem file path, problem UUIDs, full problems with rewards).
            Returns (None, [], []) if fetch failed.
        """
        try:
            eval_run_id = UUID(eval_run_id_str)
            logging.info(f"Fetching problems for run {eval_run_id_str}")
            problems = self.backend_client.get_run_problems(eval_run_id)

            if not problems:
                logging.error(f"No problems returned for suite {suite_id}")
                return None, [], []

            # Extract problem_ids for later use (e.g., log upload)
            problem_ids = []
            for problem in problems:
                pid = problem.get("problem_id") or problem.get("id")
                if pid:
                    problem_ids.append(UUID(pid) if isinstance(pid, str) else pid)

            # Write sanitized problems to per-evaluation directory.
            # Strip ground-truth fields so the sandbox cannot read answers.
            # reward_title_embeddings keys are verbatim product titles.
            eval_dir = self._eval_dir(eval_run_id_str)
            problem_file = eval_dir / "problems.jsonl"
            with open(problem_file, "w") as f:
                for problem in problems:
                    sanitized = {
                        k: v
                        for k, v in problem.items()
                        if k not in ("reward", "voucher", "reward_title_embeddings")
                    }
                    f.write(json.dumps(sanitized) + "\n")

            logging.info(f"Saved {len(problems)} problems to {problem_file}")
            return problem_file, problem_ids, problems
        except BackendError as e:
            logging.error(f"Failed to fetch problems: {e}")
            return None, [], []
        except Exception as e:
            logging.error(f"Unexpected error fetching problems: {e}")
            return None, [], []

    def run_evaluation_cycle(self, work: ClaimWorkResponse):
        """Execute a single evaluation cycle for claimed work.

        Args:
            work: ClaimWorkResponse from SDK.
        """
        eval_run_id = work.eval_run_id  # UUID from SDK
        eval_run_id_str = str(eval_run_id)  # String for file paths/logging

        inference_provider: Optional[str] = None
        inference_access_token: Optional[str] = None
        inference_base_url: Optional[str] = None
        if not isinstance(work.inference_token, Unset) and work.inference_token:
            inference_provider = work.inference_token.provider
            inference_access_token = work.inference_token.access_token
            inference_base_url = work.inference_token.base_url
            logging.info(
                f"Using miner's {inference_provider} token for {eval_run_id_str} "
                f"(base_url={inference_base_url})"
            )
        else:
            logging.warning(
                f"No miner inference token for {eval_run_id_str}, cannot run inference"
            )

        # Track temp files for cleanup
        problem_file = None
        agent_path = None
        workspace_dir = Path(self.config.workspace_dir)

        # Step 0: Verify miner inference token is present and valid
        if (
            not inference_access_token
            or not inference_base_url
            or not inference_provider
        ):
            self.completion.complete_with_failure(
                eval_run_id,
                TerminalStatus.FAILED,
                "Miner has no inference token — cannot fund inference",
            )
            return

        # Validate the token can actually make inference calls
        token_valid, token_reason = validate_inference_token(
            inference_access_token,
            inference_base_url,
            validation_model_for(inference_provider),
        )
        if not token_valid:
            self.completion.complete_with_failure(
                eval_run_id, TerminalStatus.FAILED, token_reason
            )
            return

        # Start heartbeat manager only after token validation passes
        heartbeat_mgr = HeartbeatManager(
            backend_client=self.backend_client,
            eval_run_id=eval_run_id,
            interval_seconds=self.config.heartbeat_interval,
            service_versions=self.service_versions,
            resource_metrics_provider=collect_resource_metrics,
        )
        heartbeat_mgr.start()
        logging.info(f"Heartbeat manager started for {eval_run_id_str}")

        try:
            # Step 1: Download agent code
            agent_path = self.sandbox_runner.download_agent(
                work.code_download_url, eval_run_id_str
            )
            if not agent_path:
                self.completion.complete_with_failure(
                    eval_run_id, TerminalStatus.FAILED, "Download failed"
                )
                return

            # Step 2: Fetch problems from Backend API
            # Returns sanitized file (no rewards) for sandbox + full problems for scorer
            problem_file, problem_ids, problems = self.fetch_problems(
                work.suite_id, eval_run_id_str
            )
            if not problem_file or not problems:
                self.completion.complete_with_failure(
                    eval_run_id, TerminalStatus.FAILED, "Failed to load problems"
                )
                return

            # Step 3: Run sandbox with ProgressReporter for per-problem scoring
            eval_dir = self._eval_dir(eval_run_id_str)
            output_file = eval_dir / "output.jsonl"

            # Start progress reporter (scores problems AND judges reasoning per-problem)
            progress_reporter = ProgressReporter(
                backend_client=self.backend_client,
                eval_run_id=eval_run_id,
                output_file=output_file,
                problems=problems,
                workspace_dir=workspace_dir,
                inference_access_token=inference_access_token,
                inference_provider=inference_provider,
                max_scoring_workers=self.config.reasoning_max_workers,
            )
            progress_reporter.start_monitoring()

            try:
                sandbox_output, sandbox_metadata = self.sandbox_runner.run_sandbox(
                    agent_path,
                    eval_run_id_str,
                    problem_file,
                    inference_access_token=inference_access_token,
                    inference_provider=inference_provider,
                    inference_base_url=inference_base_url,
                )
            finally:
                progress_reporter.signal_sandbox_done()
                progress_reporter.wait_for_completion()

            if not sandbox_output:
                self.completion.complete_with_failure(
                    eval_run_id,
                    TerminalStatus.FAILED,
                    "Sandbox execution failed",
                    sandbox_metadata=sandbox_metadata,
                )
                return

            # Step 4: Get aggregate score from ProgressReporter
            aggregate = progress_reporter.get_aggregate_score()

            if aggregate is None:
                self.completion.complete_with_failure(
                    eval_run_id,
                    TerminalStatus.FAILED,
                    "ProgressReporter did not compute aggregate score",
                )
                return

            success_rate = aggregate.get("success_rate", 0.0)

            # Step 4b: Get reasoning data (judged per-problem during scoring)
            reasoning_result = progress_reporter.get_reasoning_data()

            # If most judge calls failed, treat as infrastructure failure.
            judge_total = reasoning_result["judge_inference_total"]
            judge_failed = reasoning_result["judge_inference_failed"]
            if judge_total >= 3 and judge_failed > judge_total / 2:
                logging.warning(
                    f"Reasoning judge failed on {judge_failed}/{judge_total} calls, "
                    f"completing as FAILED so another validator can retry"
                )
                self.completion.complete_with_failure(
                    eval_run_id,
                    TerminalStatus.FAILED,
                    f"Reasoning judge infrastructure failure ({judge_failed}/{judge_total} calls failed)",
                    sandbox_metadata=sandbox_metadata,
                )
                return

            score = blend_final_score(
                success_rate, reasoning_result["reasoning_quality"]
            )

            aggregate["reasoning_quality"] = reasoning_result["reasoning_quality"]
            aggregate["reasoning_coefficient"] = reasoning_result[
                "reasoning_coefficient"
            ]
            aggregate["judge_inference_failed"] = reasoning_result[
                "judge_inference_failed"
            ]
            aggregate["judge_inference_total"] = reasoning_result[
                "judge_inference_total"
            ]

            logging.info(
                f"Score: final={score:.4f} "
                f"(success_rate={success_rate:.4f} * "
                f"coefficient={reasoning_result['reasoning_coefficient']:.4f}, "
                f"reasoning_quality={reasoning_result['reasoning_quality']:.4f})"
            )

            # Step 5: Upload logs (reasoning data appended to each problem's trajectory)
            results_s3_key = self.completion.upload_logs(
                eval_run_id, output_file, problem_ids, progress_reporter
            )

            # Step 6: Complete the run
            self.completion.complete_run(
                eval_run_id=eval_run_id,
                status=TerminalStatus.SUCCESS,
                score=score,
                score_components=aggregate,
                results_s3_key=results_s3_key,
                sandbox_metadata=sandbox_metadata,
            )

        except Exception as e:
            logging.error(f"Evaluation cycle failed: {e}")
            traceback.print_exc()
            self.completion.complete_with_failure(
                eval_run_id,
                TerminalStatus.FAILED,
                str(e),
                sandbox_metadata=sandbox_metadata
                if "sandbox_metadata" in locals()
                else None,
            )
        finally:
            heartbeat_mgr.stop()
            if not heartbeat_mgr.is_healthy():
                logging.warning(f"Heartbeat failures occurred during {eval_run_id_str}")

            # Cleanup per-evaluation directory
            eval_dir = self._eval_dir(eval_run_id_str)
            if eval_dir.exists():
                try:
                    import shutil

                    shutil.rmtree(eval_dir)
                except OSError as e:
                    logging.debug(f"Cleanup failed for {eval_dir}: {e}")


# Run the validator.
if __name__ == "__main__":
    validator = Validator()
    validator.run()
