import argparse
import gzip
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
from uuid import UUID

import requests
from bittensor.core.config import Config
from bittensor.core.subtensor import Subtensor
from bittensor.utils.btlogging import logging
from bittensor_wallet import Wallet
from oro_sdk.models.terminal_status import TerminalStatus
from oro_sdk.models.claim_work_response import ClaimWorkResponse
from oro_sdk.models.problem_progress_update import ProblemProgressUpdate
from oro_sdk.types import Unset
from src.agent.scoring import blend_final_score
from src.agent.types import SandboxMetadata
from .backend_client import BackendClient, BackendError
from .heartbeat_manager import HeartbeatManager
from .version_collector import collect_service_versions
from .weight_setter import WeightSetterThread
from .retry_queue import LocalRetryQueue
from .progress_reporter import ProgressReporter
from .backoff import ExponentialBackoff
from .models import CompletionRequest
from subnet.sandbox import host_path, build_sandbox_command, SANDBOX_IMAGE

# Auto-update configuration
WATCHTOWER_URL = os.environ.get("ORO_WATCHTOWER_URL", "http://watchtower:8080")
WATCHTOWER_TOKEN = os.environ.get("WATCHTOWER_TOKEN", "oro-watchtower-token")
AUTO_UPDATE_ENABLED = os.environ.get("ORO_AUTO_UPDATE", "true").lower() in (
    "true",
    "1",
    "yes",
)


def _rewrite_localhost_url(url: str) -> str:
    """Rewrite localhost URLs to host.docker.internal for Docker connectivity."""
    if url.startswith("http://localhost:"):
        return url.replace("http://localhost:", "http://host.docker.internal:", 1)
    return url


class Validator:
    def __init__(self):
        self.config = self.get_config()
        self.setup_logging()
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

    def get_config(self):
        # Set up the configuration parser.
        parser = argparse.ArgumentParser()
        # Custom validator arguments for agent evaluation.
        parser.add_argument(
            "--problem-file",
            default="data/synthesize_test.jsonl",
            help="Path to the problem JSONL file for agent evaluation.",
        )
        parser.add_argument(
            "--workspace-dir",
            default=os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            help="Path to the ShoppingBench workspace root directory.",
        )
        parser.add_argument(
            "--sandbox-timeout",
            type=int,
            default=int(os.environ.get("SANDBOX_TIMEOUT", "1800")),
            help="Timeout in seconds for the entire sandbox subprocess (env: SANDBOX_TIMEOUT, default: 1800 = 30 min).",
        )
        parser.add_argument(
            "--sandbox-max-workers",
            type=int,
            default=int(os.environ.get("SANDBOX_MAX_WORKERS", "6")),
            help="Number of parallel problem workers in sandbox (env: SANDBOX_MAX_WORKERS).",
        )
        parser.add_argument(
            "--sandbox-problem-timeout",
            type=float,
            default=float(os.environ.get("SANDBOX_PROBLEM_TIMEOUT", "300")),
            help="Timeout in seconds per problem in sandbox (env: SANDBOX_PROBLEM_TIMEOUT, default: 300 = 5 min).",
        )
        # Backend API configuration
        parser.add_argument(
            "--backend-url",
            default=os.environ.get("ORO_BACKEND_URL", "https://api.oroagents.com"),
            help="Backend API base URL (env: ORO_BACKEND_URL)",
        )
        parser.add_argument(
            "--poll-interval",
            type=int,
            default=int(os.environ.get("ORO_POLL_INTERVAL", "30")),
            help="Seconds between work claim attempts when no work (env: ORO_POLL_INTERVAL)",
        )
        parser.add_argument(
            "--heartbeat-interval",
            type=int,
            default=int(os.environ.get("ORO_HEARTBEAT_INTERVAL", "30")),
            help="Seconds between heartbeats during execution (env: ORO_HEARTBEAT_INTERVAL)",
        )
        parser.add_argument(
            "--weight-update-interval",
            type=int,
            default=int(os.environ.get("ORO_WEIGHT_UPDATE_INTERVAL", "300")),
            help="Seconds between weight updates from leaderboard (env: ORO_WEIGHT_UPDATE_INTERVAL)",
        )
        # Adds override arguments for network and netuid.
        parser.add_argument(
            "--netuid", type=int, default=15, help="The chain subnet uid."
        )
        # Adds subtensor specific arguments.
        Subtensor.add_args(parser)
        # Adds logging specific arguments.
        logging.add_args(parser)
        # Adds wallet specific arguments.
        Wallet.add_args(parser)
        # Parse the config.
        config = Config(parser)
        # Set up logging directory.
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/validator".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey,
                config.netuid,
            )
        )
        # Ensure the logging directory exists.
        os.makedirs(config.full_path, exist_ok=True)
        return config

    def setup_logging(self):
        # Set up logging — default to INFO level so run activity is visible.
        if not self.config.logging.debug and not self.config.logging.trace:
            self.config.logging.info = True
        logging(config=self.config, logging_dir=self.config.full_path)
        logging.info(
            f"Running validator for subnet: {self.config.netuid} on network: {self.config.subtensor.network} with config:"
        )
        logging.info(self.config)

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

    def download_agent(self, url: str, eval_run_id: str) -> Optional[Path]:
        """Download agent file from URL to per-evaluation directory.

        Args:
            url: The URL to download the agent file from.
            eval_run_id: The evaluation run identifier.

        Returns:
            Path to the downloaded agent file, or None if download failed.
        """
        try:
            # When running inside Docker, rewrite localhost URLs to
            # host.docker.internal so presigned S3 URLs (LocalStack) work.
            url = _rewrite_localhost_url(url)
            logging.info(f"Downloading agent from {url} for eval_run {eval_run_id}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            eval_dir = self._eval_dir(eval_run_id)
            agent_path = eval_dir / "agent.py"
            agent_path.write_text(response.text)
            logging.info(f"Successfully downloaded agent to {agent_path}")
            return agent_path
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download agent from {url}: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error downloading agent from {url}: {e}")
            return None

    def run_sandbox(
        self,
        agent_path: Path,
        eval_run_id: str,
        problem_file: Optional[Path] = None,
        chutes_access_token: Optional[str] = None,
    ) -> tuple[Optional[Path], SandboxMetadata]:
        """Run sandbox with downloaded agent, return output file path and metadata.

        Returns:
            Tuple of (output file path or None, sandbox metadata dict).
            The metadata dict contains exit_code, duration_seconds, and stderr_tail.
        """
        eval_dir = self._eval_dir(eval_run_id)
        output_file = eval_dir / "output.jsonl"

        stdout_log = eval_dir / "sandbox_stdout.log"
        stderr_log = eval_dir / "sandbox_stderr.log"

        metadata: SandboxMetadata = {"exit_code": None, "duration_seconds": None, "stderr_tail": None}

        workspace_dir = Path(self.config.workspace_dir)
        ws = str(workspace_dir)

        # Each evaluation gets an isolated subdirectory. The sandbox only sees
        # its own agent, problems, and output — not other evaluations' files.
        eval_dir_host = host_path(str(eval_dir), workspace_dir=ws)

        # Build docker run command — mount eval dir at /app/logs
        # NOTE: Do NOT mount data/ into the sandbox — it contains the problem
        # suite with ground truth answers (product_ids). Agents could read it
        # to cheat. The sandbox only needs the proxy for search/inference.
        cmd = build_sandbox_command(
            agent_host_path="",
            logs_host_path=eval_dir_host,
            problem_file_arg="/app/logs/problems.jsonl",
            output_path="/app/logs/output.jsonl",
            chutes_access_token=chutes_access_token,
            agent_container_path="/app/logs/agent.py",
            max_workers=self.config.sandbox_max_workers,
            timeout=self.config.sandbox_problem_timeout,
        )

        logging.info(f"Running sandbox for eval_run {eval_run_id}")
        log_cmd = [
            arg.split("=")[0] + "=***" if "CHUTES_ACCESS_TOKEN=" in arg else arg
            for arg in cmd
        ]
        logging.info(f"Sandbox command: {' '.join(log_cmd)}")

        try:
            start_time = time.time()
            with (
                open(stdout_log, "w") as stdout_file,
                open(stderr_log, "w") as stderr_file,
            ):
                result = subprocess.run(
                    cmd,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    timeout=self.config.sandbox_timeout,
                )
            metadata["duration_seconds"] = round(time.time() - start_time, 1)
            metadata["exit_code"] = result.returncode

            # Always log sandbox output for debugging
            if stderr_log.exists():
                stderr_content = stderr_log.read_text()
                if stderr_content.strip():
                    metadata["stderr_tail"] = stderr_content[-500:]
                    log_fn = logging.error if result.returncode != 0 else logging.info
                    log_fn(
                        f"Sandbox stderr for eval_run {eval_run_id}:\n{stderr_content}"
                    )

            if stdout_log.exists():
                stdout_content = stdout_log.read_text()
                if stdout_content.strip():
                    logging.info(
                        f"Sandbox stdout for eval_run {eval_run_id}:\n{stdout_content}"
                    )

            if result.returncode != 0:
                logging.error(
                    f"Sandbox execution failed for eval_run {eval_run_id} (exit code: {result.returncode})"
                )
                # Partial success: sandbox exits non-zero when some problems
                # fail/timeout, but still writes successful results to the
                # output file.  Return the file so those results are scored.
                if output_file.exists() and output_file.stat().st_size > 0:
                    logging.info(
                        f"Sandbox exited with errors but output file exists for {eval_run_id}, "
                        "continuing with partial results"
                    )
                    return output_file, metadata
                return None, metadata

            if output_file.exists():
                logging.info(
                    f"Sandbox completed successfully for eval_run {eval_run_id}"
                )
                return output_file, metadata
            else:
                logging.error(
                    f"Output file not found after sandbox execution: {output_file}"
                )
                if stderr_log.exists():
                    stderr_content = stderr_log.read_text()
                    if stderr_content.strip():
                        logging.error(f"Sandbox stderr:\n{stderr_content}")
                return None, metadata

        except subprocess.TimeoutExpired:
            metadata["duration_seconds"] = round(time.time() - start_time, 1)
            metadata["exit_code"] = -1
            if stderr_log.exists():
                stderr_content = stderr_log.read_text()
                if stderr_content.strip():
                    metadata["stderr_tail"] = stderr_content[-500:]
            logging.warning(
                f"Sandbox suite timeout ({self.config.sandbox_timeout}s) hit for eval_run {eval_run_id}, "
                "checking for partial results"
            )
            if output_file.exists() and output_file.stat().st_size > 0:
                logging.info(
                    f"Suite timed out but output file exists for {eval_run_id}, "
                    "continuing with partial results"
                )
                return output_file, metadata
            return None, metadata
        except Exception as e:
            logging.error(f"Error running sandbox for eval_run {eval_run_id}: {e}")
            return None, metadata

    def _check_for_updates(self):
        """Trigger Watchtower update check and pull sandbox image.

        Called between evaluation cycles. All errors are caught — never crashes the main loop.
        After Watchtower restarts services, waits for proxy /health (which transitively
        covers search-server) before returning.
        """
        if not AUTO_UPDATE_ENABLED:
            return

        try:
            logging.info("Triggering Watchtower update check...")
            resp = requests.get(
                f"{WATCHTOWER_URL}/v1/update",
                headers={"Authorization": f"Bearer {WATCHTOWER_TOKEN}"},
                timeout=300,
            )
            if resp.ok:
                logging.info(
                    f"Watchtower update check completed (status {resp.status_code})"
                )
            else:
                logging.warning(f"Watchtower update check returned {resp.status_code}")
        except requests.exceptions.ConnectionError:
            logging.debug("Watchtower not reachable, skipping update check")
        except Exception as e:
            logging.warning(f"Watchtower update check failed: {e}")

        # Wait for proxy to be healthy (covers search-server transitively).
        # Watchtower blocks during restarts but doesn't wait for Docker healthchecks.
        for attempt in range(30):
            try:
                if requests.get("http://proxy:80/health", timeout=5).ok:
                    break
            except Exception:
                pass
            time.sleep(10)

        try:
            result = subprocess.run(
                ["docker", "pull", SANDBOX_IMAGE],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                logging.warning(f"Sandbox image pull failed: {result.stderr.strip()}")
        except (subprocess.SubprocessError, OSError, FileNotFoundError) as e:
            logging.warning(f"Sandbox image pull failed: {e}")

        # Re-collect service versions after potential updates
        self.service_versions = collect_service_versions()

    def run(self):
        """Main validation loop - claims work from Backend and executes evaluations."""
        logging.info("Starting validator loop.")

        # Track current execution state for debugging
        self._current_eval_run_id = None

        # Check for updates every 5 minutes even when idle
        UPDATE_CHECK_INTERVAL = 300
        self._last_update_check = 0

        # Start weight setter thread
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
                    # Check for image updates every 5 minutes
                    if time.time() - self._last_update_check >= UPDATE_CHECK_INTERVAL:
                        self._check_for_updates()
                        self._last_update_check = time.time()

                    # Log current state
                    if self._current_eval_run_id:
                        logging.warning(
                            f"Still tracking eval run {self._current_eval_run_id} - this should not happen!"
                        )

                    # Claim work from Backend
                    logging.info("Claiming work from Backend...")
                    work = self.backend_client.claim_work(
                        service_versions=self.service_versions
                    )

                    if work is None:
                        logging.info(
                            f"No work available, sleeping {self.config.poll_interval}s"
                        )
                        time.sleep(self.config.poll_interval)
                        self.backoff.reset()
                        continue

                    logging.info(
                        f"Claimed work: {work.eval_run_id} "
                        f"(agent_version={work.agent_version_id}, suite={work.suite_id})"
                    )
                    self.backoff.reset()

                    # Track the current run
                    self._current_eval_run_id = work.eval_run_id

                    # Execute evaluation cycle
                    logging.info(f"Starting evaluation cycle for {work.eval_run_id}")
                    self.run_evaluation_cycle(work)
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
    ) -> tuple[Optional[Path], list[UUID], list[dict]]:
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

        # Extract miner's Chutes access token (minted by Backend from refresh token).
        # Miners are required to authenticate with Chutes — if no token is
        # available the evaluation cannot proceed.
        chutes_access_token: Optional[str] = None
        if not isinstance(work.chutes_access_token, Unset) and work.chutes_access_token:
            chutes_access_token = work.chutes_access_token
            logging.info(f"Using miner's Chutes token for {eval_run_id_str}")
        else:
            logging.warning(
                f"No miner Chutes token for {eval_run_id_str}, cannot run inference"
            )

        # Track temp files for cleanup
        problem_file = None
        agent_path = None
        workspace_dir = Path(self.config.workspace_dir)

        # Step 0: Verify miner Chutes token is present and valid
        if not chutes_access_token:
            self._complete_with_failure(
                eval_run_id,
                TerminalStatus.FAILED,
                "Miner has no Chutes token — cannot fund inference",
            )
            return

        # Validate the token can actually make inference calls
        token_valid, token_reason = self._validate_chutes_token(chutes_access_token)
        if not token_valid:
            self._complete_with_failure(eval_run_id, TerminalStatus.FAILED, token_reason)
            return

        # Start heartbeat manager only after token validation passes
        heartbeat_mgr = HeartbeatManager(
            backend_client=self.backend_client,
            eval_run_id=eval_run_id,
            interval_seconds=self.config.heartbeat_interval,
            service_versions=self.service_versions,
        )
        heartbeat_mgr.start()
        logging.info(f"Heartbeat manager started for {eval_run_id_str}")

        try:
            # Step 1: Download agent code
            agent_path = self.download_agent(work.code_download_url, eval_run_id_str)
            if not agent_path:
                self._complete_with_failure(
                    eval_run_id, TerminalStatus.FAILED, "Download failed"
                )
                return

            # Step 2: Fetch problems from Backend API
            # Returns sanitized file (no rewards) for sandbox + full problems for scorer
            problem_file, problem_ids, problems = self.fetch_problems(
                work.suite_id, eval_run_id_str
            )
            if not problem_file or not problems:
                self._complete_with_failure(
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
                chutes_access_token=chutes_access_token,
            )
            progress_reporter.start_monitoring()

            try:
                sandbox_output, sandbox_metadata = self.run_sandbox(
                    agent_path,
                    eval_run_id_str,
                    problem_file,
                    chutes_access_token=chutes_access_token,
                )
            finally:
                progress_reporter.signal_sandbox_done()
                progress_reporter.wait_for_completion()

            if not sandbox_output:
                self._complete_with_failure(
                    eval_run_id, TerminalStatus.FAILED, "Sandbox execution failed",
                    sandbox_metadata=sandbox_metadata,
                )
                return

            # Step 4: Get aggregate score from ProgressReporter
            aggregate = progress_reporter.get_aggregate_score()

            if aggregate is None:
                self._complete_with_failure(
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
                self._complete_with_failure(
                    eval_run_id,
                    TerminalStatus.FAILED,
                    f"Reasoning judge infrastructure failure ({judge_failed}/{judge_total} calls failed)",
                    sandbox_metadata=sandbox_metadata,
                )
                return

            score = blend_final_score(success_rate, reasoning_result["reasoning_quality"])

            aggregate["reasoning_quality"] = reasoning_result["reasoning_quality"]
            aggregate["reasoning_coefficient"] = reasoning_result["reasoning_coefficient"]
            aggregate["judge_inference_failed"] = reasoning_result["judge_inference_failed"]
            aggregate["judge_inference_total"] = reasoning_result["judge_inference_total"]

            logging.info(
                f"Score: final={score:.4f} "
                f"(success_rate={success_rate:.4f} * "
                f"coefficient={reasoning_result['reasoning_coefficient']:.4f}, "
                f"reasoning_quality={reasoning_result['reasoning_quality']:.4f})"
            )

            # Step 5: Upload logs (reasoning data appended to each problem's trajectory)
            results_s3_key = self._upload_logs(
                eval_run_id, output_file, problem_ids, progress_reporter
            )

            # Step 6: Complete the run
            self._complete_run(
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
            self._complete_with_failure(
                eval_run_id, TerminalStatus.FAILED, str(e),
                sandbox_metadata=sandbox_metadata if 'sandbox_metadata' in locals() else None,
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

    def _upload_logs(
        self,
        eval_run_id: UUID,
        output_file: Path,
        problem_ids: list[UUID],
        progress_reporter: "ProgressReporter",
    ) -> str:
        """Upload per-problem evaluation logs to S3.

        The output JSONL file contains one line per problem, each being a JSON
        array of trajectory steps with ``extra_info.problem_id``. This method
        splits the file by problem and uploads each as a separate gzipped object
        so the Frontend can fetch trajectories per-problem.

        After uploading, reports the S3 keys back to the Backend via progress
        update so the download endpoint can locate them.

        Args:
            eval_run_id: The evaluation run ID (UUID).
            output_file: Path to the output JSONL file.
            problem_ids: List of problem UUIDs from the suite.
            progress_reporter: ProgressReporter with per-problem scoring results.

        Returns:
            S3 key of the last successfully uploaded log (stored on the run).
        """
        try:
            if not output_file.exists():
                logging.warning(f"Output file not found: {output_file}")
                return ""

            if not problem_ids:
                logging.warning("No problem_ids available for log upload, skipping")
                return ""

            # Parse JSONL and group lines by problem_id
            problem_lines: dict[str, bytes] = {}
            for raw_line in output_file.read_bytes().splitlines():
                if not raw_line.strip():
                    continue
                try:
                    steps = json.loads(raw_line)
                    # Extract problem_id from first step's extra_info
                    if isinstance(steps, list) and steps:
                        pid = steps[0].get("extra_info", {}).get("problem_id")
                        if pid:
                            problem_lines[pid] = raw_line
                            continue
                except (json.JSONDecodeError, KeyError):
                    pass
                # Fallback: couldn't parse, skip this line
                logging.debug(f"Skipping unparseable JSONL line in {output_file}")

            if not problem_lines:
                # Fall back to uploading entire file under first problem_id
                logging.warning(
                    "Could not split output by problem_id, uploading as single file"
                )
                problem_lines[str(problem_ids[0])] = output_file.read_bytes()

            last_s3_key = ""
            uploaded_keys: dict[UUID, str] = {}  # problem_id → s3_key
            for pid_str, line_data in problem_lines.items():
                try:
                    pid = UUID(pid_str)
                except ValueError:
                    logging.warning(
                        f"Invalid problem_id in output: {pid_str}, skipping"
                    )
                    continue

                compressed = gzip.compress(line_data)

                presign = self.backend_client.get_presigned_upload_url(
                    content_length=len(compressed),
                    eval_run_id=eval_run_id,
                    problem_id=pid,
                )

                # Rewrite localhost URLs for Docker → host connectivity
                if hasattr(presign, "upload_url"):
                    presign.upload_url = _rewrite_localhost_url(presign.upload_url)

                self.backend_client.upload_to_s3(presign, compressed)
                logging.info(f"Uploaded logs to {presign.results_s3_key}")
                last_s3_key = presign.results_s3_key
                uploaded_keys[pid] = presign.results_s3_key

            # Report S3 keys back to Backend so download endpoint can find them
            if uploaded_keys:
                progress_updates = [
                    ProblemProgressUpdate(
                        problem_id=pid,
                        status=progress_reporter.get_problem_status(str(pid)),
                        logs_s3_key=s3_key,
                    )
                    for pid, s3_key in uploaded_keys.items()
                ]
                try:
                    self.backend_client.report_progress(eval_run_id, progress_updates)
                    logging.info(
                        f"Reported logs_s3_key for {len(uploaded_keys)} problems"
                    )
                except Exception as e:
                    logging.warning(f"Failed to report logs_s3_key: {e}")
                    for update in progress_updates:
                        self.retry_queue.add_progress(eval_run_id, update)

            return last_s3_key
        except Exception as e:
            logging.error(f"Failed to upload logs: {e}")
            return ""

    def _complete_run(
        self,
        eval_run_id: UUID,
        status: TerminalStatus,
        score: float,
        results_s3_key: str = "",
        score_components: Optional[Dict[str, Any]] = None,
        sandbox_metadata: Optional[SandboxMetadata] = None,
    ) -> None:
        """Complete an evaluation run, with retry queue fallback.

        Args:
            eval_run_id: The evaluation run ID (UUID).
            status: Terminal status (TerminalStatus enum).
            score: Evaluation score.
            results_s3_key: S3 key for logs.
            score_components: Optional dict with detailed score breakdown.
            sandbox_metadata: Optional sandbox execution metadata.
        """
        if score_components is None:
            score_components = {"success_rate": score}

        try:
            result = self.backend_client.complete_run(
                eval_run_id=eval_run_id,
                status=status,
                score=score,
                score_components=score_components,
                results_s3_key=results_s3_key,
                sandbox_metadata=sandbox_metadata,
            )
            logging.info(
                f"Completed {eval_run_id}: {result.status}, "
                f"eligible={result.agent_version_became_eligible}"
            )
        except BackendError as e:
            if e.is_run_already_complete:
                logging.info(f"Run {eval_run_id} already complete, skipping")
            elif e.is_not_run_owner:
                logging.warning(f"Lost ownership of run {eval_run_id}, skipping")
            elif e.is_eval_run_not_found:
                logging.warning(f"Run {eval_run_id} not found, skipping")
            elif e.is_transient:
                logging.warning(
                    f"Backend unavailable for complete, queueing retry: {e}"
                )
                self.retry_queue.add(
                    CompletionRequest(
                        eval_run_id=eval_run_id,
                        status=status,
                        validator_score=score,
                        score_components=score_components,
                        results_s3_key=results_s3_key,
                        sandbox_metadata=sandbox_metadata,
                    )
                )
            else:
                logging.error(f"Non-transient error completing run {eval_run_id}: {e}")

    @staticmethod
    def _validate_chutes_token(access_token: str) -> tuple[bool, str]:
        """Validate a Chutes access token by making a minimal inference call.

        A single 1-token completion catches both invalid tokens (401) and
        zero-balance accounts (402). The models listing endpoint can't
        distinguish pro-plan users (balance=0 but have quotas) from
        genuinely broke accounts.

        Returns (is_valid, failure_reason). On transient errors (5xx,
        timeout, 429), returns (True, "") to avoid failing runs unnecessarily.
        """
        import requests

        try:
            resp = requests.post(
                "https://llm.chutes.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "Qwen/Qwen3-32B-TEE",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                return True, ""
            if resp.status_code == 401:
                return False, "Chutes token invalid or expired (HTTP 401)"
            if resp.status_code == 402:
                detail = resp.json().get("detail", {})
                msg = detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail)
                return False, f"Miner's Chutes account has no credits ({msg})"
            if resp.status_code == 429:
                # Rate limited — token is valid, just throttled
                return True, ""
            logging.warning("Chutes token validation inconclusive: status=%s", resp.status_code)
            return True, ""
        except Exception as exc:
            logging.warning("Chutes token validation error: %s", exc)
            return True, ""

    def _complete_with_failure(
        self,
        eval_run_id: UUID,
        status: TerminalStatus,
        reason: str,
        sandbox_metadata: Optional[SandboxMetadata] = None,
    ) -> None:
        """Report a failed evaluation to Backend, with retry queue fallback.

        Args:
            eval_run_id: The evaluation run ID (UUID).
            status: Terminal status (TerminalStatus enum).
            reason: Failure reason for logging.
            sandbox_metadata: Optional sandbox execution metadata.
        """
        logging.error(f"Evaluation {eval_run_id} failed: {reason}")
        logging.info(f"Reporting failure to Backend with status={status.value}...")
        try:
            result = self.backend_client.complete_run(
                eval_run_id=eval_run_id,
                status=status,
                failure_reason=reason,
                sandbox_metadata=sandbox_metadata,
            )
            logging.info(
                f"Successfully completed failed run {eval_run_id}: "
                f"status={result.status}, work_item_closed={result.work_item.is_closed}"
            )
        except BackendError as e:
            if e.is_run_already_complete:
                logging.info(f"Run {eval_run_id} already complete, skipping")
            elif e.is_not_run_owner:
                logging.warning(f"Lost ownership of run {eval_run_id}, skipping")
            elif e.is_eval_run_not_found:
                logging.warning(f"Run {eval_run_id} not found, skipping")
            elif e.is_transient:
                logging.warning(
                    f"Backend unavailable for failure report, queueing retry: {e}"
                )
                self.retry_queue.add(
                    CompletionRequest(
                        eval_run_id=eval_run_id,
                        status=status,
                        failure_reason=reason,
                        sandbox_metadata=sandbox_metadata,
                    )
                )
            else:
                logging.error(
                    f"Non-transient error reporting failure for {eval_run_id}: {e} "
                    f"(status_code={e.status_code}, error_code={e.error_code})"
                )
        except Exception as e:
            logging.error(
                f"Unexpected error reporting failure to Backend: {type(e).__name__}: {e}"
            )


# Run the validator.
if __name__ == "__main__":
    validator = Validator()
    validator.run()
