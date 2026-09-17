import argparse
import asyncio
import gzip
import json
import os
import random
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
from src.agent.sandbox_executor import (
    merge_inference_stats,
    read_inference_stats,
)
from src.agent.types import ProblemDict, SandboxMetadata
from .backend_client import BackendClient, BackendError
from .bounded_io import RunStoppedEarly, read_text_lossy, run_capped
from .watchdog import ProgressWatchdog
from .heartbeat_manager import HeartbeatManager
from .output_split import split_output_by_problem
from .metrics import (
    ACTIVE_RUNS,
    CLAIM_WORK_SECONDS,
    CLAIM_WORK_TOTAL,
    SANDBOX_ACTIVE,
    SANDBOX_DURATION_SECONDS,
)
from .host_specs import check_host_min_specs
from .resource_collector import collect_resource_metrics
from .version_collector import collect_service_versions
from .weight_setter import WeightSetterThread
from .retry_queue import LocalRetryQueue
from .progress_reporter import ProgressReporter
from .backoff import ExponentialBackoff
from .drain import handle_drain_tick
from .env_pack_loader import fetch_and_validate_pack, fetch_and_validate_race_pack
from .environment_preflight import (
    environment_preflight_run_id,
    run_environment_preflight,
)
from .episode_emitter import emit_finalized_results
from .generated_evaluation import (
    GENERATED_SCORE_SCHEMA,
    aggregate_results,
    select_run_task_roster,
    summarize_episode_resource_usage,
    validate_run_results,
    write_problem_file,
)
from .generated_progress_reporter import GeneratedProgressReporter
from .models import CompletionRequest
from .session_registry import SessionRegistry
from .session_service import SessionRuntime, SessionServer
from subnet.sandbox import host_path, build_sandbox_command, SANDBOX_IMAGE

# Auto-update configuration
WATCHTOWER_URL = os.environ.get("ORO_WATCHTOWER_URL", "http://watchtower:8080")
WATCHTOWER_TOKEN = os.environ.get("WATCHTOWER_TOKEN", "oro-watchtower-token")
AUTO_UPDATE_ENABLED = os.environ.get("ORO_AUTO_UPDATE", "true").lower() in (
    "true",
    "1",
    "yes",
)

# Port the Prometheus /metrics endpoint listens on inside the container.
# Hardcoded because the bundled prometheus.yml scrapes this exact port; an
# operator-tunable arg adds surface area without buying anything.
METRICS_PORT = 9100

# Parallel uploads in _upload_logs. Each worker does a presign roundtrip +
# an S3 PUT, both flowing through the BackendClient's persistent connection
# pools, so 20 workers comfortably fan out an N-problem eval without
# starving the auth-client httpx pool. The matching Backend per-IP cap on
# /v1/validator/* sits comfortably above the resulting RPM.
_UPLOAD_LOGS_WORKERS = 20


@dataclass(frozen=True)
class _EvaluationCompletion:
    score: float
    score_components: dict[str, Any]
    results_s3_key: str
    sandbox_metadata: SandboxMetadata


# Inference-token 401 retry backoff base (seconds). Multiplied by
# (attempt + 1) * random.uniform(0.5, 1.5) per sleep. Widening the base
# extends the total retry budget past a longer OpenRouter mint-propagation
# tail; narrowing it fast-fails bad keys sooner.
#
# Validated at process start rather than per-call: a malformed override
# should fail the container fast, not silently unwind out of the
# per-run try/except and leave a CLAIMED run to go STALE.
#
# Clamped to [0.1, 30.0] to bound the max sleep budget under the 600s
# Backend lease even at the ceiling:
#   base=30, 5 retries → 30*(1+2+3+4+5)*1.5 = 675s worst-case sleep,
#   +6×15s HTTP timeouts = ~765s — over lease.
# The cap keeps the operator lever bounded to a lease-safe window:
#   base=10 → sleep worst-case 450s, +90s timeouts = 540s. Under 600s.
#   base=15 → sleep worst-case 675s → OVER lease. So max is 10-ish.
# Cap 30 leaves headroom for a future lease bump or attribution
# redesign that eliminates the bounded-by-lease requirement.
_TOKEN_401_BACKOFF_BASE_DEFAULT = 5.0
_TOKEN_401_BACKOFF_BASE_MIN = 0.1
_TOKEN_401_BACKOFF_BASE_MAX = 30.0


def _parse_token_401_backoff_base() -> float:
    raw = os.environ.get("ORO_TOKEN_401_BACKOFF_BASE", "").strip()
    if not raw:
        return _TOKEN_401_BACKOFF_BASE_DEFAULT
    try:
        val = float(raw)
    except ValueError as exc:
        raise SystemExit(
            f"ORO_TOKEN_401_BACKOFF_BASE must be numeric (seconds), got {raw!r}"
        ) from exc
    if not (_TOKEN_401_BACKOFF_BASE_MIN <= val <= _TOKEN_401_BACKOFF_BASE_MAX):
        raise SystemExit(
            f"ORO_TOKEN_401_BACKOFF_BASE={val} out of range "
            f"[{_TOKEN_401_BACKOFF_BASE_MIN}, {_TOKEN_401_BACKOFF_BASE_MAX}]"
        )
    return val


TOKEN_401_BACKOFF_BASE = _parse_token_401_backoff_base()


def _parse_env_bool(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in ("true", "1", "yes"):
        return True
    if normalized in ("false", "0", "no"):
        return False
    raise SystemExit(f"{name} must be true or false, got {raw!r}")


def _rewrite_localhost_url(url: str) -> str:
    """Rewrite localhost URLs to host.docker.internal for Docker connectivity."""
    if url.startswith("http://localhost:"):
        return url.replace("http://localhost:", "http://host.docker.internal:", 1)
    return url


def _claim_string(work: ClaimWorkResponse, field_name: str) -> str | None:
    """Read a nullable claim field across old and regenerated SDK models."""

    value = getattr(work, field_name, None)
    if value is None or isinstance(value, Unset):
        value = getattr(work, "additional_properties", {}).get(field_name)
    if value is None:
        return None
    if isinstance(value, UUID):
        return str(value)
    if not isinstance(value, str) or not value:
        raise ValueError(f"claim field {field_name} must be a non-empty string")
    return value


def _claim_environment_binding(
    work: ClaimWorkResponse,
) -> str | None:
    """Return the immutable pack binding supplied by claim work.

    The pack sha is the sole frozen binding — content-addressed, so it cannot
    drift. Returns ``None`` for legacy work with no pack.
    """

    pack_sha256 = _claim_string(work, "env_pack_sha256")
    if pack_sha256 is None:
        return None
    if len(pack_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in pack_sha256
    ):
        raise ValueError("claim field env_pack_sha256 must be 64 lowercase hex chars")
    return pack_sha256


class Validator:
    def __init__(self):
        self.config = self.get_config()
        self.setup_logging()
        check_host_min_specs()
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

        self.session_runtime = SessionRuntime()
        self.session_server = SessionServer(
            self.session_runtime,
            host=self.config.session_runtime_host,
            port=self.config.session_runtime_port,
        )

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
            default=int(os.environ.get("SANDBOX_TIMEOUT") or "1800"),
            help="Timeout in seconds for the entire sandbox subprocess (env: SANDBOX_TIMEOUT, default: 1800 = 30 min).",
        )
        parser.add_argument(
            "--sandbox-log-max-bytes",
            type=int,
            default=int(
                os.environ.get("SANDBOX_LOG_MAX_BYTES") or str(256 * 1024 * 1024)
            ),
            help=(
                "Max bytes captured per sandbox stdout/stderr log before truncation "
                "(env: SANDBOX_LOG_MAX_BYTES, default: 256 MiB). Bounds disk use so a "
                "runaway agent cannot fill the host (ORO-1414)."
            ),
        )
        parser.add_argument(
            "--validator-watchdog-timeout",
            type=float,
            default=float(os.environ.get("VALIDATOR_WATCHDOG_TIMEOUT") or "5400"),
            help=(
                "Seconds the main loop may go without progress before the watchdog "
                "aborts the process so the container restart policy recovers it "
                "(env: VALIDATOR_WATCHDOG_TIMEOUT, default: 5400 = 90 min). Must "
                "exceed the longest legitimate single iteration. 0 disables (ORO-1414)."
            ),
        )
        parser.add_argument(
            "--sandbox-max-workers",
            type=int,
            default=int(os.environ.get("SANDBOX_MAX_WORKERS") or "45"),
            help="Number of parallel problem workers in sandbox (env: SANDBOX_MAX_WORKERS).",
        )
        parser.add_argument(
            "--sandbox-problem-timeout",
            type=float,
            default=float(os.environ.get("SANDBOX_PROBLEM_TIMEOUT") or "300"),
            help="Timeout in seconds per problem in sandbox (env: SANDBOX_PROBLEM_TIMEOUT, default: 300 = 5 min).",
        )
        parser.add_argument(
            "--reasoning-max-workers",
            type=int,
            default=int(os.environ.get("REASONING_MAX_WORKERS") or "8"),
            help="Number of parallel reasoning judge workers (env: REASONING_MAX_WORKERS).",
        )
        parser.add_argument(
            "--session-runtime-host",
            default=os.environ.get("ORO_SESSION_RUNTIME_HOST", "0.0.0.0"),
            help="Internal session HTTP bind host (env: ORO_SESSION_RUNTIME_HOST).",
        )
        parser.add_argument(
            "--session-runtime-port",
            type=int,
            default=int(os.environ.get("ORO_SESSION_RUNTIME_PORT", "9101")),
            help="Internal session HTTP port (env: ORO_SESSION_RUNTIME_PORT).",
        )
        parser.add_argument(
            "--session-tool-timeout",
            type=float,
            default=float(os.environ.get("ORO_SESSION_TOOL_TIMEOUT", "10")),
            help="Per-action timeout before quarantine (env: ORO_SESSION_TOOL_TIMEOUT).",
        )
        parser.add_argument(
            "--session-simulator-timeout",
            type=float,
            default=float(os.environ.get("ORO_SESSION_SIMULATOR_TIMEOUT") or "60"),
            help=(
                "Shopper simulator timeout before quarantine "
                "(env: ORO_SESSION_SIMULATOR_TIMEOUT)."
            ),
        )
        parser.add_argument(
            "--environment-runtime-enabled",
            action=argparse.BooleanOptionalAction,
            default=_parse_env_bool("ORO_ENVIRONMENT_RUNTIME_ENABLED"),
            help=(
                "Enable the sealed environment runtime and its startup preflight "
                "(env: ORO_ENVIRONMENT_RUNTIME_ENABLED, default: false)."
            ),
        )
        parser.add_argument(
            "--environment-preflight-pack-sha256",
            default=os.environ.get("ORO_ENVIRONMENT_PREFLIGHT_PACK_SHA256", ""),
            help=(
                "Sealed pack hash used by the environment startup preflight "
                "(env: ORO_ENVIRONMENT_PREFLIGHT_PACK_SHA256)."
            ),
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
            default=int(os.environ.get("ORO_WEIGHT_UPDATE_INTERVAL", "1320")),
            help=(
                "Seconds between weight-set attempts (env: ORO_WEIGHT_UPDATE_INTERVAL). "
                "Default 1320 (22 min) sits just above the on-chain 20-min "
                "WeightsSetRateLimit so every attempt lands."
            ),
        )
        parser.add_argument(
            "--reveal-period-epochs",
            type=int,
            default=int(os.environ.get("ORO_REVEAL_PERIOD_EPOCHS", "1")),
            help=(
                "Commit-reveal period in epochs, mirroring the on-chain "
                "commit_reveal_period (SN15 = 1). Used to derive the epoch index "
                "for the burn anchor (env: ORO_REVEAL_PERIOD_EPOCHS)."
            ),
        )
        # Adds override arguments for network and netuid.
        parser.add_argument(
            "--netuid", type=int, default=15, help="The chain subnet uid."
        )
        # Adds subtensor specific arguments.
        Subtensor.add_args(parser)
        # bittensor's chain_endpoint flag has no env-var default (unlike our
        # own knobs above). Docker-compose can't emit `--subtensor.chain_endpoint`
        # conditionally in a YAML list, and passing the flag with an empty
        # value makes bittensor route to `("unknown", "")` and crash on connect
        # (see `bittensor.utils.determine_chain_endpoint_and_network`), so
        # inject it into argv here only when the env var is non-empty. Lets
        # validators point at a private subtensor RPC via `.env` without
        # inheriting the shared foundation-finney 429s.
        endpoint = (os.environ.get("SUBTENSOR_CHAIN_ENDPOINT") or "").strip()
        if endpoint and "--subtensor.chain_endpoint" not in sys.argv:
            sys.argv.extend(["--subtensor.chain_endpoint", endpoint])
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

    @staticmethod
    def _simulator_inference_stats_file(eval_run_id_str: str) -> Path:
        """Validator-private counters; the sandbox cannot modify this file."""
        return Path("/tmp") / f"oro-simulator-inference-{eval_run_id_str}.jsonl"

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
        inference_access_token: Optional[str] = None,
        inference_provider: Optional[str] = None,
        inference_base_url: Optional[str] = None,
        stop_event: threading.Event | None = None,
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

        metadata: SandboxMetadata = {
            "exit_code": None,
            "duration_seconds": None,
            "stderr_tail": None,
        }

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
            inference_access_token=inference_access_token,
            inference_provider=inference_provider,
            inference_base_url=inference_base_url,
            agent_container_path="/app/logs/agent.py",
            max_workers=self.config.sandbox_max_workers,
            timeout=self.config.sandbox_problem_timeout,
            container_name=f"oro-sandbox-{eval_run_id}",
        )

        logging.info(f"Running sandbox for eval_run {eval_run_id}")
        log_cmd = [
            arg.split("=")[0] + "=***"
            if any(
                s in arg for s in ("INFERENCE_ACCESS_TOKEN=",)
            )
            else arg
            for arg in cmd
        ]
        logging.info(f"Sandbox command: {' '.join(log_cmd)}")

        SANDBOX_ACTIVE.inc()
        start_time = time.time()
        try:
            return self._run_sandbox_inner(
                cmd=cmd,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                output_file=output_file,
                eval_run_id=eval_run_id,
                metadata=metadata,
                stop_event=stop_event,
            )
        finally:
            duration = time.time() - start_time
            SANDBOX_DURATION_SECONDS.observe(duration)
            metadata["duration_seconds"] = round(duration, 1)
            SANDBOX_ACTIVE.dec()

    def _run_sandbox_inner(
        self,
        *,
        cmd: list[str],
        stdout_log: Path,
        stderr_log: Path,
        output_file: Path,
        eval_run_id: str,
        metadata: SandboxMetadata,
        stop_event: threading.Event | None = None,
    ) -> tuple[Optional[Path], SandboxMetadata]:
        try:
            # Capture through a byte cap so a runaway agent's stdout/stderr can't
            # fill the host disk (ORO-1414); the stream is still fully drained so
            # the sandbox never blocks on a full pipe.
            returncode = run_capped(
                cmd,
                stdout_path=stdout_log,
                stderr_path=stderr_log,
                max_bytes=self.config.sandbox_log_max_bytes,
                timeout=self.config.sandbox_timeout,
                stop_event=stop_event,
            )
            metadata["exit_code"] = returncode

            # Always log sandbox output for debugging
            if stderr_log.exists():
                stderr_content = read_text_lossy(stderr_log)
                if stderr_content.strip():
                    metadata["stderr_tail"] = stderr_content[-500:]
                    log_fn = logging.error if returncode != 0 else logging.info
                    log_fn(
                        f"Sandbox stderr for eval_run {eval_run_id}:\n{stderr_content}"
                    )

            if stdout_log.exists():
                stdout_content = read_text_lossy(stdout_log)
                if stdout_content.strip():
                    logging.info(
                        f"Sandbox stdout for eval_run {eval_run_id}:\n{stdout_content}"
                    )

            if returncode != 0:
                logging.error(
                    f"Sandbox execution failed for eval_run {eval_run_id} (exit code: {returncode})"
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
                    stderr_content = read_text_lossy(stderr_log)
                    if stderr_content.strip():
                        logging.error(f"Sandbox stderr:\n{stderr_content}")
                return None, metadata

        except RunStoppedEarly:
            metadata["exit_code"] = -1
            self._kill_sandbox_container(eval_run_id)
            return None, metadata
        except subprocess.TimeoutExpired:
            metadata["exit_code"] = -1
            # run_capped SIGKILLs the `docker run` CLI, but the container keeps
            # running on the daemon (it's attached, not the client's child), so
            # a timed-out runaway agent would keep burning CPU/network until its
            # own internal timeout. Kill the orphan by name (ORO-1414).
            self._kill_sandbox_container(eval_run_id)
            if stderr_log.exists():
                stderr_content = read_text_lossy(stderr_log)
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
            # run_capped may have already started the container before failing
            # (e.g. opening the cap files raised on a full disk); kill the orphan
            # here too, not just on the timeout path. No-op if it never started.
            self._kill_sandbox_container(eval_run_id)
            return None, metadata

    def _kill_sandbox_container(self, eval_run_id: str) -> None:
        """Best-effort kill of the sandbox container left running after a timeout.

        Matches the ``--name`` set in :func:`build_sandbox_command`. The container
        is ``--rm`` so killing it also removes it. Never raises: if it already
        exited (the common case) ``docker kill`` just errors out harmlessly.
        """
        name = f"oro-sandbox-{eval_run_id}"
        try:
            result = subprocess.run(
                ["docker", "kill", name],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                logging.info(f"Killed orphaned sandbox container {name}")
        except Exception as e:
            logging.warning(f"Failed to kill sandbox container {name}: {e}")

    def _check_for_updates(self):
        """Trigger Watchtower update check and pull sandbox image.

        Called between evaluation cycles. All errors are caught — never crashes the main loop.
        After Watchtower restarts services, waits for proxy /health before returning.
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

        # Wait for proxy to be healthy. Watchtower blocks during restarts but
        # doesn't wait for Docker healthchecks.
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

        self.session_server.start()
        logging.info(
            "Session runtime listening on "
            f"{self.config.session_runtime_host}:{self.config.session_runtime_port}"
        )

        try:
            self._run_environment_preflight_if_enabled()
        except Exception:
            self.session_runtime.clear()
            self.session_server.stop()
            raise

        # Expose a /metrics endpoint for the bundled Prometheus to scrape.
        # Default registry already includes Python process collectors (CPU,
        # memory, fd count, GC). Bound to 0.0.0.0 inside the container; the
        # docker network keeps it off the public internet.
        from prometheus_client import start_http_server

        start_http_server(METRICS_PORT)
        logging.info(f"Prometheus /metrics server listening on :{METRICS_PORT}")

        # Track current execution state for debugging
        self._current_eval_run_id = None

        # Check for updates every 5 minutes even when idle
        UPDATE_CHECK_INTERVAL = 300
        self._last_update_check = 0

        weight_setter = WeightSetterThread(
            backend_client=self.backend_client,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
            wallet=self.wallet,
            netuid=self.config.netuid,
            interval_seconds=self.config.weight_update_interval,
            reveal_period_epochs=self.config.reveal_period_epochs,
        )
        weight_setter.start()
        logging.info(
            f"Weight setter started (interval: {self.config.weight_update_interval}s)"
        )

        # Recover a wedged validator: if the loop stops making progress (e.g. the
        # host disk fills and an iteration hangs), abort so the container's
        # restart policy brings a fresh one up (ORO-1414).
        watchdog: Optional[ProgressWatchdog] = None
        if self.config.validator_watchdog_timeout > 0:
            watchdog = ProgressWatchdog(self.config.validator_watchdog_timeout)
            watchdog.start()
            logging.info(
                f"Watchdog started (timeout: {self.config.validator_watchdog_timeout}s)"
            )

        self._drain_state: dict = {}

        try:
            while True:
                if watchdog is not None:
                    watchdog.beat()
                try:
                    # Drain check sits ABOVE auto-update so a Watchtower
                    # image roll can't restart mid-drain (ORO-1150).
                    if handle_drain_tick(
                        self._drain_state,
                        self.retry_queue,
                        self.config.poll_interval,
                    ):
                        continue

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
            self.session_runtime.clear()
            self.session_server.stop()
            if watchdog is not None:
                watchdog.stop()
            weight_setter.stop()
            logging.info("Validator stopped.")

    def _run_environment_preflight_if_enabled(self) -> dict[str, Any] | None:
        """Run the default-off environment gate before claiming miner work."""

        if not self.config.environment_runtime_enabled:
            logging.info("Environment runtime feature flag is disabled")
            return None

        pack_sha256 = self.config.environment_preflight_pack_sha256
        if len(pack_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in pack_sha256
        ):
            raise RuntimeError(
                "ORO_ENVIRONMENT_PREFLIGHT_PACK_SHA256 must be 64 lowercase hex "
                "characters when ORO_ENVIRONMENT_RUNTIME_ENABLED=true"
            )

        loaded_pack = asyncio.run(
            fetch_and_validate_pack(
                pack_sha256,
                self.config.backend_url,
                self.wallet.hotkey,
                download_url_rewriter=_rewrite_localhost_url,
            )
        )
        if loaded_pack is None:
            raise RuntimeError(
                f"environment preflight pack {pack_sha256} failed validation"
            )

        run_dir = self._eval_dir(environment_preflight_run_id(pack_sha256))
        logging.info(
            f"Running feature-flagged environment preflight for pack={pack_sha256}"
        )
        summary = run_environment_preflight(
            loaded_pack=loaded_pack,
            runtime=self.session_runtime,
            run_dir=run_dir,
            sandbox_runner=self.run_sandbox,
        )
        logging.info(
            f"Environment preflight receipt: {json.dumps(summary, sort_keys=True)}"
        )
        logging.info(f"Environment preflight passed: pack={pack_sha256}")
        return summary

    def prepare_environment_sessions(
        self,
        work: ClaimWorkResponse,
        *,
        inference_access_token: str,
    ) -> SessionRegistry | None:
        """Provision optional sealed sessions alongside the legacy evaluation."""

        if not self.config.environment_runtime_enabled:
            return None

        try:
            binding = _claim_environment_binding(work)
        except ValueError as exc:
            logging.warning(
                f"Invalid environment binding for evaluation {work.eval_run_id}: {exc}"
            )
            return None
        if binding is None:
            return None
        try:
            registry, _, _ = self._create_environment_sessions(
                work,
                inference_access_token=inference_access_token,
            )
            return registry
        except Exception as exc:
            logging.warning(
                f"Environment session setup failed for evaluation {work.eval_run_id}: "
                f"{type(exc).__name__}: {exc}"
            )
            return None

    def _create_environment_sessions(
        self,
        work: ClaimWorkResponse,
        *,
        inference_access_token: str,
    ) -> tuple[SessionRegistry, list[dict[str, Any]], dict[str, str]]:
        """Create the bound sessions, raising when authoritative setup fails."""

        pack_sha256 = _claim_environment_binding(work)
        if pack_sha256 is None:
            raise ValueError("generated evaluation requires a bound env pack")

        # Race work items get the race-scoped sub-archive (only the
        # 60 selected task specs) instead of the pack's qualifying sub-archive.
        # The race pack is presigned only to validators with an active
        # EvaluationRun on the race, so this fetch fails cleanly if the run
        # has expired between claim and load.
        race_id = _claim_string(work, "race_id")
        if race_id is not None:
            loaded_pack = asyncio.run(
                fetch_and_validate_race_pack(
                    race_id,
                    pack_sha256,
                    self.config.backend_url,
                    self.wallet.hotkey,
                    download_url_rewriter=_rewrite_localhost_url,
                )
            )
            if loaded_pack is None:
                raise RuntimeError(
                    f"race pack race_id={race_id} pack={pack_sha256} failed validation"
                )
        else:
            loaded_pack = asyncio.run(
                fetch_and_validate_pack(
                    pack_sha256,
                    self.config.backend_url,
                    self.wallet.hotkey,
                    download_url_rewriter=_rewrite_localhost_url,
                )
            )
            if loaded_pack is None:
                raise RuntimeError(f"environment pack {pack_sha256} failed validation")

        registry: SessionRegistry | None = None
        try:
            selected_roster = select_run_task_roster(
                self.backend_client.get_run_problems(work.eval_run_id),
                {
                    task_id: task.family
                    for task_id, task in zip(
                        loaded_pack.task_ids, loaded_pack.task_specs, strict=True
                    )
                },
            )
            registry = SessionRegistry(
                loaded_pack,
                tool_timeout_s=self.config.session_tool_timeout,
                simulator_timeout_s=self.config.session_simulator_timeout,
                max_workers=self.config.sandbox_max_workers,
                inference_access_token=inference_access_token,
                inference_stats_file=str(
                    self._simulator_inference_stats_file(str(work.eval_run_id))
                ),
            )
            sessions = [
                registry.start(
                    evaluation_run_id=str(work.eval_run_id),
                    agent_version_id=str(work.agent_version_id),
                    task_id=task_id,
                )
                for task_id in selected_roster
            ]
            session_file = self._eval_dir(str(work.eval_run_id)) / (
                "environment_sessions.json"
            )
            session_file.write_text(
                json.dumps(
                    {
                        "schema_version": "oro.session_bootstrap.v1",
                        "sessions": sessions,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            os.chmod(session_file, 0o444)
            self.session_runtime.install(registry)
            logging.info(
                f"Provisioned {len(sessions)} sealed sessions for evaluation "
                f"{work.eval_run_id}"
            )
            return registry, sessions, selected_roster
        except Exception:
            if registry is None:
                loaded_pack.close()
            else:
                self.session_runtime.clear(registry)
                registry.close()
            raise

    def _emit_environment_results(
        self,
        work: ClaimWorkResponse,
        registry: SessionRegistry,
        *,
        expected_task_roster: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        results = registry.finalized_results()
        self._emit_environment_result_batch(
            work,
            results,
            expected_task_roster,
            require_complete=True,
        )
        return results

    def _emit_environment_result_batch(
        self,
        work: ClaimWorkResponse,
        results: list[dict[str, Any]],
        expected_task_roster: dict[str, str] | None = None,
        *,
        require_complete: bool = False,
    ) -> None:
        env_pack_sha256 = _claim_environment_binding(work)
        if env_pack_sha256 is None:
            raise ValueError("environment binding disappeared during evaluation")
        if expected_task_roster is not None:
            validate_run_results(
                results,
                expected_task_roster,
                evaluation_run_id=str(work.eval_run_id),
                agent_version_id=str(work.agent_version_id),
                pack_sha256=env_pack_sha256,
                require_complete=require_complete,
            )
        submitted = asyncio.run(
            emit_finalized_results(
                backend_url=self.config.backend_url,
                validator_keypair=self.wallet.hotkey,
                env_pack_sha256=env_pack_sha256,
                results=results,
                download_url_rewriter=_rewrite_localhost_url,
            )
        )
        logging.info(
            "Persisted generated-environment progress for "
            f"{work.eval_run_id}: {submitted.get('counts', {})}"
        )

    def finalize_environment_sessions(
        self,
        work: ClaimWorkResponse,
        registry: SessionRegistry,
    ) -> None:
        """Persist all environment outcomes without changing legacy scoring."""

        try:
            self._emit_environment_results(work, registry)
        except Exception as exc:
            # Episode capture is preview telemetry. A transport or schema defect
            # must never change the legacy evaluation's score or completion path.
            logging.warning(
                "Generated-environment episode emission failed for "
                f"{work.eval_run_id}: {type(exc).__name__}: {exc}"
            )
        finally:
            try:
                self.session_runtime.clear(registry)
            except Exception as exc:
                logging.warning(
                    "Generated-environment session cleanup failed for "
                    f"{work.eval_run_id}: {type(exc).__name__}: {exc}"
                )

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

    def _run_claimed_evaluation(
        self,
        work: ClaimWorkResponse,
        agent_path: Path,
        *,
        inference_access_token: str,
        inference_provider: str,
        inference_base_url: str,
    ) -> _EvaluationCompletion | None:
        """Select the evaluator from the immutable contract on claimed work."""

        runner = (
            self._run_generated_evaluation
            if _claim_environment_binding(work) is not None
            else self._run_legacy_evaluation
        )
        return runner(
            work,
            agent_path,
            inference_access_token=inference_access_token,
            inference_provider=inference_provider,
            inference_base_url=inference_base_url,
        )

    def _run_generated_evaluation(
        self,
        work: ClaimWorkResponse,
        agent_path: Path,
        *,
        inference_access_token: str,
        inference_provider: str,
        inference_base_url: str,
    ) -> _EvaluationCompletion | None:
        """Execute and score the sealed pack bound to the claimed work."""

        eval_run_id = work.eval_run_id
        eval_run_id_str = str(eval_run_id)
        registry, sessions, selected_roster = self._create_environment_sessions(
            work,
            inference_access_token=inference_access_token,
        )
        sandbox_output = None
        sandbox_metadata: SandboxMetadata = {}
        reporter = GeneratedProgressReporter(
            registry,
            lambda batch: self._emit_environment_result_batch(
                work, batch, selected_roster
            ),
        )
        try:
            reporter.start()
            problem_file = self._eval_dir(eval_run_id_str) / "problems.jsonl"
            write_problem_file(problem_file, sessions)
            try:
                sandbox_output, sandbox_metadata = self.run_sandbox(
                    agent_path,
                    eval_run_id_str,
                    problem_file,
                    inference_access_token=inference_access_token,
                    inference_provider=inference_provider,
                    inference_base_url=inference_base_url,
                    stop_event=registry.key_exhausted,
                )
            finally:
                reporter.stop()
                results = registry.finalized_results()
                env_pack_sha256 = _claim_environment_binding(work)
                if env_pack_sha256 is None:
                    raise ValueError(
                        "environment binding disappeared before finalization"
                    )
                validate_run_results(
                    results,
                    selected_roster,
                    evaluation_run_id=str(work.eval_run_id),
                    agent_version_id=str(work.agent_version_id),
                    pack_sha256=env_pack_sha256,
                )
                try:
                    reporter.flush(results)
                except Exception as exc:
                    logging.warning(
                        "Final generated progress batch failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
        finally:
            reporter.stop()
            self.session_runtime.clear(registry)

        if registry.key_exhausted.is_set():
            self._complete_with_failure(
                eval_run_id,
                TerminalStatus.FAILED,
                "Miner inference key exhausted",
                sandbox_metadata=sandbox_metadata,
            )
            return None

        if not sandbox_output:
            self._complete_with_failure(
                eval_run_id,
                TerminalStatus.FAILED,
                "Sandbox execution failed",
                sandbox_metadata=sandbox_metadata,
            )
            return None

        try:
            score = aggregate_results(results)
        except ValueError as exc:
            self._complete_with_failure(
                eval_run_id,
                TerminalStatus.FAILED,
                str(exc),
                sandbox_metadata=sandbox_metadata,
            )
            return None
        logging.info(
            f"Generated evaluation score: {score:.6f} across {len(results)} tasks"
        )
        try:
            eval_dir = self._eval_dir(eval_run_id_str)
            output_stats = read_inference_stats(str(sandbox_output))
            sidecar_stats = read_inference_stats(
                str(eval_dir / "inference_stats.jsonl")
            )
            # The runner-owned output snapshot survives even when the shared
            # append-only sidecar is absent at finalization. Prefer it per
            # episode so the same agent calls are never counted twice.
            agent_stats = {
                problem_id: output_stats.get(problem_id, usage)
                for problem_id, usage in sidecar_stats.items()
            }
            agent_stats.update(output_stats)
            simulator_stats = read_inference_stats(
                str(self._simulator_inference_stats_file(eval_run_id_str))
            )
            by_episode = summarize_episode_resource_usage(
                results,
                merge_inference_stats(agent_stats, simulator_stats),
                inference_provider,
            )
        except Exception:
            logging.warning(
                "Unable to collect shadow episode inference usage",
                exc_info=True,
            )
        else:
            sandbox_metadata = dict(sandbox_metadata)
            sandbox_metadata["_shadow_resource_usage"] = {
                "by_episode": by_episode
            }
        return _EvaluationCompletion(
            score=score,
            score_components={"schema_version": GENERATED_SCORE_SCHEMA},
            results_s3_key="",
            sandbox_metadata=sandbox_metadata,
        )

    def _run_legacy_evaluation(
        self,
        work: ClaimWorkResponse,
        agent_path: Path,
        *,
        inference_access_token: str,
        inference_provider: str,
        inference_base_url: str,
    ) -> _EvaluationCompletion | None:
        """Run the existing ShoppingBench evaluation unchanged."""

        eval_run_id = work.eval_run_id
        eval_run_id_str = str(eval_run_id)
        problem_file, problem_ids, problems = self.fetch_problems(
            work.suite_id, eval_run_id_str
        )
        if not problem_file or not problems:
            self._complete_with_failure(
                eval_run_id, TerminalStatus.FAILED, "Failed to load problems"
            )
            return None

        eval_dir = self._eval_dir(eval_run_id_str)
        output_file = eval_dir / "output.jsonl"
        progress_reporter = ProgressReporter(
            backend_client=self.backend_client,
            eval_run_id=eval_run_id,
            output_file=output_file,
            problems=problems,
            workspace_dir=Path(self.config.workspace_dir),
            inference_access_token=inference_access_token,
            inference_provider=inference_provider,
            max_scoring_workers=self.config.reasoning_max_workers,
        )
        progress_reporter.start_monitoring()
        environment_registry = self.prepare_environment_sessions(
            work,
            inference_access_token=inference_access_token,
        )

        try:
            sandbox_output, sandbox_metadata = self.run_sandbox(
                agent_path,
                eval_run_id_str,
                problem_file,
                inference_access_token=inference_access_token,
                inference_provider=inference_provider,
                inference_base_url=inference_base_url,
            )
        finally:
            try:
                progress_reporter.signal_sandbox_done()
                progress_reporter.wait_for_completion()
            finally:
                if environment_registry is not None:
                    self.finalize_environment_sessions(work, environment_registry)

        if not sandbox_output:
            self._complete_with_failure(
                eval_run_id,
                TerminalStatus.FAILED,
                "Sandbox execution failed",
                sandbox_metadata=sandbox_metadata,
            )
            return None

        aggregate = progress_reporter.get_aggregate_score()
        if aggregate is None:
            self._complete_with_failure(
                eval_run_id,
                TerminalStatus.FAILED,
                "ProgressReporter did not compute aggregate score",
            )
            return None

        success_rate = aggregate.get("success_rate", 0.0)
        # Missing or failed judgments count as zero. Incomplete reasoning-judge
        # coverage does not fail the run because the judge is being retired.
        reasoning_result = progress_reporter.get_reasoning_data()
        score = blend_final_score(success_rate, reasoning_result["reasoning_quality"])
        aggregate.update(reasoning_result)
        logging.info(
            f"Score: final={score:.4f} "
            f"(success_rate={success_rate:.4f} * "
            f"coefficient={reasoning_result['reasoning_coefficient']:.4f}, "
            f"reasoning_quality={reasoning_result['reasoning_quality']:.4f})"
        )
        results_s3_key = self._upload_logs(
            eval_run_id, output_file, problem_ids, progress_reporter
        )
        return _EvaluationCompletion(
            score=score,
            score_components=aggregate,
            results_s3_key=results_s3_key,
            sandbox_metadata=sandbox_metadata,
        )

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

        agent_path = None
        sandbox_metadata: SandboxMetadata | None = None

        # Step 0: Verify miner inference token is present and valid
        if (
            not inference_access_token
            or not inference_base_url
            or not inference_provider
        ):
            self._complete_with_failure(
                eval_run_id,
                TerminalStatus.FAILED,
                "Miner has no inference token — cannot fund inference",
            )
            return

        # Validate the token can actually make inference calls
        token_valid, token_reason = self._validate_inference_token(
            inference_access_token,
            inference_base_url,
            self._validation_model_for(inference_provider),
        )
        if not token_valid:
            self._complete_with_failure(
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
            agent_path = self.download_agent(work.code_download_url, eval_run_id_str)
            if not agent_path:
                self._complete_with_failure(
                    eval_run_id, TerminalStatus.FAILED, "Download failed"
                )
                return

            completion = self._run_claimed_evaluation(
                work,
                agent_path,
                inference_access_token=inference_access_token,
                inference_provider=inference_provider,
                inference_base_url=inference_base_url,
            )
            if completion is None:
                return
            sandbox_metadata = completion.sandbox_metadata

            self._complete_run(
                eval_run_id=eval_run_id,
                status=TerminalStatus.SUCCESS,
                score=completion.score,
                score_components=completion.score_components,
                results_s3_key=completion.results_s3_key,
                sandbox_metadata=completion.sandbox_metadata,
            )

        except Exception as e:
            logging.error(f"Evaluation cycle failed: {e}")
            traceback.print_exc()
            self._complete_with_failure(
                eval_run_id,
                TerminalStatus.FAILED,
                str(e),
                sandbox_metadata=sandbox_metadata,
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
            try:
                self._simulator_inference_stats_file(eval_run_id_str).unlink(
                    missing_ok=True
                )
            except OSError as e:
                logging.debug(f"Simulator stats cleanup failed: {e}")

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

            problem_lines = split_output_by_problem(output_file, problem_ids)

            # Build (pid, compressed_bytes) tuples up front so the worker
            # function is pure I/O. Invalid problem_ids are dropped here
            # before they reach the executor.
            jobs: list[tuple[UUID, bytes]] = []
            for pid_str, line_data in problem_lines.items():
                try:
                    pid = UUID(pid_str)
                except ValueError:
                    logging.warning(
                        f"Invalid problem_id in output: {pid_str}, skipping"
                    )
                    continue
                jobs.append((pid, gzip.compress(line_data)))

            def _upload_one(job: tuple[UUID, bytes]) -> tuple[UUID, str] | None:
                pid, compressed = job
                try:
                    presign = self.backend_client.get_presigned_upload_url(
                        content_length=len(compressed),
                        eval_run_id=eval_run_id,
                        problem_id=pid,
                    )
                    # Rewrite localhost URLs for Docker → host connectivity
                    if hasattr(presign, "upload_url"):
                        presign.upload_url = _rewrite_localhost_url(presign.upload_url)
                    self.backend_client.upload_to_s3(presign, compressed)
                    return pid, presign.results_s3_key
                except Exception as e:
                    logging.warning(f"Upload failed for problem {pid}: {e}")
                    return None

            # Parallel uploads. Sized to match the BackendClient S3 session's
            # pool_connections so we don't queue inside urllib3. Each worker
            # does one presign roundtrip + one S3 PUT; both calls reuse the
            # connection pool, so per-problem wall time drops from ~600 ms
            # to ~150-200 ms.
            uploaded_keys: dict[UUID, str] = {}
            last_s3_key = ""
            if jobs:
                with ThreadPoolExecutor(
                    max_workers=_UPLOAD_LOGS_WORKERS,
                    thread_name_prefix="upload-logs",
                ) as pool:
                    for result in pool.map(_upload_one, jobs):
                        if result is None:
                            continue
                        pid, s3_key = result
                        uploaded_keys[pid] = s3_key
                        last_s3_key = s3_key
                        logging.info(f"Uploaded logs to {s3_key}")

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
    def _validation_model_for(provider: str) -> str:
        """Pick a small model present on each provider for the smoke-test."""
        if provider == "chutes":
            return "Qwen/Qwen3-32B-TEE"
        if provider == "openrouter":
            return "openai/gpt-oss-20b"
        raise ValueError(f"unknown inference provider: {provider}")

    @staticmethod
    def _validate_inference_token(
        access_token: str, base_url: str, model: str
    ) -> tuple[bool, str]:
        """Smoke-test a minted inference token by making a 1-token completion.

        Catches both invalid tokens (401) and zero-balance accounts (402)
        against any OpenAI-compatible chat/completions endpoint. On
        transient errors (5xx, timeout, 429), returns (True, "") to avoid
        failing runs unnecessarily.

        A freshly-minted scoped token can return 401 briefly while the
        provider propagates the newly-created key. The token is valid, so a
        401 is retried a few times before the run is failed. A genuinely
        invalid/revoked key stays 401 through every attempt and still fails
        fast, after which the run re-claims a fresh token.

        Retry policy: exponential backoff with jitter. The exponential
        curve rejects genuinely-bad keys quickly on the early attempts
        while extending total budget past the observed propagation tail.
        Jitter spreads the mint-burst so a race full of validators
        doesn't all retry at the same wall-clock instant and hit the
        propagation window in lockstep.
        """
        # Only the 401 path loops; every other outcome returns on first attempt.
        # Sleeps between retries: base * (attempt + 1) * random(0.5, 1.5).
        # attempt 0 → 5.0s ± 50% ≈ 2.5–7.5s
        # attempt 1 → 10.0s ± 50% ≈ 5.0–15.0s
        # attempt 2 → 15.0s ± 50% ≈ 7.5–22.5s
        # attempt 3 → 20.0s ± 50% ≈ 10.0–30.0s
        # attempt 4 → 25.0s ± 50% ≈ 12.5–37.5s
        # Sleep budget worst-case: ~113s. Absolute ceiling if every
        # HTTP attempt also hits the 15s timeout: 6 × 15 + 113 = ~203s.
        # Fits comfortably under the 600s Backend lease. Typical bad-key
        # early-exit stays sub-second (401 comes back fast, next sleep
        # is 2.5–7.5s before attempt 1). Typical propagation-blip resolve:
        # 5–30s across attempts 1–3.
        #
        # Base bumped 1.5s → 5.0s (default) to widen the retry budget
        # past the observed OpenRouter mint-propagation tail — the
        # consecutive-failures alarm was firing on runs whose 401 storm
        # outlasted the prior ~34s ceiling. `ORO_TOKEN_401_BACKOFF_BASE`
        # env override lets on-call retune during a live incident
        # without a code change + image roll — validated at process
        # start (see `_parse_token_401_backoff_base`), so a malformed
        # value fails the container fast rather than silently unwinding
        # a claimed run into STALE.
        #
        # Distinguishing a fresh-mint propagation blip from a revoked
        # key is out of scope here — that attribution work is the real
        # ORO-1597 redesign. This bump is the operational lever until
        # that lands.
        max_401_retries = 5
        backoff_base = TOKEN_401_BACKOFF_BASE
        jitter_low, jitter_high = 0.5, 1.5

        url = f"{base_url.rstrip('/')}/chat/completions"
        for attempt in range(max_401_retries + 1):
            try:
                resp = requests.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                    },
                    timeout=15,
                )
                if resp.status_code == 200:
                    return True, ""
                if resp.status_code == 401:
                    try:
                        body = (resp.text or "")[:500]
                    except Exception:
                        body = ""
                    if attempt < max_401_retries:
                        sleep_for = (
                            backoff_base
                            * (attempt + 1)
                            * random.uniform(jitter_low, jitter_high)
                        )
                        logging.warning(
                            f"Inference token 401 (attempt {attempt + 1}/"
                            f"{max_401_retries + 1}) — likely provider "
                            f"key-propagation lag, retrying in {sleep_for:.1f}s "
                            f"body={body!r}"
                        )
                        time.sleep(sleep_for)
                        continue
                    logging.warning(
                        f"Inference token 401 on final attempt "
                        f"({max_401_retries + 1}/{max_401_retries + 1}), "
                        f"giving up — body={body!r}"
                    )
                    return False, "Inference token invalid or expired (HTTP 401)"
                if resp.status_code == 402:
                    detail = resp.json().get("detail", {})
                    msg = (
                        detail.get("message", str(detail))
                        if isinstance(detail, dict)
                        else str(detail)
                    )
                    return False, f"Inference account has no credits ({msg})"
                if resp.status_code == 429:
                    return True, ""
                logging.warning(
                    "Inference token validation inconclusive: status=%s url=%s",
                    resp.status_code,
                    url,
                )
                return True, ""
            except Exception as exc:
                logging.warning(
                    "Inference token validation error against %s: %s", url, exc
                )
                return True, ""

        # Unreachable: the loop returns on every path, but keeps mypy happy.
        return False, "Inference token invalid or expired (HTTP 401)"

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
