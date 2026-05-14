"""Agent download + sandbox subprocess execution.

Owns the on-disk dance for one evaluation: pulling the agent code from the
presigned URL into a per-eval directory and running the Docker sandbox over
it. Streams stdout/stderr to log files, returns the output JSONL path and
sandbox metadata (exit_code, duration, stderr tail).
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

import requests
from bittensor.core.config import Config
from bittensor.utils.btlogging import logging

from src.agent.types import SandboxMetadata
from subnet.sandbox import host_path, build_sandbox_command

from .metrics import SANDBOX_ACTIVE, SANDBOX_DURATION_SECONDS
from .url_utils import rewrite_localhost_url


class SandboxRunner:
    """Download agents and run them in the docker sandbox.

    `eval_dir_for` is a callable returning the per-evaluation directory for an
    eval_run_id — injected so this class doesn't need to know the workspace
    layout.
    """

    def __init__(self, config: Config, eval_dir_for: Callable[[str], Path]):
        self.config = config
        self.eval_dir_for = eval_dir_for

    def download_agent(self, url: str, eval_run_id: str) -> Optional[Path]:
        """Download agent file from URL to per-evaluation directory.

        Returns the file path on success, None on failure.
        """
        try:
            url = rewrite_localhost_url(url)
            logging.info(f"Downloading agent from {url} for eval_run {eval_run_id}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            eval_dir = self.eval_dir_for(eval_run_id)
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
    ) -> tuple[Optional[Path], SandboxMetadata]:
        """Run sandbox with the downloaded agent.

        Returns (output_file or None, sandbox metadata). Metadata always
        carries exit_code, duration_seconds, and stderr_tail.
        """
        eval_dir = self.eval_dir_for(eval_run_id)
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
        )

        logging.info(f"Running sandbox for eval_run {eval_run_id}")
        log_cmd = [
            arg.split("=")[0] + "=***"
            if any(
                s in arg for s in ("CHUTES_ACCESS_TOKEN=", "INFERENCE_ACCESS_TOKEN=")
            )
            else arg
            for arg in cmd
        ]
        logging.info(f"Sandbox command: {' '.join(log_cmd)}")

        SANDBOX_ACTIVE.inc()
        start_time = time.time()
        try:
            return self._run_inner(
                cmd=cmd,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                output_file=output_file,
                eval_run_id=eval_run_id,
                metadata=metadata,
            )
        finally:
            duration = time.time() - start_time
            SANDBOX_DURATION_SECONDS.observe(duration)
            metadata["duration_seconds"] = round(duration, 1)
            SANDBOX_ACTIVE.dec()

    def _run_inner(
        self,
        *,
        cmd: list[str],
        stdout_log: Path,
        stderr_log: Path,
        output_file: Path,
        eval_run_id: str,
        metadata: SandboxMetadata,
    ) -> tuple[Optional[Path], SandboxMetadata]:
        try:
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
            metadata["exit_code"] = result.returncode

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
                # output file. Return the file so those results are scored.
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
