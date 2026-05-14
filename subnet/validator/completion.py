"""Log upload + run-completion reporting (with retry-queue fallback).

The validator finishes a run in two steps:
  1. Split the sandbox output JSONL by problem, gzip each, upload to S3, and
     report the resulting `logs_s3_key` per problem so the Frontend can fetch
     trajectories per-problem.
  2. Call the Backend `complete_run` endpoint with the final score (success
     case) or failure reason (failure case). Transient Backend failures get
     queued on the local retry queue rather than dropping the run.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import UUID

from bittensor.utils.btlogging import logging
from oro_sdk.models.problem_progress_update import ProblemProgressUpdate
from oro_sdk.models.terminal_status import TerminalStatus

from src.agent.types import SandboxMetadata

from .backend_client import BackendClient, BackendError
from .models import CompletionRequest
from .output_split import split_output_by_problem
from .progress_reporter import ProgressReporter
from .retry_queue import LocalRetryQueue
from .url_utils import rewrite_localhost_url


class CompletionReporter:
    """Reports terminal evaluation state (success or failure) to the Backend.

    Holds references to the backend client and retry queue so failure paths
    don't need to thread them through every call site.
    """

    def __init__(self, backend_client: BackendClient, retry_queue: LocalRetryQueue):
        self.backend_client = backend_client
        self.retry_queue = retry_queue

    def upload_logs(
        self,
        eval_run_id: UUID,
        output_file: Path,
        problem_ids: list[UUID],
        progress_reporter: ProgressReporter,
    ) -> str:
        """Split output JSONL by problem, gzip+upload each, report keys.

        Returns the last successfully uploaded S3 key (stored on the run).
        """
        try:
            if not output_file.exists():
                logging.warning(f"Output file not found: {output_file}")
                return ""

            if not problem_ids:
                logging.warning("No problem_ids available for log upload, skipping")
                return ""

            problem_lines = split_output_by_problem(output_file, problem_ids)

            last_s3_key = ""
            uploaded_keys: dict[UUID, str] = {}
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

                if hasattr(presign, "upload_url"):
                    presign.upload_url = rewrite_localhost_url(presign.upload_url)

                self.backend_client.upload_to_s3(presign, compressed)
                logging.info(f"Uploaded logs to {presign.results_s3_key}")
                last_s3_key = presign.results_s3_key
                uploaded_keys[pid] = presign.results_s3_key

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

    def complete_run(
        self,
        eval_run_id: UUID,
        status: TerminalStatus,
        score: float,
        results_s3_key: str = "",
        score_components: Optional[Dict[str, Any]] = None,
        sandbox_metadata: Optional[SandboxMetadata] = None,
    ) -> None:
        """Report a successful evaluation. Transient errors → retry queue."""
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

    def complete_with_failure(
        self,
        eval_run_id: UUID,
        status: TerminalStatus,
        reason: str,
        sandbox_metadata: Optional[SandboxMetadata] = None,
    ) -> None:
        """Report a failed evaluation. Transient errors → retry queue."""
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
