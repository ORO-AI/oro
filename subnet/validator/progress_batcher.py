"""Batches per-problem progress updates and reports them to the backend.

`maybe_report()` is the throttled entry point called every loop tick;
`batch_report()` is the unconditional flush for end-of-run and forced
checkpoints. Both build their payload from the shared results dict under
ProgressReporter's lock.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional
from uuid import UUID

import requests
from bittensor.utils.btlogging import logging

from oro_sdk.models import ProblemProgressUpdate
from oro_sdk.types import UNSET, Unset

from src.agent.types import ScoreComponentsSummary

from .backend_client import BackendClient, BackendError
from .types import ProblemResult


# Report to backend at most every N seconds
REPORT_INTERVAL_SECONDS = 10.0


def problem_result_to_update(
    result: ProblemResult, logs_s3_key: str | Unset = UNSET
) -> ProblemProgressUpdate:
    """Build a ProblemProgressUpdate from a scored ProblemResult.

    Shared by ProgressBatcher.batch_report (periodic score push) and
    Validator._upload_logs (post-scoring flush that also carries the
    per-problem log S3 key). Backend upsert is idempotent so calling both
    is safe — this guards against ProgressBatcher silently dropping scores
    on fast evals (see progress_batcher.py:108).
    """
    scs: Optional[ScoreComponentsSummary] = None
    if result.reasoning_score is not None:
        scs = {
            "reasoning_explanation": result.reasoning_explanation,
            "reasoning_model": result.reasoning_model,
        }
    return ProblemProgressUpdate(
        problem_id=UUID(result.problem_id),
        status=result.status,
        score=result.score,
        reasoning_score=result.reasoning_score,
        score_components_summary=scs,
        inference_failure_count=result.inference_failures
        if result.inference_total > 0
        else None,
        inference_total=result.inference_total if result.inference_total > 0 else None,
        execution_time=result.execution_time,
        logs_s3_key=logs_s3_key,
    )


class ProgressBatcher:
    """Periodic, lock-aware batch reporter to the backend."""

    def __init__(
        self,
        backend_client: BackendClient,
        eval_run_id: UUID,
        total_problems: int,
        results: Dict[str, ProblemResult],
        lock: threading.Lock,
        report_interval: float = REPORT_INTERVAL_SECONDS,
    ):
        self._backend_client = backend_client
        self._eval_run_id = eval_run_id
        self._total_problems = total_problems
        self._results = results
        self._lock = lock
        self._report_interval = report_interval
        self._last_report_time = 0.0
        self._last_reported_count = 0

    def reset(self) -> None:
        """Reset rolling state at start_monitoring()."""
        self._last_report_time = 0.0
        self._last_reported_count = 0

    def maybe_report(self) -> None:
        """Send a batch if at least one new result exists and the interval elapsed."""
        with self._lock:
            current_count = len(self._results)

        if current_count == self._last_reported_count:
            return

        now = time.time()
        if now - self._last_report_time >= self._report_interval:
            self.batch_report()
            self._last_report_time = now
            self._last_reported_count = current_count

    def batch_report(self) -> None:
        """Send all accumulated results to backend in one request."""
        with self._lock:
            results = list(self._results.values())

        if not results:
            return

        updates = [problem_result_to_update(r) for r in results]

        try:
            self._backend_client.report_progress(self._eval_run_id, updates)
            logging.info(
                f"Batch reported {len(updates)}/{self._total_problems} problems"
            )
        except (BackendError, requests.RequestException) as e:
            logging.warning(f"Batch report failed ({len(updates)} problems): {e}")
