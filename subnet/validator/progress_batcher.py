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

from src.agent.types import ScoreComponentsSummary

from .backend_client import BackendClient, BackendError
from .types import ProblemResult


# Report to backend at most every N seconds
REPORT_INTERVAL_SECONDS = 10.0


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

        updates = []
        for r in results:
            # Include per-problem reasoning data if judge ran
            scs: Optional[ScoreComponentsSummary] = None
            if r.reasoning_score is not None:
                scs = {
                    "reasoning_explanation": r.reasoning_explanation,
                    "reasoning_model": r.reasoning_model,
                }

            update = ProblemProgressUpdate(
                problem_id=UUID(r.problem_id),
                status=r.status,
                score=r.score,
                reasoning_score=r.reasoning_score,
                score_components_summary=scs,
                inference_failure_count=r.inference_failures
                if r.inference_total > 0
                else None,
                inference_total=r.inference_total if r.inference_total > 0 else None,
                execution_time=r.execution_time,
            )
            updates.append(update)

        try:
            self._backend_client.report_progress(self._eval_run_id, updates)
            logging.info(
                f"Batch reported {len(updates)}/{self._total_problems} problems"
            )
        except (BackendError, requests.RequestException) as e:
            logging.warning(f"Batch report failed ({len(updates)} problems): {e}")
