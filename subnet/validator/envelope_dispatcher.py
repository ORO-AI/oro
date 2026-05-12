"""Sandbox envelope-stream reader.

Translates raw envelope records from OutputWatcher into actions:

- SUCCESS  → dispatch the dialogue to the scoring pool
- FAILED / TIMED_OUT → write a terminal ProblemResult directly to results
- End-of-run sweep → mark never-seen problems TIMED_OUT

Holds no scoring state of its own; envelope metadata, results, and the
id-to-problem map are owned by ProgressReporter and accessed under its
shared lock.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Dict, Optional

from bittensor.utils.btlogging import logging

from oro_sdk.models import ProblemStatus

from src.agent.sandbox_status import SandboxProblemStatus
from src.agent.types import ProblemDict

from .output_watcher import ErrorInfo, OutputWatcher
from .scoring_pool import ScoringPool
from .types import EnvelopeMeta, ProblemResult


class EnvelopeDispatcher:
    """Reads sandbox envelope stream and routes each record."""

    def __init__(
        self,
        watcher: OutputWatcher,
        results: Dict[str, ProblemResult],
        envelope_meta: Dict[str, EnvelopeMeta],
        id_to_problem: Dict[str, ProblemDict],
        lock: threading.Lock,
        scoring_pool: ScoringPool,
        output_file: Path,
    ):
        self._watcher = watcher
        self._results = results
        self._envelope_meta = envelope_meta
        self._id_to_problem = id_to_problem
        self._lock = lock
        self._scoring_pool = scoring_pool
        self._output_file = output_file
        self._hard_deadline: Optional[float] = None

    def set_hard_deadline(self, deadline: Optional[float]) -> None:
        """Wall-clock deadline at which the read loop bails early."""
        self._hard_deadline = deadline

    def reset(self) -> None:
        """Rewind the watcher cursor; called at start_monitoring()."""
        self._watcher.reset()

    def output_file_exists(self) -> bool:
        return self._output_file.exists()

    def read_and_dispatch(self) -> int:
        """Read new records from the watcher and act on each.

        SUCCESS records are dispatched to the scoring pool; FAILED and
        TIMED_OUT records are stored directly without scoring. Returns
        the number of newly dispatched (SUCCESS) problems.
        """
        newly_dispatched = 0
        for record in self._watcher.read_new():
            with self._lock:
                if record.problem_id in self._results or self._scoring_pool.has_future(
                    record.problem_id
                ):
                    continue
                self._envelope_meta[record.problem_id] = EnvelopeMeta(
                    inference_failure_count=record.inference_failure_count,
                    inference_total=record.inference_total,
                    execution_time=record.execution_time,
                )

            if record.status is SandboxProblemStatus.SUCCESS:
                self._scoring_pool.submit(record.problem_id, record.dialogue or [])
                newly_dispatched += 1
            else:
                # FAILED / TIMED_OUT — record directly, no scoring dispatch.
                terminal = self.build_terminal_result(
                    problem_id=record.problem_id,
                    status=record.status,
                    error=record.error,
                )
                if terminal is not None:
                    with self._lock:
                        self._results[record.problem_id] = terminal

            if self._hard_deadline is not None and time.time() >= self._hard_deadline:
                break

        return newly_dispatched

    def build_terminal_result(
        self,
        *,
        problem_id: str,
        status: SandboxProblemStatus,
        error: Optional[ErrorInfo],
    ) -> Optional[ProblemResult]:
        """Build a non-success ProblemResult from envelope-only data."""
        problem = self._id_to_problem.get(problem_id)
        if not problem:
            logging.warning(f"Unknown problem_id in terminal envelope: {problem_id}")
            return None
        category = problem.get("category", "product").lower()
        problem_status = ProblemStatus(status.value)
        with self._lock:
            meta = self._envelope_meta.get(problem_id)
        inf_fail = meta.inference_failure_count if meta else 0
        inf_total = meta.inference_total if meta else 0
        exec_time = meta.execution_time if meta else 0.0
        if error and error.message:
            logging.info(
                f"Recording terminal {status} for {problem_id}: {error.message[:80]}"
            )
        return ProblemResult(
            problem_id=problem_id,
            category=category,
            status=problem_status,
            score=0.0,
            inference_failures=inf_fail,
            inference_total=inf_total,
            execution_time=exec_time,
        )

    def mark_remaining_timed_out(self) -> None:
        """Mark all unscored problems as TIMED_OUT in local results.

        Only reaches problems that never produced an envelope (sandbox death
        before write, never-started, or partial run cut off by hard deadline).
        """
        with self._lock:
            scored_ids = set(self._results.keys())

        unscored = set(self._id_to_problem.keys()) - scored_ids
        if not unscored:
            return

        logging.info(f"Marking {len(unscored)} unscored problems as TIMED_OUT")
        with self._lock:
            for pid in unscored:
                self._results[pid] = ProblemResult(
                    problem_id=pid,
                    category=self._id_to_problem[pid]
                    .get("category", "product")
                    .lower(),
                    status=ProblemStatus.TIMED_OUT,
                    score=0.0,
                )
