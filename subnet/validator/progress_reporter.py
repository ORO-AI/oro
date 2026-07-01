"""Validator-side orchestrator for sandbox progress reporting.

Owns the shared results dict + lock and the monitor thread; delegates
each concern to a focused component:

  EnvelopeDispatcher  — read sandbox output, dispatch to scoring
  ScoringPool         — per-problem scoring on a thread pool
  ReasoningJudge      — LLM reasoning-quality judge (called by the pool)
  ProgressBatcher     — periodic backend progress reports

Loop exits when all problems are confirmed or the hard timeout expires.
Aggregate score is computed on demand from the shared results dict.
"""

import time
import threading
from pathlib import Path
from typing import Optional, List, Dict
from uuid import UUID

from oro_sdk.models import ProblemStatus

from bittensor.utils.btlogging import logging

from src.agent.scoring import compute_aggregate, reasoning_coefficient
from src.agent.types import (
    AggregateScore,
    ProblemDict,
    ReasoningSummary,
)

from .backend_client import BackendClient
from .envelope_dispatcher import EnvelopeDispatcher
from .output_watcher import OutputWatcher
from .progress_batcher import ProgressBatcher
from .reasoning_judge import ReasoningJudge
from .scoring_pool import DEFAULT_SCORING_WORKERS, ScoringPool
from .types import EnvelopeMeta, ProblemResult


class ProgressReporter:
    """Monitors sandbox output, scores problems, and reports to Backend."""

    # How long to wait with no new output before giving up (seconds)
    IDLE_TIMEOUT = 120.0

    def __init__(
        self,
        backend_client: BackendClient,
        eval_run_id: UUID,
        output_file: Path,
        problems: List[ProblemDict],
        workspace_dir: Path,
        poll_interval: float = 1.0,
        scoring_timeout: float = 900.0,
        inference_access_token: Optional[str] = None,
        inference_provider: Optional[str] = None,
        max_scoring_workers: int = DEFAULT_SCORING_WORKERS,
    ):
        self.backend_client = backend_client
        self.eval_run_id = eval_run_id
        self.output_file = output_file
        self.problems = problems
        self.workspace_dir = workspace_dir
        self.poll_interval = poll_interval
        self.scoring_timeout = scoring_timeout

        self._stop_event = threading.Event()
        self._hard_deadline: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Single source of truth for all problem results
        self._results: Dict[str, ProblemResult] = {}
        self._total_problems = len(problems)

        # Build problem_id -> problem lookup
        self._id_to_problem: Dict[str, ProblemDict] = {}
        for problem in problems:
            problem_id = str(problem.get("problem_id") or problem.get("id"))
            if problem_id:
                self._id_to_problem[problem_id] = problem

        # Shared between dispatcher (writer) and scoring pool (reader); the
        # reporter itself never reads it after wire-up.
        envelope_meta: Dict[str, EnvelopeMeta] = {}

        self._watcher = OutputWatcher(output_file)
        self._reasoning_judge = ReasoningJudge(
            inference_access_token=inference_access_token,
            inference_provider=inference_provider or "chutes",
            backend_base_url=backend_client.base_url,
        )
        self._scoring_pool = ScoringPool(
            problems=problems,
            results=self._results,
            envelope_meta=envelope_meta,
            id_to_problem=self._id_to_problem,
            lock=self._lock,
            reasoning_judge=self._reasoning_judge,
            max_workers=max_scoring_workers,
        )
        self._envelope_dispatcher = EnvelopeDispatcher(
            watcher=self._watcher,
            results=self._results,
            envelope_meta=envelope_meta,
            id_to_problem=self._id_to_problem,
            lock=self._lock,
            scoring_pool=self._scoring_pool,
        )
        self._batcher = ProgressBatcher(
            backend_client=backend_client,
            eval_run_id=eval_run_id,
            total_problems=self._total_problems,
            results=self._results,
            lock=self._lock,
        )

    def start_monitoring(self) -> None:
        """Start the background monitoring loop."""
        self._stop_event.clear()
        self._hard_deadline = None
        self._watcher.reset()
        self._results.clear()
        self._batcher.reset()
        self._scoring_pool.futures.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def signal_sandbox_done(self) -> None:
        """Signal that the sandbox has exited. Starts the hard timeout clock."""
        self._stop_event.set()

    def wait_for_completion(self, timeout: Optional[float] = None) -> None:
        """Block until the monitoring loop exits.

        The loop exits when all problems are confirmed reported to the backend,
        or when the hard timeout expires (remaining marked as TIMED_OUT).
        """
        join_timeout = timeout or (self.scoring_timeout + 60)
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
            if self._thread.is_alive():
                logging.warning(
                    f"Monitoring thread did not finish within {join_timeout}s"
                )
        self._scoring_pool.shutdown()

    def get_aggregate_score(self) -> Optional[AggregateScore]:
        """Compute aggregate score on demand from _results."""
        total = self._total_problems if self._total_problems > 0 else 1

        with self._lock:
            results = [
                {"category": r.category, "score_dict": r.score_dict}
                for r in self._results.values()
            ]
            scored_count = len(self._results)

        aggregate = compute_aggregate(results, total)

        logging.info(
            f"Aggregate score computed: "
            f"success_rate={aggregate['success_rate']:.4f}, "
            f"gt_rate={aggregate['ground_truth_rate']:.4f}, "
            f"format={aggregate['format_score']:.4f}, "
            f"field_matching={aggregate['field_matching']:.4f} "
            f"({aggregate['successful_problems']}/{total} succeeded, "
            f"{scored_count} scored)"
        )

        return aggregate

    def get_reasoning_data(self) -> ReasoningSummary:
        """Return aggregate reasoning metrics for run-level score_components.

        Per-problem reasoning data is now sent via progress reports.
        This method returns only the summary fields needed at run completion.
        """
        with self._lock:
            results = list(self._results.values())

        if not results:
            return {
                "reasoning_quality": 0.0,
                "reasoning_coefficient": reasoning_coefficient(0.0),
                "judge_inference_failed": 0,
                "judge_inference_total": 0,
                "judge_inference_402": 0,
            }

        judged = [r for r in results if r.reasoning_score is not None]
        total_score = sum(r.reasoning_score for r in judged)
        avg = round(total_score / len(judged), 4) if judged else 0.0
        coeff = reasoning_coefficient(avg)
        total_inf_failed = sum(r.reasoning_inf_failed for r in results)
        total_inf_total = sum(r.reasoning_inf_total for r in results)
        total_inf_402 = sum(r.reasoning_inf_402 for r in results)

        logging.info(
            f"Reasoning aggregate: quality={avg:.4f}, coefficient={coeff:.4f} "
            f"({len(judged)} problems judged)"
        )

        return {
            "reasoning_quality": avg,
            "reasoning_coefficient": coeff,
            "judge_inference_failed": total_inf_failed,
            "judge_inference_total": total_inf_total,
            "judge_inference_402": total_inf_402,
        }

    def get_problem_status(self, problem_id: str) -> ProblemStatus:
        """Return the status for a scored problem."""
        with self._lock:
            result = self._results.get(problem_id)
        if result:
            return result.status
        return ProblemStatus.FAILED

    def _run(self) -> None:
        """Main monitoring loop.

        Tails the output file and dispatches scoring to the thread pool.
        Batch-reports periodically instead of after each problem.
        Exits when:
        - All problems have results, OR
        - Hard timeout (scoring_timeout) expires, OR
        - No new output for IDLE_TIMEOUT seconds after sandbox exit
        """
        last_activity_at: Optional[float] = None

        while True:
            newly_dispatched = self._envelope_dispatcher.read_and_dispatch(
                hard_deadline=self._hard_deadline
            )
            self._scoring_pool.collect_completed()

            if newly_dispatched > 0:
                last_activity_at = time.time()

            self._batcher.maybe_report()

            # Check exit: all problems have results
            with self._lock:
                result_count = len(self._results)
            if result_count >= self._total_problems:
                self._batcher.batch_report()
                logging.info(f"All {self._total_problems} problems completed")
                break

            # Start hard timeout when sandbox exits
            if self._stop_event.is_set() and self._hard_deadline is None:
                self._hard_deadline = time.time() + self.scoring_timeout
                if last_activity_at is None:
                    last_activity_at = time.time()
                logging.info(
                    f"Sandbox exited, scoring timeout in {self.scoring_timeout}s"
                )

            # Check idle timeout: no new output and no in-flight scoring
            pending_futures = self._scoring_pool.pending_count()
            if (
                self._hard_deadline is not None
                and last_activity_at is not None
                and pending_futures == 0
                and (time.time() - last_activity_at) >= self.IDLE_TIMEOUT
            ):
                idle_secs = int(time.time() - last_activity_at)
                unscored = self._total_problems - result_count
                self._envelope_dispatcher.mark_remaining_timed_out()
                self._batcher.batch_report()
                logging.warning(
                    f"No new output for {idle_secs}s, marked "
                    f"{unscored} remaining as TIMED_OUT "
                    f"({result_count}/{self._total_problems} scored)"
                )
                break

            # Check hard timeout
            if self._hard_deadline is not None and time.time() >= self._hard_deadline:
                self._envelope_dispatcher.mark_remaining_timed_out()
                self._batcher.batch_report()
                with self._lock:
                    result_count = len(self._results)
                logging.warning(
                    f"Hard timeout expired with {result_count}/{self._total_problems} "
                    f"scored, marked remaining as TIMED_OUT"
                )
                break

            # No output file at all after sandbox exit = genuine failure
            if (
                self._hard_deadline is not None
                and result_count == 0
                and not self.output_file.exists()
            ):
                self._envelope_dispatcher.mark_remaining_timed_out()
                self._batcher.batch_report()
                logging.warning("No output file found after sandbox exit")
                break

            # Log progress periodically after sandbox exits
            if self._hard_deadline is not None and newly_dispatched == 0:
                elapsed = int(
                    time.time() - (self._hard_deadline - self.scoring_timeout)
                )
                logging.info(
                    f"Scored {result_count}/{self._total_problems} problems "
                    f"({pending_futures} in-flight), "
                    f"waiting for remaining output ({elapsed}s elapsed)"
                )

            time.sleep(self.poll_interval)
