"""File watcher for scoring problems and reporting progress to Backend.

Architecture:
  - Main loop reads output file and dispatches scoring to a thread pool
  - Each worker runs the full per-problem pipeline: outcome + reasoning judge
  - _results dict is the sole source of truth for problem state
  - Loop exits when all problems are confirmed or hard timeout expires
  - Aggregate score is computed on-demand from _results
"""

import time
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Optional, List, Dict
from uuid import UUID

from oro_sdk.models import ProblemProgressUpdate, ProblemStatus

from bittensor.utils.btlogging import logging

from src.agent.sandbox_status import SandboxProblemStatus
from src.agent.scoring import compute_aggregate, reasoning_coefficient
from src.agent.types import (
    AggregateScore,
    ProblemDict,
    ReasoningSummary,
    ScoreComponentsSummary,
)
import requests

from .backend_client import BackendClient, BackendError
from .output_watcher import ErrorInfo, OutputWatcher
from .reasoning_judge import ReasoningJudge
from .scoring_pool import DEFAULT_SCORING_WORKERS, ScoringPool
from .types import EnvelopeMeta, ProblemResult


# Report to backend at most every N seconds
REPORT_INTERVAL_SECONDS = 10.0


class ProgressReporter:
    """Monitors sandbox output, scores problems, and reports to Backend.

    Architecture:
    - Main loop tails output file and dispatches lines to a thread pool
    - Each worker scores one problem end-to-end (outcome + reasoning judge)
    - Batch reports consolidated every REPORT_INTERVAL_SECONDS
    - Exit when confirmed == total_problems or timeout expires
    """

    def __init__(
        self,
        backend_client: BackendClient,
        eval_run_id: UUID,
        output_file: Path,
        problems: List[ProblemDict],
        workspace_dir: Path,
        poll_interval: float = 1.0,
        scoring_timeout: float = 900.0,
        chutes_access_token: Optional[str] = None,
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
        self._watcher = OutputWatcher(output_file)
        self._lock = threading.Lock()

        # Single source of truth for all problem results
        self._results: Dict[str, ProblemResult] = {}
        self._total_problems = len(problems)

        self._envelope_meta: Dict[str, EnvelopeMeta] = {}

        # Batch reporting state
        self._last_report_time = 0.0
        self._last_reported_count = 0

        # Build problem_id -> problem lookup
        self._id_to_problem: Dict[str, ProblemDict] = {}
        for problem in problems:
            problem_id = str(problem.get("problem_id") or problem.get("id"))
            if problem_id:
                self._id_to_problem[problem_id] = problem

        self._reasoning_judge = ReasoningJudge(
            chutes_access_token=chutes_access_token,
            inference_provider=inference_provider or "chutes",
            backend_base_url=backend_client.base_url,
        )
        self._scoring_pool = ScoringPool(
            problems=problems,
            results=self._results,
            envelope_meta=self._envelope_meta,
            id_to_problem=self._id_to_problem,
            lock=self._lock,
            total_problems=self._total_problems,
            reasoning_judge=self._reasoning_judge,
            max_workers=max_scoring_workers,
        )

    # Back-compat shims for tests and historical call sites that poke at
    # these attributes directly on the reporter.
    @property
    def _scoring_futures(self) -> Dict[str, Future]:
        return self._scoring_pool.futures

    @property
    def _scorers(self) -> Dict:
        return self._scoring_pool.scorers

    @_scorers.setter
    def _scorers(self, value: Dict) -> None:
        self._scoring_pool.scorers = value

    def start_monitoring(self) -> None:
        """Start the background monitoring loop."""
        self._stop_event.clear()
        self._hard_deadline = None
        self._watcher.reset()
        self._results.clear()
        self._last_reported_count = 0
        self._last_report_time = 0.0
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
            }

        judged = [r for r in results if r.reasoning_score is not None]
        total_score = sum(r.reasoning_score for r in judged)
        avg = round(total_score / len(judged), 4) if judged else 0.0
        coeff = reasoning_coefficient(avg)
        total_inf_failed = sum(r.reasoning_inf_failed for r in results)
        total_inf_total = sum(r.reasoning_inf_total for r in results)

        logging.info(
            f"Reasoning aggregate: quality={avg:.4f}, coefficient={coeff:.4f} "
            f"({len(judged)} problems judged)"
        )

        return {
            "reasoning_quality": avg,
            "reasoning_coefficient": coeff,
            "judge_inference_failed": total_inf_failed,
            "judge_inference_total": total_inf_total,
        }

    def get_problem_status(self, problem_id: str) -> ProblemStatus:
        """Return the status for a scored problem."""
        with self._lock:
            result = self._results.get(problem_id)
        if result:
            return result.status
        return ProblemStatus.FAILED

    # How long to wait with no new output before giving up (seconds)
    IDLE_TIMEOUT = 120.0

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
            # Read new lines and dispatch to thread pool
            newly_dispatched = self._read_and_dispatch()

            # Collect completed futures
            self._scoring_pool.collect_completed()

            if newly_dispatched > 0:
                last_activity_at = time.time()

            # Periodic batch report
            self._maybe_report()

            # Check exit: all problems have results
            with self._lock:
                result_count = len(self._results)
            if result_count >= self._total_problems:
                self._batch_report()
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
                self._mark_remaining_timed_out()
                self._batch_report()
                logging.warning(
                    f"No new output for {idle_secs}s, marked "
                    f"{unscored} remaining as TIMED_OUT "
                    f"({result_count}/{self._total_problems} scored)"
                )
                break

            # Check hard timeout
            if self._hard_deadline is not None and time.time() >= self._hard_deadline:
                self._mark_remaining_timed_out()
                self._batch_report()
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
                self._mark_remaining_timed_out()
                self._batch_report()
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

    def _read_and_dispatch(self) -> int:
        """Read new records from the OutputWatcher and act on each.

        SUCCESS records are dispatched to the scoring pool; FAILED and
        TIMED_OUT records are stored directly without scoring.

        Returns the number of newly dispatched (SUCCESS) problems.
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
                terminal = self._build_terminal_result(
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

    def _build_terminal_result(
        self,
        *,
        problem_id: str,
        status: SandboxProblemStatus,
        error: Optional[ErrorInfo],
    ) -> Optional["ProblemResult"]:
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

    def _maybe_report(self) -> None:
        """Report to backend if enough time has passed or new results are available."""
        with self._lock:
            current_count = len(self._results)

        if current_count == self._last_reported_count:
            return

        now = time.time()
        if now - self._last_report_time >= REPORT_INTERVAL_SECONDS:
            self._batch_report()
            self._last_report_time = now
            self._last_reported_count = current_count

    def _batch_report(self) -> None:
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
            self.backend_client.report_progress(self.eval_run_id, updates)
            logging.info(
                f"Batch reported {len(updates)}/{self._total_problems} problems"
            )
        except (BackendError, requests.RequestException) as e:
            logging.warning(f"Batch report failed ({len(updates)} problems): {e}")

    def _mark_remaining_timed_out(self) -> None:
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
