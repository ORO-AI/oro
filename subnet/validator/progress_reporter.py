"""File watcher for scoring problems and reporting progress to Backend.

Architecture:
  - Main loop reads output file and dispatches scoring to a thread pool
  - Each worker runs the full per-problem pipeline: outcome + reasoning judge
  - _results dict is the sole source of truth for problem state
  - Loop exits when all problems are confirmed or hard timeout expires
  - Aggregate score is computed on-demand from _results
"""

import json
import time
import traceback
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
from uuid import UUID

from oro_sdk.models import ProblemProgressUpdate, ProblemStatus

from bittensor.utils.btlogging import logging

from src.agent.problem_scorer import ProblemScorer, clear_product_cache
from src.agent.sandbox_status import SandboxProblemStatus
from src.agent.scoring import is_problem_successful, compute_aggregate, reasoning_coefficient
from src.agent.types import ScoreDict
from subnet.sandbox import attach_title_embeddings
import requests

from .backend_client import BackendClient, BackendError


@dataclass
class EnvelopeMeta:
    """Per-problem metadata captured from the sandbox envelope line.

    Held under ``ProgressReporter._lock`` and read by both dispatch (terminal
    branch) and the scoring worker thread.
    """

    inference_failure_count: int
    inference_total: int
    execution_time: float


@dataclass
class ProblemResult:
    """Single source of truth for one problem's scoring outcome."""

    problem_id: str
    category: str
    status: ProblemStatus
    score: float
    score_dict: ScoreDict = field(default_factory=dict)
    inference_failures: int = 0
    inference_total: int = 0
    reasoning_score: float | None = None
    reasoning_explanation: str = ""
    reasoning_model: str = ""
    reasoning_inf_failed: int = 0
    reasoning_inf_total: int = 0
    execution_time: float | None = None


# Default number of concurrent scoring workers
DEFAULT_SCORING_WORKERS = 4

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
        problems: List[Dict[str, Any]],
        workspace_dir: Path,
        poll_interval: float = 1.0,
        scoring_timeout: float = 900.0,
        chutes_access_token: Optional[str] = None,
        max_scoring_workers: int = DEFAULT_SCORING_WORKERS,
    ):
        self.backend_client = backend_client
        self.eval_run_id = eval_run_id
        self.output_file = output_file
        self.problems = problems
        self.workspace_dir = workspace_dir
        self.poll_interval = poll_interval
        self.scoring_timeout = scoring_timeout
        self._chutes_access_token = chutes_access_token

        self._stop_event = threading.Event()
        self._hard_deadline: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._file_position = 0
        self._lock = threading.Lock()

        # Single source of truth for all problem results
        self._results: Dict[str, ProblemResult] = {}
        self._total_problems = len(problems)
        self._scorers: Dict[str, Any] = {}

        self._envelope_meta: Dict[str, EnvelopeMeta] = {}

        # Circuit breaker for reasoning judge — stop after N consecutive
        # total failures to avoid retry storms on bad tokens/infra issues.
        self._consecutive_judge_failures = 0
        self._judge_circuit_open = False

        # Batch reporting state
        self._last_report_time = 0.0
        self._last_reported_count = 0

        # Thread pool for concurrent scoring
        self._scoring_executor = ThreadPoolExecutor(
            max_workers=max_scoring_workers,
            thread_name_prefix="scorer",
        )
        self._scoring_futures: Dict[str, Future] = {}

        # Build problem_id -> problem lookup
        self._id_to_problem: Dict[str, Dict[str, Any]] = {}
        for problem in problems:
            problem_id = str(problem.get("problem_id") or problem.get("id"))
            if problem_id:
                self._id_to_problem[problem_id] = problem

        self._initialize_scorers()

    def _initialize_scorers(self) -> None:
        """Initialize per-category ProblemScorers from problem metadata."""
        try:
            clear_product_cache()

            category_rewards: Dict[str, Dict] = {}
            category_vouchers: Dict[str, Dict] = {}

            for problem in self.problems:
                query = problem.get("query")
                reward = problem.get("reward")
                category = problem.get("category", "product").lower()

                if category not in ("product", "shop", "voucher"):
                    category = "product"

                if query and reward:
                    attach_title_embeddings(reward, problem.get("reward_title_embeddings"))
                    category_rewards.setdefault(category, {})[query] = reward

                if category == "voucher":
                    voucher = problem.get("voucher")
                    if query and voucher:
                        category_vouchers.setdefault(category, {})[query] = voucher

            for category, rewards in category_rewards.items():
                vouchers = category_vouchers.get(category, {})
                self._scorers[category] = ProblemScorer(
                    task=category, rewards=rewards, vouchers=vouchers
                )
                logging.info(
                    f"Created ProblemScorer for '{category}' with {len(rewards)} problems"
                )

            logging.info(
                f"Initialized {len(self._scorers)} scorers: {list(self._scorers.keys())}"
            )

        except (ImportError, OSError, ValueError, TypeError, KeyError) as e:
            logging.error(f"Failed to initialize ProblemScorers: {e}")
            self._scorers = {}

    def start_monitoring(self) -> None:
        """Start the background monitoring loop."""
        self._stop_event.clear()
        self._hard_deadline = None
        self._file_position = 0
        self._results = {}
        self._last_reported_count = 0
        self._last_report_time = 0.0
        self._scoring_futures = {}
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
        self._scoring_executor.shutdown(wait=False)

    def get_aggregate_score(self) -> Optional[Dict[str, Any]]:
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

    def get_reasoning_data(self) -> Dict[str, Any]:
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
            self._collect_completed_futures()

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
            pending_futures = sum(1 for f in self._scoring_futures.values() if not f.done())
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
                elapsed = int(time.time() - (self._hard_deadline - self.scoring_timeout))
                logging.info(
                    f"Scored {result_count}/{self._total_problems} problems "
                    f"({pending_futures} in-flight), "
                    f"waiting for remaining output ({elapsed}s elapsed)"
                )

            time.sleep(self.poll_interval)

    def _read_and_dispatch(self) -> int:
        """Read new envelope lines from output file.

        SUCCESS envelopes are dispatched to the scoring pool. FAILED and
        TIMED_OUT envelopes are recorded directly without scoring (no dialogue
        to score). Per-problem inference counts and execution_time come from
        the envelope itself — no sidecar reads on the validator side.

        Returns the number of newly dispatched (SUCCESS) problems.
        """
        newly_dispatched = 0
        try:
            if not self.output_file.exists():
                return 0

            file_size = self.output_file.stat().st_size
            if file_size < self._file_position:
                logging.info("Output file truncated, resetting position")
                self._file_position = 0

            with open(self.output_file) as f:
                f.seek(self._file_position)
                new_lines = f.readlines()
                self._file_position = f.tell()

            for line in new_lines:
                line = line.strip()
                if not line:
                    continue

                envelope = self._parse_envelope(line)
                if envelope is None:
                    continue

                problem_id = envelope.get("problem_id")
                status: SandboxProblemStatus = envelope["status"]
                if not problem_id:
                    continue
                problem_id = str(problem_id)

                with self._lock:
                    if problem_id in self._results:
                        continue
                    if problem_id in self._scoring_futures:
                        continue
                    self._envelope_meta[problem_id] = EnvelopeMeta(
                        inference_failure_count=int(
                            envelope.get("inference_failure_count") or 0
                        ),
                        inference_total=int(envelope.get("inference_total") or 0),
                        execution_time=float(envelope.get("execution_time") or 0.0),
                    )

                if status is SandboxProblemStatus.SUCCESS:
                    dialogue = envelope.get("dialogue") or []
                    future = self._scoring_executor.submit(
                        self._score_problem, dialogue, problem_id
                    )
                    self._scoring_futures[problem_id] = future
                    newly_dispatched += 1
                else:
                    # FAILED / TIMED_OUT — record directly, no scoring dispatch.
                    terminal = self._build_terminal_result(
                        problem_id=problem_id,
                        status=status,
                        error=envelope.get("error"),
                    )
                    if terminal is not None:
                        with self._lock:
                            self._results[problem_id] = terminal

                if self._hard_deadline is not None and time.time() >= self._hard_deadline:
                    break

        except (OSError, IOError) as e:
            logging.warning(f"Error reading output file: {e}")
            self._file_position = 0

        return newly_dispatched

    def _parse_envelope(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse one envelope line. Returns None for malformed input.

        Coerces ``status`` to ``SandboxProblemStatus`` once at the boundary;
        unrecognized values cause the line to be skipped entirely so callers
        can rely on `envelope["status"]` being a typed enum.
        """
        try:
            line = line.replace("\x00", "")
            envelope = json.loads(line)
        except json.JSONDecodeError as e:
            logging.warning(f"Skipping malformed envelope line: {e}")
            return None
        if not isinstance(envelope, dict):
            logging.warning(
                f"Skipping non-dict envelope line: {type(envelope).__name__}"
            )
            return None
        raw_status = envelope.get("status")
        if not raw_status:
            logging.warning("Skipping envelope without status field")
            return None
        try:
            envelope["status"] = SandboxProblemStatus(raw_status)
        except ValueError:
            logging.warning(
                f"Skipping envelope with unrecognized status '{raw_status}'"
            )
            return None
        return envelope

    def _build_terminal_result(
        self,
        *,
        problem_id: str,
        status: SandboxProblemStatus,
        error: Optional[Dict[str, Any]],
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
        if error:
            err_msg = error.get("message") if isinstance(error, dict) else None
            if err_msg:
                logging.info(
                    f"Recording terminal {status} for {problem_id}: {err_msg[:80]}"
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

    def _collect_completed_futures(self) -> None:
        """Check for completed scoring futures and update last_activity_at."""
        completed = [pid for pid, f in self._scoring_futures.items() if f.done()]
        for pid in completed:
            future = self._scoring_futures.pop(pid)
            exc = future.exception()
            if exc:
                logging.error(f"Scoring worker failed for {pid}: {exc}")

    def _score_problem(self, dialogue: list, problem_id: str) -> None:
        """Score a single problem end-to-end. Runs in a worker thread.

        Receives the parsed dialogue (list of step dicts) directly from the
        envelope, instead of reparsing a JSONL line.
        """
        if not self._scorers:
            return

        if not isinstance(dialogue, list) or not dialogue:
            return

        try:
            problem = self._id_to_problem.get(str(problem_id))
            if not problem:
                logging.warning(f"Unknown problem_id: {problem_id}")
                return

            extra_info = (dialogue[0].get("extra_info") or {}) if dialogue else {}
            with self._lock:
                meta = self._envelope_meta.get(str(problem_id))
            execution_time = (
                meta.execution_time if meta is not None else extra_info.get("execution_time")
            )
            query = problem.get("query") or extra_info.get("query")
            category = problem.get("category", "product").lower()

            scorer = self._scorers.get(category)
            if not scorer:
                logging.warning(f"No scorer for category '{category}'")
                return

            with self._lock:
                scored_count = len(self._results) + 1
            logging.info(
                f"Scoring problem {scored_count}/{self._total_problems}: "
                f"{query[:50]}..."
            )

            score_dict = scorer.score_problem(query=query, output=dialogue)
            is_successful = is_problem_successful(score_dict, category)
            score = 1.0 if is_successful else 0.0
            status = ProblemStatus.SUCCESS if is_successful else ProblemStatus.FAILED
            inf_failures = meta.inference_failure_count if meta else 0
            inf_total = meta.inference_total if meta else 0

            reasoning = self._run_reasoning_judge(dialogue, problem_id)

            result = ProblemResult(
                problem_id=str(problem_id),
                category=category,
                status=status,
                score=score,
                score_dict=score_dict if isinstance(score_dict, dict) else {},
                inference_failures=inf_failures,
                inference_total=inf_total,
                execution_time=execution_time,
                **reasoning,
            )
            with self._lock:
                self._results[str(problem_id)] = result
                completed = len(self._results)

            logging.info(
                f"Problem {completed}/{self._total_problems} scored: "
                f"{score:.4f} (query: {query[:50]}...)"
            )

        except Exception as e:
            logging.error(f"Error scoring problem {problem_id}: {e}")
            traceback.print_exc()

    def _run_reasoning_judge(self, dialogue: list, problem_id: str) -> dict:
        """Run the LLM reasoning judge with circuit breaker protection.

        Returns a dict of reasoning fields to unpack into ProblemResult.
        """
        empty = {
            "reasoning_score": None,
            "reasoning_explanation": "",
            "reasoning_model": "",
            "reasoning_inf_failed": 0,
            "reasoning_inf_total": 0,
        }

        with self._lock:
            should_judge = bool(self._chutes_access_token and not self._judge_circuit_open)

        if not should_judge:
            return empty

        try:
            from src.agent.reasoning_scorer import score_reasoning_quality

            judge_result = score_reasoning_quality(
                dialogue, api_key=self._chutes_access_token
            )

            with self._lock:
                if judge_result["inference_total"] > 0 and judge_result["inference_failed"] == judge_result["inference_total"]:
                    self._consecutive_judge_failures += 1
                    if self._consecutive_judge_failures >= 3:
                        self._judge_circuit_open = True
                        logging.error(
                            "Reasoning judge circuit breaker tripped: "
                            "3 consecutive problems with 100% judge failure."
                        )
                else:
                    self._consecutive_judge_failures = 0

            logging.info(
                f"Reasoning score: {judge_result['score']:.2f} "
                f"(problem={problem_id}, model={judge_result['model']})"
            )

            return {
                "reasoning_score": judge_result["score"],
                "reasoning_explanation": judge_result["explanation"],
                "reasoning_model": judge_result["model"],
                "reasoning_inf_failed": judge_result["inference_failed"],
                "reasoning_inf_total": judge_result["inference_total"],
            }

        except Exception as e:
            logging.warning(f"Reasoning judge failed for {problem_id}: {e}")
            with self._lock:
                self._consecutive_judge_failures += 1
                if self._consecutive_judge_failures >= 3:
                    self._judge_circuit_open = True
                    logging.error(
                        "Reasoning judge circuit breaker tripped after exceptions."
                    )
            return empty

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
            scs = None
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
                inference_failure_count=r.inference_failures if r.inference_total > 0 else None,
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

        After ORO-907 the envelope path records FAILED/TIMED_OUT results
        directly when their envelopes arrive. This sweep therefore only
        reaches problems that genuinely never produced an envelope — typically
        a sandbox death before write, a never-started problem, or a partial
        run cut off by hard deadline.
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
                    category=self._id_to_problem[pid].get("category", "product").lower(),
                    status=ProblemStatus.TIMED_OUT,
                    score=0.0,
                )
