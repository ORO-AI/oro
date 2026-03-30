"""File watcher for reporting per-problem progress to Backend.

Architecture:
  - Single dict (_results) is the source of truth for all problem state
  - Background thread tails output file during sandbox run for real-time progress
  - stop_monitoring(deadline) lets the thread keep scoring until all done or timeout
  - Aggregate is always computed on-demand from _results — never cached separately
"""

import json
import os
import sys
import time
import traceback
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from uuid import UUID

from oro_sdk.models import ProblemProgressUpdate, ProblemStatus

from bittensor.utils.btlogging import logging

from src.agent.scoring import is_problem_successful, compute_aggregate
from .backend_client import BackendClient

if TYPE_CHECKING:
    from .retry_queue import LocalRetryQueue


@dataclass
class ProblemResult:
    """Single source of truth for one problem's scoring outcome."""

    problem_id: str
    category: str
    status: ProblemStatus
    score: float
    score_dict: Dict[str, Any] = field(default_factory=dict)
    inference_failures: int = 0
    inference_total: int = 0


class ProgressReporter:
    """Monitors sandbox output file, scores problems, and reports to Backend.

    Tails the output JSONL file, scores each problem using ProblemScorer
    as it completes, reports individual scores to Backend, and computes
    aggregate score on demand from a single results dict.
    """

    def __init__(
        self,
        backend_client: BackendClient,
        eval_run_id: UUID,
        output_file: Path,
        problems: List[Dict[str, Any]],
        workspace_dir: Path,
        poll_interval: float = 1.0,
        retry_queue: Optional["LocalRetryQueue"] = None,
        scoring_timeout: float = 300.0,
    ):
        self.backend_client = backend_client
        self.eval_run_id = eval_run_id
        self.output_file = output_file
        self.problems = problems
        self.workspace_dir = workspace_dir
        self.poll_interval = poll_interval
        self.scoring_timeout = scoring_timeout

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._file_position = 0
        self._lock = threading.Lock()

        # Single source of truth for all problem results
        self._results: Dict[str, ProblemResult] = {}
        self._total_problems = len(problems)
        self._scorers: Dict[str, Any] = {}
        self._retry_queue = retry_queue
        self._failed_reports: List[ProblemProgressUpdate] = []

        # Build problem_id -> problem lookup
        self._id_to_problem: Dict[str, Dict[str, Any]] = {}
        for problem in problems:
            problem_id = str(problem.get("problem_id") or problem.get("id"))
            if problem_id:
                self._id_to_problem[problem_id] = problem

        self._initialize_scorers()

    # ─── Scorer initialization ──────────────────────────────────────────

    def _initialize_scorers(self) -> None:
        """Initialize per-category ProblemScorers from problem metadata."""
        try:
            original_dir = os.getcwd()
            os.chdir(str(self.workspace_dir))

            scorer_path = str(self.workspace_dir / "src" / "agent")
            if scorer_path not in sys.path:
                sys.path.insert(0, scorer_path)

            from problem_scorer import ProblemScorer, clear_product_cache

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
                    title_embeddings = problem.get("reward_title_embeddings")
                    if title_embeddings:
                        if isinstance(reward, dict):
                            reward["_title_embeddings"] = title_embeddings
                        elif isinstance(reward, list):
                            for item in reward:
                                if isinstance(item, dict):
                                    item["_title_embeddings"] = title_embeddings
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
            os.chdir(original_dir)

        except Exception as e:
            logging.error(f"Failed to initialize ProblemScorers: {e}")
            self._scorers = {}
            try:
                os.chdir(original_dir)
            except Exception:
                pass

    # ─── Public API ─────────────────────────────────────────────────────

    def start_monitoring(self) -> None:
        """Start the file watcher background thread."""
        self._stop_event.clear()
        self._file_position = 0
        self._results = {}
        self._failed_reports = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop_monitoring(self) -> None:
        """Signal the thread to finish scoring, then wait for it.

        The thread will keep reading and scoring until all problems are done
        or the scoring timeout expires. This replaces the old retry loop.
        """
        # Tell the thread the sandbox is done — it should finish up
        self._stop_event.set()

        # Wait for the thread to finish (scoring timeout + buffer)
        join_timeout = self.scoring_timeout + 30
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
            if self._thread.is_alive():
                logging.warning(f"Scoring thread did not finish within {join_timeout}s")

        # Retry any failed progress reports
        if self._failed_reports:
            still_failed = []
            for progress in self._failed_reports:
                try:
                    self.backend_client.report_progress(self.eval_run_id, [progress])
                    logging.info(f"Retry succeeded for {progress.problem_id}")
                except Exception:
                    still_failed.append(progress)

            if still_failed and self._retry_queue:
                for progress in still_failed:
                    self._retry_queue.add_progress(self.eval_run_id, progress)
                logging.warning(
                    f"{len(still_failed)} progress report(s) queued for retry"
                )

    def get_aggregate_score(self) -> Optional[Dict[str, Any]]:
        """Compute aggregate score on demand from _results.

        Always recomputed — never cached — so it always reflects
        the current state of _results.
        """
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

    def get_problem_status(self, problem_id: str) -> ProblemStatus:
        """Return the status for a scored problem."""
        with self._lock:
            result = self._results.get(problem_id)
        if result:
            return result.status
        return ProblemStatus.FAILED

    def report_unscored_as_timed_out(self) -> None:
        """Report problems that were never scored as TIMED_OUT."""
        with self._lock:
            scored_ids = set(self._results.keys())
        all_ids = set(self._id_to_problem.keys())
        unscored_ids = all_ids - scored_ids

        if not unscored_ids:
            return

        logging.info(f"Reporting {len(unscored_ids)} unscored problems as TIMED_OUT")
        for pid in unscored_ids:
            try:
                progress = ProblemProgressUpdate(
                    problem_id=UUID(pid),
                    status=ProblemStatus.TIMED_OUT,
                    score=0.0,
                )
                self._report_progress(progress)
            except Exception as e:
                logging.warning(f"Failed to report timed-out problem {pid}: {e}")

    # ─── Background thread ──────────────────────────────────────────────

    def _run(self) -> None:
        """Background thread: tail file during sandbox, finish scoring after."""
        # Phase 1: Tail file while sandbox is running (real-time progress)
        while not self._stop_event.is_set():
            self._read_and_score()
            self._stop_event.wait(self.poll_interval)

        # Phase 2: Sandbox exited — keep scoring remaining output until done or timeout
        deadline = time.time() + self.scoring_timeout
        while time.time() < deadline:
            self._read_and_score()
            with self._lock:
                scored = len(self._results)
            if scored >= self._total_problems:
                logging.info(f"All {self._total_problems} problems scored")
                break
            # No output file at all = sandbox genuinely failed
            if scored == 0 and not self.output_file.exists():
                break
            elapsed = int(self.scoring_timeout - (deadline - time.time()))
            logging.info(
                f"Scored {scored}/{self._total_problems} problems, "
                f"waiting for remaining output ({elapsed}s elapsed)"
            )
            time.sleep(self.poll_interval)

    def _read_and_score(self) -> None:
        """Read new lines from the output file and score them."""
        try:
            if not self.output_file.exists():
                return

            file_size = self.output_file.stat().st_size
            if file_size < self._file_position:
                logging.info("Output file truncated, resetting position")
                self._file_position = 0

            with open(self.output_file) as f:
                f.seek(self._file_position)
                new_lines = f.readlines()
                self._file_position = f.tell()

            for line in new_lines:
                if line.strip():
                    self._score_and_report(line)

        except (OSError, IOError) as e:
            logging.warning(f"Error reading output file: {e}")
            self._file_position = 0
        except Exception as e:
            logging.warning(f"Error processing output: {e}")

    def _score_and_report(self, line: str) -> None:
        """Score a single problem line and report to backend.

        This is the ONE place where scoring happens. Both the background
        thread (real-time) and post-sandbox processing use this method.
        """
        if not self._scorers:
            return

        try:
            dialogue = json.loads(line.strip())
            if not isinstance(dialogue, list) or not dialogue:
                return

            # Extract problem_id and query
            extra_info = (dialogue[0].get("extra_info") or {}) if dialogue else {}
            problem_id = extra_info.get("problem_id")
            if not problem_id:
                logging.warning("No problem_id in dialogue extra_info")
                return

            # Skip if already scored
            with self._lock:
                if problem_id in self._results:
                    return

            # Look up problem metadata
            problem = self._id_to_problem.get(str(problem_id))
            if not problem:
                logging.warning(f"Unknown problem_id: {problem_id}")
                return

            query = problem.get("query") or extra_info.get("query")
            category = problem.get("category", "product").lower()

            scorer = self._scorers.get(category)
            if not scorer:
                logging.warning(f"No scorer for category '{category}'")
                return

            # Score
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

            # Read inference stats
            inf_failures, inf_total = self._read_inference_stats(problem_id)

            # Store result (single source of truth)
            result = ProblemResult(
                problem_id=str(problem_id),
                category=category,
                status=status,
                score=score,
                score_dict=score_dict if isinstance(score_dict, dict) else {},
                inference_failures=inf_failures,
                inference_total=inf_total,
            )
            with self._lock:
                self._results[str(problem_id)] = result
                completed = len(self._results)

            logging.info(
                f"Problem {completed}/{self._total_problems} scored: "
                f"{score:.4f} (query: {query[:50]}...)"
            )

            # Report to backend
            update = ProblemProgressUpdate(
                problem_id=UUID(problem_id)
                if isinstance(problem_id, str)
                else problem_id,
                status=status,
                score=score,
            )
            if inf_total > 0:
                from oro_sdk.models import (
                    ProblemProgressUpdateScoreComponentsSummaryType0,
                )

                update.score_components_summary = (
                    ProblemProgressUpdateScoreComponentsSummaryType0.from_dict(
                        {
                            "inference_failure_count": inf_failures,
                            "inference_total": inf_total,
                        }
                    )
                )
            self._report_progress(update)

        except json.JSONDecodeError:
            logging.warning(f"Failed to parse line: {line[:100]}...")
        except Exception as e:
            logging.error(f"Error scoring problem: {e}")
            traceback.print_exc()

    # ─── Helpers ────────────────────────────────────────────────────────

    def _read_inference_stats(self, problem_id: str) -> tuple[int, int]:
        """Read inference stats from the shared JSONL sidecar file."""
        stats_path = self.output_file.parent / "inference_stats.jsonl"
        try:
            with open(stats_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if str(entry.get("problem_id")) == str(problem_id):
                        return entry.get("inference_failed", 0), entry.get(
                            "inference_total", 0
                        )
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        return 0, 0

    def _report_progress(self, progress: ProblemProgressUpdate) -> None:
        """Report a single problem's progress to Backend."""
        try:
            self.backend_client.report_progress(self.eval_run_id, [progress])
            logging.info(
                f"Reported progress for {progress.problem_id}: "
                f"{progress.status}, score={progress.score:.4f}"
            )
        except Exception as e:
            self._failed_reports.append(progress)
            logging.warning(f"Failed to report progress for {progress.problem_id}: {e}")
