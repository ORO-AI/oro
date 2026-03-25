"""File watcher for reporting per-problem progress to Backend."""

import json
import sys
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from uuid import UUID

from oro_sdk.models import ProblemProgressUpdate, ProblemStatus

from bittensor.utils.btlogging import logging

from .backend_client import BackendClient

if TYPE_CHECKING:
    from .retry_queue import LocalRetryQueue


class ProgressReporter:
    """Monitors sandbox output file, scores problems, and reports to Backend.

    Tails the output JSONL file (which contains dialogue entries),
    scores each problem using ProblemScorer as it completes,
    reports individual scores to Backend, and computes aggregate score.
    """

    STATUS_MAP = {
        "success": ProblemStatus.SUCCESS,
        "failed": ProblemStatus.FAILED,
        "error": ProblemStatus.FAILED,
        "timeout": ProblemStatus.TIMED_OUT,
        "skipped": ProblemStatus.SKIPPED,
        "running": ProblemStatus.RUNNING,
        "pending": ProblemStatus.PENDING,
    }

    def __init__(
        self,
        backend_client: BackendClient,
        eval_run_id: UUID,
        output_file: Path,
        problems: List[Dict[str, Any]],
        workspace_dir: Path,
        poll_interval: float = 1.0,
        retry_queue: Optional["LocalRetryQueue"] = None,
        stop_retry_interval: float = 3.0,
    ):
        """Initialize ProgressReporter.

        Args:
            backend_client: Client for Backend API
            eval_run_id: UUID of the evaluation run
            output_file: Path to sandbox output JSONL file
            problems: List of problem dicts with metadata (problem_id, query, reward, category, etc.)
            workspace_dir: Path to ShoppingBench workspace (for ProblemScorer)
            poll_interval: How often to check file for new entries (seconds)
        """
        self.backend_client = backend_client
        self.eval_run_id = eval_run_id
        self.output_file = output_file
        self.problems = problems
        self.workspace_dir = workspace_dir
        self.poll_interval = poll_interval
        self.stop_retry_interval = stop_retry_interval

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_position = 0

        # Track scoring progress
        self._problem_scores: Dict[str, float] = {}  # Map problem_id -> gt score
        self._problem_score_dicts: Dict[
            str, Dict[str, Any]
        ] = {}  # Map problem_id -> full score dict
        self._problem_categories: Dict[str, str] = {}  # Map problem_id -> category
        self._completed_count = 0
        self._total_problems = len(problems)
        self._scorers: Dict[str, Any] = {}  # Map category -> ProblemScorer
        self._aggregate_score: Optional[Dict[str, Any]] = None
        self._retry_queue = retry_queue
        self._failed_reports: List[ProblemProgressUpdate] = []

        # Create problem_id -> problem lookup for matching dialogue to problems
        self._id_to_problem: Dict[str, Dict[str, Any]] = {}
        for problem in problems:
            problem_id = str(problem.get("problem_id") or problem.get("id"))
            if problem_id:
                self._id_to_problem[problem_id] = problem

        # Initialize ProblemScorers (one per category)
        self._initialize_scorers()

    def _initialize_scorers(self) -> None:
        """Initialize per-category ProblemScorers from problem metadata."""
        try:
            # Change to workspace directory so ProblemScorer can find indexes
            import os

            original_dir = os.getcwd()
            os.chdir(str(self.workspace_dir))

            # Add workspace to path so we can import ShoppingBench modules
            scorer_path = str(self.workspace_dir / "src" / "agent")
            if scorer_path not in sys.path:
                sys.path.insert(0, scorer_path)

            from problem_scorer import ProblemScorer, clear_product_cache

            # Clear product cache from previous evaluation runs
            clear_product_cache()

            # Group rewards and vouchers by category
            category_rewards: Dict[str, Dict] = {}
            category_vouchers: Dict[str, Dict] = {}

            for problem in self.problems:
                query = problem.get("query")
                reward = problem.get("reward")
                category = problem.get("category", "product").lower()

                if category not in ("product", "shop", "voucher"):
                    category = "product"

                if query and reward:
                    category_rewards.setdefault(category, {})[query] = reward

                if category == "voucher":
                    voucher = problem.get("voucher")
                    if query and voucher:
                        category_vouchers.setdefault(category, {})[query] = voucher

            # Create one ProblemScorer per category
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

            # Restore original directory
            os.chdir(original_dir)

        except Exception as e:
            logging.error(f"Failed to initialize ProblemScorers: {e}")
            logging.error("Per-problem scoring will not work without ProblemScorers")
            self._scorers = {}
            # Restore original directory even on error
            try:
                os.chdir(original_dir)
            except Exception as restore_error:
                logging.error(
                    f"Failed to restore directory to {original_dir}: {restore_error}"
                )

    def start_monitoring(self) -> None:
        """Start the file watcher background thread."""
        self._stop_event.clear()
        self._last_position = 0
        self._problem_scores = {}
        self._problem_score_dicts = {}
        self._problem_categories = {}
        self._completed_count = 0
        self._aggregate_score = None
        self._failed_reports = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop_monitoring(self) -> None:
        """Stop the file watcher thread and process any remaining lines."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

        # Final sweep: process any lines written after we stopped monitoring.
        # Retry a few times with delay to handle Docker volume mount sync lag —
        # subprocess.run() returns when the container exits, but the host may
        # not see the full file contents immediately via the volume mount.
        import time

        max_retries = 10
        for attempt in range(max_retries):
            # Reset aggregate so it gets recomputed with any newly scored problems
            self._aggregate_score = None
            self._process_remaining_lines()
            if self._completed_count >= self._total_problems:
                break
            # Only skip retries if no problems scored AND the output file
            # doesn't exist — that means the sandbox genuinely failed.
            # If the file exists but we scored 0, Docker volume sync may
            # not have caught up yet, so keep retrying.
            if self._completed_count == 0 and not self.output_file.exists():
                break
            if attempt < max_retries - 1:
                logging.info(
                    f"Scored {self._completed_count}/{self._total_problems} problems, "
                    f"waiting for remaining output (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(self.stop_retry_interval)

        # Retry any failed reports one final time before queuing
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
                    f"Progress reporting: {len(still_failed)} report(s) failed "
                    f"and added to retry queue"
                )
            elif still_failed:
                logging.warning(
                    f"Progress reporting: {len(still_failed)} report(s) failed "
                    f"(no retry queue available)"
                )

    def get_aggregate_score(self) -> Optional[Dict[str, Any]]:
        """Get the aggregated score after all problems complete.

        Returns:
            Dict with aggregate metrics (success_rate, etc.) or None if scoring
            was not initialized (no ProblemScorers).
        """
        # Compute aggregate if we haven't yet - even with 0 scored problems,
        # we want an aggregate that reflects the full suite denominator
        if self._aggregate_score is None:
            logging.info(
                f"Computing aggregate from {len(self._problem_scores)} scored problems"
            )
            self._compute_aggregate()

        return self._aggregate_score

    def get_problem_status(self, problem_id: str) -> ProblemStatus:
        """Return the status for a scored problem.

        Uses the same logic as _process_line: SUCCESS if score > 0, else FAILED.
        If the problem was never scored, returns FAILED as a safe default.

        Args:
            problem_id: The problem ID string.

        Returns:
            ProblemStatus.SUCCESS or ProblemStatus.FAILED.
        """
        score = self._problem_scores.get(problem_id)
        if score is not None and score > 0:
            return ProblemStatus.SUCCESS
        return ProblemStatus.FAILED

    def _read_inference_stats(self, problem_id: str) -> tuple[int, int]:
        """Read inference stats for a problem from the shared JSONL file.

        ProxyClient appends one line per problem to inference_stats.jsonl
        in the same directory as the output file. Returns (failure_count, total).
        """
        stats_path = self.output_file.parent / "inference_stats.jsonl"
        try:
            with open(stats_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if str(entry.get("problem_id")) == str(problem_id):
                        return entry.get("inference_failed", 0), entry.get("inference_total", 0)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        return 0, 0

    def _process_line(self, line: str) -> Optional[ProblemProgressUpdate]:
        """Parse a dialogue entry from sandbox output and score it.

        The sandbox output file contains one JSONL entry per completed problem.
        Each entry is a list of dialogue steps (the agent's conversation history).

        Returns:
            ProblemProgressUpdate with problem_id, status, and score, or None if parsing/scoring failed
        """
        if not self._scorers:
            logging.debug("Skipping scoring - no ProblemScorers initialized")
            return None

        try:
            dialogue = json.loads(line.strip())

            # Sandbox output is a list of dialogue steps
            if not isinstance(dialogue, list) or not dialogue:
                logging.debug(f"Skipping non-list or empty entry: {type(dialogue)}")
                return None

            # Extract problem_id from dialogue's extra_info (first step contains it)
            problem_id = None
            query = None
            if dialogue and dialogue[0].get("extra_info"):
                extra_info = dialogue[0]["extra_info"]
                problem_id = extra_info.get("problem_id")
                query = extra_info.get("query")  # Still extract for logging

            if not problem_id:
                logging.warning("Could not extract problem_id from dialogue extra_info")
                return None

            # Match problem_id to problem
            problem = self._id_to_problem.get(str(problem_id))
            if not problem:
                logging.warning(
                    f"No matching problem found for problem_id: {problem_id}"
                )
                return None

            # Use query from problem metadata (more reliable than dialogue)
            query = problem.get("query") or query

            # Check if already scored (prevent duplicate scoring)
            if problem_id in self._problem_scores:
                logging.debug(f"Problem {problem_id} already scored, skipping")
                return None

            # Select the scorer for this problem's category
            category = problem.get("category", "product").lower()
            scorer = self._scorers.get(category)
            if not scorer:
                logging.warning(
                    f"No scorer for category '{category}', skipping problem {problem_id}"
                )
                return None

            # Score the dialogue
            logging.info(
                f"Scoring problem {self._completed_count + 1}/{self._total_problems}: {query[:50]}..."
            )
            score_dict = scorer.score_problem(query=query, output=dialogue)

            # Extract numerical score (success_rate or total score)
            score = score_dict.get("gt", 0.0) if isinstance(score_dict, dict) else 0.0
            self._problem_scores[problem_id] = score
            if isinstance(score_dict, dict):
                self._problem_score_dicts[problem_id] = score_dict
                self._problem_categories[problem_id] = category
            self._completed_count += 1

            # Determine status based on score
            status = ProblemStatus.SUCCESS if score > 0 else ProblemStatus.FAILED

            logging.info(
                f"Problem {self._completed_count}/{self._total_problems} scored: {score:.4f} (query: {query[:50]}...)"
            )

            # Check if all problems complete - compute aggregate
            if self._completed_count == self._total_problems:
                self._compute_aggregate()

            # Read inference stats from sidecar file written by ProxyClient
            inf_failures, inf_total = self._read_inference_stats(problem_id)

            update = ProblemProgressUpdate(
                problem_id=UUID(problem_id)
                if isinstance(problem_id, str)
                else problem_id,
                status=status,
                score=score,
            )
            # Add inference stats to score_components_summary (Backend stores all JSONB fields)
            if inf_total > 0:
                from oro_sdk.models import ProblemProgressUpdateScoreComponentsSummaryType0
                summary = ProblemProgressUpdateScoreComponentsSummaryType0.from_dict({
                    "inference_failure_count": inf_failures,
                    "inference_total": inf_total,
                })
                update.score_components_summary = summary
            return update

        except json.JSONDecodeError:
            logging.warning(f"Failed to parse dialogue line: {line[:100]}...")
            return None
        except Exception as e:
            logging.error(f"Error scoring dialogue: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _compute_aggregate(self) -> None:
        """Compute aggregate score from all individual problem scores.

        Uses total_problems (from the suite) as the denominator, not just
        the number of scored problems. Problems that didn't appear in the
        sandbox output (timeout/error) count as failures (score=0).

        Produces the keys required by Backend validation (ORO-243):
        ground_truth_rate, success_rate, format_score, field_matching.
        """
        total = (
            self._total_problems
            if self._total_problems > 0
            else max(len(self._problem_scores), 1)
        )

        # Compute ground_truth_rate: fraction with gt >= 1
        gt_successes = sum(
            1 for s in self._problem_score_dicts.values() if s.get("gt", 0) >= 1
        )
        ground_truth_rate = gt_successes / total

        # Compute success_rate: category-aware
        # product: rule >= 1, shop: rule >= 1 AND shop >= 1, voucher: rule >= 1 AND budget >= 1
        success_count = 0
        for pid, sd in self._problem_score_dicts.items():
            cat = self._problem_categories.get(pid, "product")
            rule_ok = sd.get("rule", 0) >= 1
            if cat == "product" and rule_ok:
                success_count += 1
            elif cat == "shop" and rule_ok and sd.get("shop", 0) >= 1:
                success_count += 1
            elif cat == "voucher" and rule_ok and sd.get("budget", 0) >= 1:
                success_count += 1
        success_rate = success_count / total

        # Compute format_score: average format across scored problems (0 for unscored)
        format_total = sum(
            sd.get("format", 0) for sd in self._problem_score_dicts.values()
        )
        format_score = format_total / total

        # Compute field_matching: average rule score across scored problems (0 for unscored)
        rule_total = sum(sd.get("rule", 0) for sd in self._problem_score_dicts.values())
        field_matching = rule_total / total

        self._aggregate_score = {
            "ground_truth_rate": min(ground_truth_rate, 1.0),
            "success_rate": min(success_rate, 1.0),
            "format_score": min(format_score, 1.0),
            "field_matching": min(field_matching, 1.0),
            "total_problems": total,
            "successful_problems": success_count,
            "scored_problems": len(self._problem_scores),
        }

        logging.info(
            f"Aggregate score computed: success_rate={success_rate:.4f}, "
            f"gt_rate={ground_truth_rate:.4f}, format={format_score:.4f}, "
            f"field_matching={field_matching:.4f} "
            f"({success_count}/{total} problems succeeded, "
            f"{len(self._problem_scores)} scored)"
        )

    def _process_remaining_lines(self) -> None:
        """Process any lines in the output file that weren't caught during monitoring.

        This is called after stop_monitoring() to handle the case where the sandbox
        writes all output at the end (rather than incrementally).
        """
        if not self.output_file.exists():
            logging.warning(f"Output file does not exist: {self.output_file}")
            return

        try:
            with open(self.output_file) as f:
                f.seek(self._last_position)
                remaining_lines = f.readlines()
                self._last_position = f.tell()

            if remaining_lines:
                logging.info(
                    f"Processing {len(remaining_lines)} remaining lines from output file"
                )

            for line in remaining_lines:
                if line.strip():
                    progress = self._process_line(line)
                    if progress:
                        self._report_progress(progress)

            # Compute aggregate if we processed any problems but haven't computed it yet
            if self._problem_scores and self._aggregate_score is None:
                self._compute_aggregate()

        except Exception as e:
            logging.error(f"Error processing remaining lines: {e}")
            import traceback

            traceback.print_exc()

    def report_unscored_as_timed_out(self) -> None:
        """Report problems that were never scored as TIMED_OUT.

        Called after sandbox completes to ensure all problems have a final status
        instead of staying as 'Pending' forever.
        """
        scored_ids = set(self._problem_scores.keys())
        all_ids = set(self._id_to_problem.keys())
        unscored_ids = all_ids - scored_ids

        if not unscored_ids:
            return

        logging.info(f"Reporting {len(unscored_ids)} unscored problems as TIMED_OUT")
        for problem_id_str in unscored_ids:
            try:
                from uuid import UUID

                progress = ProblemProgressUpdate(
                    problem_id=UUID(problem_id_str),
                    status=ProblemStatus.TIMED_OUT,
                    score=0.0,
                )
                self._report_progress(progress)
            except Exception as e:
                logging.warning(
                    f"Failed to report timed-out problem {problem_id_str}: {e}"
                )

    def _run(self) -> None:
        """Background thread main loop - tails output file."""
        while not self._stop_event.is_set():
            try:
                if self.output_file.exists():
                    # Check if file was truncated/rotated
                    file_size = self.output_file.stat().st_size
                    if file_size < self._last_position:
                        logging.info(
                            f"Output file was truncated/rotated, resetting position "
                            f"(was {self._last_position}, now {file_size})"
                        )
                        self._last_position = 0

                    with open(self.output_file) as f:
                        f.seek(self._last_position)
                        new_lines = f.readlines()
                        self._last_position = f.tell()

                    for line in new_lines:
                        if line.strip():
                            progress = self._process_line(line)
                            if progress:
                                self._report_progress(progress)
            except (OSError, IOError) as e:
                # File access error - reset position and try again
                logging.warning(f"Error reading progress file, resetting position: {e}")
                self._last_position = 0
            except Exception as e:
                logging.warning(f"Error processing progress file: {e}")

            self._stop_event.wait(self.poll_interval)

    def _report_progress(self, progress: ProblemProgressUpdate) -> None:
        """Report a single problem's progress to Backend.

        The SDK's BittensorAuthClient already retries transient errors (429,
        502-504, connection errors) with exponential backoff.  If the call
        still fails, the update is queued for a final retry in stop_monitoring()
        and persisted to disk if that also fails.
        """
        try:
            self.backend_client.report_progress(
                self.eval_run_id,
                [progress],
            )
            logging.info(
                f"Reported progress for {progress.problem_id}: "
                f"{progress.status}, score={progress.score:.4f}"
            )
        except Exception as e:
            self._failed_reports.append(progress)
            logging.warning(f"Failed to report progress for {progress.problem_id}: {e}")
