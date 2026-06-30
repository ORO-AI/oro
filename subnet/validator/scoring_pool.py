"""Thread-pool-backed per-problem scoring.

Owns the executor, the per-category ProblemScorers, and the in-flight
futures table. Workers read from the shared envelope-meta and
id-to-problem maps, then publish ProblemResults back into the shared
results dict — all under a single lock owned by ProgressReporter.
"""

from __future__ import annotations

import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List

from bittensor.utils.btlogging import logging

from oro_sdk.models import ProblemStatus

from src.agent.problem_scorer import ProblemScorer, clear_product_cache
from src.agent.scoring import is_problem_successful
from src.agent.types import ProblemDict
from subnet.sandbox import attach_title_embeddings

from .reasoning_judge import ReasoningJudge
from .types import EnvelopeMeta, ProblemFailureReason, ProblemResult


DEFAULT_SCORING_WORKERS = 4


class ScoringPool:
    """Owns the scoring thread pool and the per-category scorers."""

    def __init__(
        self,
        problems: List[ProblemDict],
        results: Dict[str, ProblemResult],
        envelope_meta: Dict[str, EnvelopeMeta],
        id_to_problem: Dict[str, ProblemDict],
        lock: threading.Lock,
        reasoning_judge: ReasoningJudge,
        max_workers: int = DEFAULT_SCORING_WORKERS,
    ):
        self._results = results
        self._envelope_meta = envelope_meta
        self._id_to_problem = id_to_problem
        self._lock = lock
        self._total_problems = len(problems)
        self._reasoning_judge = reasoning_judge
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="scorer"
        )
        self.futures: Dict[str, Future] = {}
        self.scorers: Dict[str, Any] = {}
        self._initialize_scorers(problems)

    def has_future(self, problem_id: str) -> bool:
        return problem_id in self.futures

    def pending_count(self) -> int:
        return sum(1 for f in self.futures.values() if not f.done())

    def submit(self, problem_id: str, dialogue: list) -> None:
        future = self._executor.submit(self._score_problem, dialogue, problem_id)
        self.futures[problem_id] = future

    def collect_completed(self) -> None:
        """Reap completed futures and log any worker exceptions."""
        completed = [pid for pid, f in self.futures.items() if f.done()]
        for pid in completed:
            future = self.futures.pop(pid)
            exc = future.exception()
            if exc:
                logging.error(f"Scoring worker failed for {pid}: {exc}")

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)

    def _initialize_scorers(self, problems: List[ProblemDict]) -> None:
        """Build per-category ProblemScorers from problem metadata."""
        try:
            clear_product_cache()
            category_rewards: Dict[str, Dict] = {}
            category_vouchers: Dict[str, Dict] = {}
            for problem in problems:
                query = problem.get("query")
                reward = problem.get("reward")
                category = problem.get("category", "product").lower()
                if category not in ("product", "shop", "voucher"):
                    category = "product"
                if query and reward:
                    attach_title_embeddings(
                        reward, problem.get("reward_title_embeddings")
                    )
                    category_rewards.setdefault(category, {})[query] = reward
                if category == "voucher":
                    voucher = problem.get("voucher")
                    if query and voucher:
                        category_vouchers.setdefault(category, {})[query] = voucher
            for category, rewards in category_rewards.items():
                vouchers = category_vouchers.get(category, {})
                self.scorers[category] = ProblemScorer(
                    task=category, rewards=rewards, vouchers=vouchers
                )
                logging.info(
                    f"Created ProblemScorer for '{category}' with {len(rewards)} problems"
                )
            logging.info(
                f"Initialized {len(self.scorers)} scorers: {list(self.scorers.keys())}"
            )
        except (ImportError, OSError, ValueError, TypeError, KeyError) as e:
            logging.error(f"Failed to initialize ProblemScorers: {e}")
            self.scorers = {}

    def _score_problem(self, dialogue: list, problem_id: str) -> None:
        """Score a single problem end-to-end. Runs in a worker thread.

        Always publishes a ProblemResult for ``problem_id``. Failure paths
        write a FAILED result with a specific ``failure_reason`` instead
        of leaving the problem unscored — the end-of-run sweep would
        otherwise mark them TIMED_OUT and mask the real cause (ORO-1461).
        """
        pid = str(problem_id)
        problem = self._id_to_problem.get(pid)
        category = problem.get("category", "product").lower() if problem else "product"

        if not self.scorers:
            self._record_failure(
                pid, category, ProblemFailureReason.NO_SCORER_FOR_CATEGORY
            )
            return
        if not isinstance(dialogue, list) or not dialogue:
            self._record_failure(pid, category, ProblemFailureReason.NO_DIALOGUE)
            return
        if not problem:
            self._record_failure(pid, category, ProblemFailureReason.UNKNOWN_PROBLEM)
            return

        scorer = self.scorers.get(category)
        if not scorer:
            self._record_failure(
                pid, category, ProblemFailureReason.NO_SCORER_FOR_CATEGORY
            )
            return

        extra_info = (dialogue[0].get("extra_info") or {}) if dialogue else {}
        with self._lock:
            meta = self._envelope_meta.get(pid)
        execution_time = (
            meta.execution_time
            if meta is not None
            else extra_info.get("execution_time")
        )
        query = problem.get("query") or extra_info.get("query")
        inf_failures = meta.inference_failure_count if meta else 0
        inf_total = meta.inference_total if meta else 0

        # Voucher problems must carry a `voucher` budget dict for the scorer
        # to evaluate the constraint. Without it, ProblemScorer raises deep
        # inside score_problem and the result was silently lost (ORO-1461).
        if category == "voucher" and not problem.get("voucher"):
            self._record_failure(
                pid,
                category,
                ProblemFailureReason.MISSING_METADATA,
                execution_time=execution_time,
                inf_failures=inf_failures,
                inf_total=inf_total,
                extra={"missing_field": "voucher"},
            )
            return

        with self._lock:
            scored_count = len(self._results) + 1
        logging.info(
            f"Scoring problem {scored_count}/{self._total_problems}: {query[:50]}..."
        )

        try:
            score_dict = scorer.score_problem(query=query, output=dialogue)
        except Exception as e:
            logging.warning(
                f"AUDIT scoring_exception problem_id={pid} category={category} "
                f"exc={type(e).__name__} msg={str(e)[:200]}"
            )
            traceback.print_exc()
            self._record_failure(
                pid,
                category,
                ProblemFailureReason.SCORING_EXCEPTION,
                execution_time=execution_time,
                inf_failures=inf_failures,
                inf_total=inf_total,
                extra={"exception_type": type(e).__name__},
            )
            return

        if not isinstance(score_dict, dict):
            logging.warning(
                f"AUDIT scoring_returned_none problem_id={pid} category={category} "
                f"type={type(score_dict).__name__}"
            )
            self._record_failure(
                pid,
                category,
                ProblemFailureReason.SCORING_RETURNED_NONE,
                execution_time=execution_time,
                inf_failures=inf_failures,
                inf_total=inf_total,
            )
            return

        is_successful = is_problem_successful(score_dict, category)
        score = 1.0 if is_successful else 0.0
        status = ProblemStatus.SUCCESS if is_successful else ProblemStatus.FAILED

        reasoning = self._reasoning_judge.score(dialogue, problem_id)

        result = ProblemResult(
            problem_id=pid,
            category=category,
            status=status,
            score=score,
            score_dict=score_dict,
            inference_failures=inf_failures,
            inference_total=inf_total,
            execution_time=execution_time,
            **reasoning,
        )
        with self._lock:
            self._results[pid] = result
            completed = len(self._results)

        logging.info(
            f"Problem {completed}/{self._total_problems} scored: "
            f"{score:.4f} (query: {query[:50]}...)"
        )

    def _record_failure(
        self,
        problem_id: str,
        category: str,
        reason: ProblemFailureReason,
        *,
        execution_time: float | None = None,
        inf_failures: int = 0,
        inf_total: int = 0,
        extra: dict | None = None,
    ) -> None:
        """Write a FAILED ProblemResult with an attributed failure_reason."""
        logging.warning(
            f"AUDIT problem_failure problem_id={problem_id} category={category} "
            f"reason={reason.value} extra={extra or {}}"
        )
        result = ProblemResult(
            problem_id=problem_id,
            category=category,
            status=ProblemStatus.FAILED,
            score=0.0,
            inference_failures=inf_failures,
            inference_total=inf_total,
            execution_time=execution_time,
            failure_reason=reason,
        )
        with self._lock:
            self._results[problem_id] = result
