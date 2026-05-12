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
from .types import EnvelopeMeta, ProblemResult


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
        """Score a single problem end-to-end. Runs in a worker thread."""
        if not self.scorers:
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
                meta.execution_time
                if meta is not None
                else extra_info.get("execution_time")
            )
            query = problem.get("query") or extra_info.get("query")
            category = problem.get("category", "product").lower()

            scorer = self.scorers.get(category)
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

            reasoning = self._reasoning_judge.score(dialogue, problem_id)

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
