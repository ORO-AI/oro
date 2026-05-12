"""LLM reasoning-quality judge with circuit-breaker protection.

Scores one dialogue per call from a scoring worker thread. A bad token or a
flapping inference provider can otherwise turn into a per-problem retry
storm, so consecutive total-failure outcomes (or exceptions) trip a circuit
breaker that suppresses further calls for the rest of the run.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from bittensor.utils.btlogging import logging


# Trip after this many consecutive total-failure / exception outcomes.
# Low by design — a bad token usually fails on the first call.
CIRCUIT_BREAKER_THRESHOLD = 3

# Returned when the judge is disabled, tripped, or errored. Keys are the
# reasoning_* fields ProblemResult accepts as **kwargs.
_EMPTY_RESULT: Dict[str, Any] = {
    "reasoning_score": None,
    "reasoning_explanation": "",
    "reasoning_model": "",
    "reasoning_inf_failed": 0,
    "reasoning_inf_total": 0,
}


class ReasoningJudge:
    """Score reasoning quality for a single problem."""

    def __init__(
        self,
        chutes_access_token: Optional[str],
        inference_provider: str,
        backend_base_url: str,
    ):
        self._token = chutes_access_token
        self._provider = inference_provider
        self._backend_base_url = backend_base_url
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._circuit_open = False

    def score(self, dialogue: list, problem_id: str) -> Dict[str, Any]:
        """Return reasoning-fields dict to unpack into ProblemResult."""
        with self._lock:
            should_judge = bool(self._token and not self._circuit_open)
        if not should_judge:
            return dict(_EMPTY_RESULT)

        try:
            from src.agent.reasoning_scorer import score_reasoning_quality

            judge_result = score_reasoning_quality(
                dialogue,
                api_key=self._token,
                provider=self._provider,
                backend_url=self._backend_base_url,
            )

            inf_failed = judge_result["inference_failed"]
            inf_total = judge_result["inference_total"]
            with self._lock:
                if inf_total > 0 and inf_failed == inf_total:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
                        self._circuit_open = True
                        logging.error(
                            "Reasoning judge circuit breaker tripped: "
                            f"{CIRCUIT_BREAKER_THRESHOLD} consecutive problems "
                            "with 100% judge failure."
                        )
                else:
                    self._consecutive_failures = 0

            logging.info(
                f"Reasoning score: {judge_result['score']:.2f} "
                f"(problem={problem_id}, model={judge_result['model']})"
            )
            return {
                "reasoning_score": judge_result["score"],
                "reasoning_explanation": judge_result["explanation"],
                "reasoning_model": judge_result["model"],
                "reasoning_inf_failed": inf_failed,
                "reasoning_inf_total": inf_total,
            }
        except Exception as e:
            logging.warning(f"Reasoning judge failed for {problem_id}: {e}")
            with self._lock:
                self._consecutive_failures += 1
                if self._consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
                    self._circuit_open = True
                    logging.error(
                        "Reasoning judge circuit breaker tripped after exceptions."
                    )
            return dict(_EMPTY_RESULT)
