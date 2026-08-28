"""Reasoning coverage must be complete before an evaluation can score."""

import threading

from oro_sdk.models import ProblemStatus

from validator.progress_reporter import ProgressReporter
from validator.types import ProblemResult


def _reporter(*results: ProblemResult) -> ProgressReporter:
    reporter = object.__new__(ProgressReporter)
    reporter._lock = threading.Lock()
    reporter._results = {result.problem_id: result for result in results}
    return reporter


def _result(problem_id: str, score: float | None, *, credit_402: bool = False):
    return ProblemResult(
        problem_id=problem_id,
        category="product",
        status=ProblemStatus.SUCCESS,
        score=1.0,
        reasoning_score=score,
        reasoning_inf_402=int(credit_402),
        reasoning_judgment_expected=True,
    )


def test_complete_coverage_scores_normally():
    reporter = _reporter(_result("p1", 0.8), _result("p2", 1.0))

    assert reporter.get_reasoning_failure_reason() is None
    assert reporter.get_reasoning_data()["reasoning_quality"] == 0.9


def test_confirmed_credit_failure_fails_the_whole_run():
    reporter = _reporter(_result("p1", 0.9), _result("p2", None, credit_402=True))

    assert "insufficient miner credits" in reporter.get_reasoning_failure_reason()
    assert reporter.get_reasoning_data()["reasoning_quality"] == 0.0


def test_other_missing_judgments_are_infrastructure_failures():
    reporter = _reporter(_result("p1", 0.9), _result("p2", None))

    assert "infrastructure failure" in reporter.get_reasoning_failure_reason()
