"""Reasoning scoring treats a missing per-problem judgment as 0.

The reasoning judge no longer blocks a submission on incomplete coverage: a
missed/failed judgment simply counts as 0 in the reasoning-quality average and
the run always completes. (The judge is being removed; this is the pre-#266
behavior restored.)
"""

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


def test_complete_coverage_averages_judged_scores():
    reporter = _reporter(_result("p1", 0.8), _result("p2", 1.0))

    assert reporter.get_reasoning_data()["reasoning_quality"] == 0.9


def test_single_missing_judgment_counts_as_zero():
    reporter = _reporter(_result("p1", 0.8), _result("p2", None))

    # 0.8 for the judged problem, 0 for the missing one, averaged over both.
    assert reporter.get_reasoning_data()["reasoning_quality"] == 0.4


def test_all_missing_judgments_score_zero():
    reporter = _reporter(_result("p1", None), _result("p2", None))

    assert reporter.get_reasoning_data()["reasoning_quality"] == 0.0


def test_confirmed_402_counts_as_zero_and_still_completes():
    # A confirmed out-of-credits 402 no longer fails the run; it scores 0.
    reporter = _reporter(_result("p1", 0.9), _result("p2", None, credit_402=True))

    data = reporter.get_reasoning_data()
    assert data["reasoning_quality"] == 0.45  # 0.9 / 2
    assert data["judge_inference_402"] == 1


def test_no_scorable_trajectories_scores_zero():
    reporter = _reporter(
        ProblemResult(
            problem_id="p1",
            category="product",
            status=ProblemStatus.SUCCESS,
            score=1.0,
            reasoning_judgment_expected=False,
        )
    )

    assert reporter.get_reasoning_data()["reasoning_quality"] == 0.0


def test_reasoning_failure_gate_is_removed():
    # The fail-closed coverage gate (#266) is gone: the reporter no longer
    # exposes a reasoning failure reason at all.
    assert not hasattr(ProgressReporter, "get_reasoning_failure_reason")
