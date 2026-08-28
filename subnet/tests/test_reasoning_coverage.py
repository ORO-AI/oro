"""Problem-level reasoning coverage must be complete before scoring."""

import threading

from oro_sdk.models import ProblemStatus

from validator.progress_reporter import (
    ProgressReporter,
    reasoning_coverage_failure_reason,
)
from validator.types import ProblemResult


def _result(
    problem_id: int,
    status: str,
    *,
    score: float | None = None,
    failure_class: str | None = None,
) -> ProblemResult:
    return ProblemResult(
        problem_id=str(problem_id),
        category="product",
        status=ProblemStatus.SUCCESS,
        score=1.0,
        reasoning_score=score,
        reasoning_judgment_expected=True,
        reasoning_judgment_status=status,
        reasoning_failure_class=failure_class,
        reasoning_inf_total=1 if status != "skipped" else 0,
        reasoning_inf_failed=1 if status == "failed" else 0,
        reasoning_inf_402=1 if failure_class == "insufficient_miner_credits" else 0,
        reasoning_inf_402_credits=(
            1 if failure_class == "insufficient_miner_credits" else 0
        ),
    )


def _summary(results: list[ProblemResult]):
    reporter = object.__new__(ProgressReporter)
    reporter._lock = threading.Lock()
    reporter._results = {result.problem_id: result for result in results}
    return reporter.get_reasoning_data()


def test_thirty_of_thirty_valid_is_authoritative():
    summary = _summary([_result(i, "valid", score=0.9) for i in range(30)])

    assert summary["reasoning_judgments_expected"] == 30
    assert summary["reasoning_judgments_valid"] == 30
    assert summary["reasoning_quality"] == 0.9
    assert reasoning_coverage_failure_reason(summary) is None


def test_first_credit_402_cannot_finalize_zero_sample():
    results = [
        _result(0, "failed", failure_class="insufficient_miner_credits"),
        *[
            _result(i, "skipped", failure_class="insufficient_miner_credits")
            for i in range(1, 30)
        ],
    ]
    summary = _summary(results)

    assert summary["reasoning_judgments_valid"] == 0
    assert summary["reasoning_judgments_failed"] == 1
    assert summary["reasoning_judgments_skipped"] == 29
    assert summary["reasoning_quality"] == 0.0
    assert "insufficient miner credits" in reasoning_coverage_failure_reason(summary)


def test_minions_partial_sample_cannot_distort_score():
    results = [_result(i, "valid", score=0.9) for i in range(3)]
    results.append(
        _result(3, "failed", failure_class="insufficient_miner_credits")
    )
    results.extend(
        _result(i, "skipped", failure_class="insufficient_miner_credits")
        for i in range(4, 30)
    )

    summary = _summary(results)

    assert summary["reasoning_judgments_valid"] == 3
    assert summary["reasoning_judgments_failed"] == 1
    assert summary["reasoning_judgments_skipped"] == 26
    assert summary["reasoning_quality"] == 0.0
    assert reasoning_coverage_failure_reason(summary) is not None


def test_concurrent_completion_order_does_not_change_coverage():
    results = [_result(i, "valid", score=0.7) for i in range(30)]

    forward = _summary(results)
    reverse = _summary(list(reversed(results)))

    assert forward == reverse


def test_terminal_problem_does_not_require_a_reasoning_judgment():
    terminal = ProblemResult(
        problem_id="terminal",
        category="product",
        status=ProblemStatus.TIMED_OUT,
        score=0.0,
    )

    summary = _summary([_result(1, "valid", score=0.8), terminal])

    assert summary["reasoning_judgments_expected"] == 1
    assert summary["reasoning_judgments_valid"] == 1
    assert reasoning_coverage_failure_reason(summary) is None


def test_run_with_no_scorable_trajectories_has_complete_empty_coverage():
    terminal = ProblemResult(
        problem_id="terminal",
        category="product",
        status=ProblemStatus.TIMED_OUT,
        score=0.0,
    )

    summary = _summary([terminal])

    assert summary["reasoning_judgments_expected"] == 0
    assert summary["reasoning_judgments_valid"] == 0
    assert summary["reasoning_failure_class"] is None
    assert reasoning_coverage_failure_reason(summary) is None
