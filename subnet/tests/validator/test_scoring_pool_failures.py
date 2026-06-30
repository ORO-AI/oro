"""Scoring-pool failure attribution (ORO-1461).

Before this change, any non-SUCCESS path through `_score_problem` left
the problem unscored, and the end-of-run sweep marked it `TIMED_OUT`.
That masked the real cause (missing voucher metadata, scorer crash, etc.)
and made staging triage hard. Each test below exercises one failure path
and asserts the result is a FAILED `ProblemResult` with the expected
`failure_reason` instead of a silent drop.
"""

import threading
from unittest.mock import MagicMock

from oro_sdk.models import ProblemStatus

from validator.scoring_pool import ScoringPool
from validator.types import ProblemFailureReason


def _make_pool(problems, scorers, judge=None):
    results = {}
    envelope_meta = {}
    id_to_problem = {p["problem_id"]: p for p in problems}
    lock = threading.Lock()
    judge = judge or MagicMock()
    judge.score.return_value = {
        "reasoning_score": None,
        "reasoning_explanation": "",
        "reasoning_model": "",
        "reasoning_inf_failed": 0,
        "reasoning_inf_total": 0,
        "reasoning_inf_402": 0,
    }
    pool = ScoringPool.__new__(ScoringPool)
    pool._results = results
    pool._envelope_meta = envelope_meta
    pool._id_to_problem = id_to_problem
    pool._lock = lock
    pool._total_problems = len(problems)
    pool._reasoning_judge = judge
    pool.futures = {}
    pool.scorers = scorers
    return pool, results


def _dialogue():
    return [{"role": "u", "content": "x", "extra_info": {"step": 1}}]


def test_voucher_missing_metadata_recorded_not_dropped():
    problem = {"problem_id": "p1", "query": "q", "category": "voucher"}  # no voucher
    pool, results = _make_pool([problem], scorers={"voucher": MagicMock()})

    pool._score_problem(_dialogue(), "p1")

    assert "p1" in results
    assert results["p1"].status == ProblemStatus.FAILED
    assert results["p1"].failure_reason == ProblemFailureReason.MISSING_METADATA


def test_scoring_exception_recorded_not_dropped():
    scorer = MagicMock()
    scorer.score_problem.side_effect = KeyError("missing field")
    problem = {"problem_id": "p1", "query": "q", "category": "product"}
    pool, results = _make_pool([problem], scorers={"product": scorer})

    pool._score_problem(_dialogue(), "p1")

    assert results["p1"].failure_reason == ProblemFailureReason.SCORING_EXCEPTION
    assert results["p1"].status == ProblemStatus.FAILED


def test_scoring_returned_none_recorded_not_dropped():
    scorer = MagicMock()
    scorer.score_problem.return_value = None
    problem = {"problem_id": "p1", "query": "q", "category": "product"}
    pool, results = _make_pool([problem], scorers={"product": scorer})

    pool._score_problem(_dialogue(), "p1")

    assert results["p1"].failure_reason == ProblemFailureReason.SCORING_RETURNED_NONE
    assert results["p1"].status == ProblemStatus.FAILED


def test_unknown_problem_recorded_not_dropped():
    pool, results = _make_pool(
        [{"problem_id": "p1", "query": "q", "category": "product"}],
        scorers={"product": MagicMock()},
    )

    pool._score_problem(_dialogue(), "p999")

    assert results["p999"].failure_reason == ProblemFailureReason.UNKNOWN_PROBLEM


def test_no_scorer_for_category_recorded_not_dropped():
    problem = {"problem_id": "p1", "query": "q", "category": "voucher"}
    pool, results = _make_pool([problem], scorers={"product": MagicMock()})

    pool._score_problem(_dialogue(), "p1")

    assert results["p1"].failure_reason == ProblemFailureReason.NO_SCORER_FOR_CATEGORY


def test_empty_dialogue_recorded_not_dropped():
    problem = {"problem_id": "p1", "query": "q", "category": "product"}
    pool, results = _make_pool([problem], scorers={"product": MagicMock()})

    pool._score_problem([], "p1")

    assert results["p1"].failure_reason == ProblemFailureReason.NO_DIALOGUE


def test_success_path_unaffected_by_attribution_changes():
    scorer = MagicMock()
    scorer.score_problem.return_value = {"score": 1.0}
    problem = {"problem_id": "p1", "query": "q", "category": "product"}
    pool, results = _make_pool([problem], scorers={"product": scorer})

    # is_problem_successful checks score_dict structure; stub the helper
    # by giving the scorer a score_dict the real check accepts. The
    # simplest passing shape is {"score": 1.0} which routes through to
    # is_problem_successful returning False unless category-specific
    # fields are present — for this regression test we only assert that
    # the result is published and carries no failure_reason.
    pool._score_problem(_dialogue(), "p1")

    assert "p1" in results
    assert results["p1"].failure_reason is None
