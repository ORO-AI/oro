"""Validator parses envelope format from ORO-907."""

import json
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from oro_sdk.models import ProblemStatus

from validator.progress_reporter import ProgressReporter
from validator.types import ProblemResult


def _write_envelope(path: Path, **fields):
    envelope = {
        "problem_id": fields.get("problem_id", "p1"),
        "status": fields.get("status", "SUCCESS"),
        "execution_time": fields.get("execution_time", 1.0),
        "inference_failure_count": fields.get("inference_failure_count", 0),
        "inference_total": fields.get("inference_total", 1),
        "error": fields.get("error"),
        "dialogue": fields.get(
            "dialogue", [{"role": "u", "content": "x", "extra_info": {"step": 1}}]
        ),
    }
    with open(path, "a") as f:
        f.write(json.dumps(envelope) + "\n")


# Fixed problem UUIDs the tests reuse — must be valid UUIDs because
# _batch_report calls UUID(r.problem_id), but the in-memory _results
# dict tolerates any string. Use valid UUIDs throughout for safety.
_P1 = "11111111-1111-1111-1111-111111111111"
_P2 = "22222222-2222-2222-2222-222222222222"
_P3 = "33333333-3333-3333-3333-333333333333"


@pytest.fixture
def reporter(tmp_path) -> ProgressReporter:
    out = tmp_path / "output.jsonl"
    out.touch()
    problems = [
        {"problem_id": _P1, "query": "q1", "category": "product"},
        {"problem_id": _P2, "query": "q2", "category": "product"},
        {"problem_id": _P3, "query": "q3", "category": "product"},
    ]
    backend = MagicMock()
    rep = ProgressReporter(
        backend_client=backend,
        eval_run_id=uuid4(),
        output_file=out,
        problems=problems,
        workspace_dir=tmp_path,
    )
    # Disable scoring side effects — tests assert on dispatch/no-dispatch,
    # not on what scoring computes. Replace scorers with a stub that always
    # produces a clean SUCCESS so dispatched futures complete promptly.
    rep._scoring_pool.scorers = {}
    return rep


class TestEnvelopeParsing:
    def test_success_envelope_dispatches_to_scoring(self, reporter, tmp_path):
        _write_envelope(tmp_path / "output.jsonl", problem_id=_P1, status="SUCCESS")
        reporter._envelope_dispatcher.read_and_dispatch()
        assert _P1 in reporter._scoring_pool.futures

    def test_failure_envelope_records_without_scoring(self, reporter, tmp_path):
        _write_envelope(
            tmp_path / "output.jsonl",
            problem_id=_P1,
            status="FAILED",
            dialogue=None,
            error={"type": "RuntimeError", "message": "boom"},
        )
        reporter._envelope_dispatcher.read_and_dispatch()
        assert _P1 not in reporter._scoring_pool.futures
        assert reporter._results[_P1].status == ProblemStatus.FAILED

    def test_timeout_envelope_records_without_scoring(self, reporter, tmp_path):
        _write_envelope(
            tmp_path / "output.jsonl",
            problem_id=_P1,
            status="TIMED_OUT",
            dialogue=None,
            error={"type": "TimeoutError", "message": "..."},
        )
        reporter._envelope_dispatcher.read_and_dispatch()
        assert _P1 not in reporter._scoring_pool.futures
        assert reporter._results[_P1].status == ProblemStatus.TIMED_OUT

    def test_inference_counts_come_from_envelope(self, reporter, tmp_path):
        _write_envelope(
            tmp_path / "output.jsonl",
            problem_id=_P1,
            status="FAILED",
            dialogue=None,
            inference_failure_count=2,
            inference_total=7,
            error={"type": "RuntimeError", "message": "x"},
        )
        reporter._envelope_dispatcher.read_and_dispatch()
        # Inference counts captured from envelope at dispatch time.
        meta = reporter._envelope_dispatcher.envelope_meta[_P1]
        assert (meta.inference_failure_count, meta.inference_total) == (2, 7)
        # And materialized into the terminal result.
        assert reporter._results[_P1].inference_failures == 2
        assert reporter._results[_P1].inference_total == 7

    def test_no_inference_stats_jsonl_read(self, reporter, tmp_path):
        # Make sidecar unreadable to prove validator does not touch it.
        sidecar = tmp_path / "inference_stats.jsonl"
        sidecar.write_text("CORRUPT NOT JSON\n")
        _write_envelope(
            tmp_path / "output.jsonl",
            problem_id=_P1,
            status="FAILED",
            dialogue=None,
            inference_failure_count=0,
            inference_total=1,
            error={"type": "RuntimeError", "message": "x"},
        )
        # Should not raise. If validator reads sidecar, JSONDecodeError surfaces.
        reporter._envelope_dispatcher.read_and_dispatch()
        assert reporter._results[_P1].status == ProblemStatus.FAILED

    def test_validator_no_longer_has_read_inference_stats(self, reporter):
        """Sidecar reader is gone from validator side."""
        assert not hasattr(reporter, "_read_inference_stats")

    def test_execution_time_from_envelope(self, reporter, tmp_path):
        _write_envelope(
            tmp_path / "output.jsonl",
            problem_id=_P1,
            status="TIMED_OUT",
            dialogue=None,
            execution_time=42.0,
            error={"type": "TimeoutError", "message": "..."},
        )
        reporter._envelope_dispatcher.read_and_dispatch()
        assert reporter._results[_P1].execution_time == 42.0

    def test_malformed_line_skipped(self, reporter, tmp_path):
        out = tmp_path / "output.jsonl"
        with open(out, "a") as f:
            f.write("not json\n")
        _write_envelope(
            out,
            problem_id=_P1,
            status="FAILED",
            dialogue=None,
            error={"type": "RuntimeError", "message": "x"},
        )
        reporter._envelope_dispatcher.read_and_dispatch()
        assert reporter._results[_P1].status == ProblemStatus.FAILED


class TestSweepNarrowing:
    def test_sweep_skips_problems_already_in_envelope(self, reporter, tmp_path):
        # p1 has FAILED envelope. Sweep at deadline must NOT overwrite to TIMED_OUT.
        _write_envelope(
            tmp_path / "output.jsonl",
            problem_id=_P1,
            status="FAILED",
            dialogue=None,
            error={"type": "RuntimeError", "message": "x"},
        )
        reporter._envelope_dispatcher.read_and_dispatch()
        reporter._envelope_dispatcher.mark_remaining_timed_out()
        assert reporter._results[_P1].status == ProblemStatus.FAILED

    def test_sweep_marks_only_never_seen_problems(self, reporter):
        # No envelope written. Sweep should mark all three TIMED_OUT.
        reporter._envelope_dispatcher.mark_remaining_timed_out()
        assert reporter._results[_P1].status == ProblemStatus.TIMED_OUT
        assert reporter._results[_P2].status == ProblemStatus.TIMED_OUT
        assert reporter._results[_P3].status == ProblemStatus.TIMED_OUT

    def test_sweep_skips_problems_with_in_flight_future(self, reporter):
        # p1 was already submitted to the scoring pool and is still running
        # (future not done) when the caller's hard-timeout/no-output-file
        # path calls the sweep without checking pending_count() first.
        # Marking it TIMED_OUT here would be clobbered moments later when
        # the worker writes the real score, but only *after* a TIMED_OUT/0.0
        # has already gone out in a batch report with nothing left to
        # correct it — so the sweep must leave it alone.
        from concurrent.futures import Future

        still_running: Future = Future()
        reporter._scoring_pool.futures[_P1] = still_running

        reporter._envelope_dispatcher.mark_remaining_timed_out()

        assert _P1 not in reporter._results
        assert reporter._results[_P2].status == ProblemStatus.TIMED_OUT
        assert reporter._results[_P3].status == ProblemStatus.TIMED_OUT

    def test_sweep_marks_terminal_future_with_no_result(self, reporter):
        # p1's future finished (e.g. _score_problem hit an early-return path
        # or swallowed an internal exception) but wrote nothing to _results,
        # and collect_completed() hasn't reaped it yet. has_future() alone
        # can't distinguish this from a still-running future -- the sweep
        # must still mark it TIMED_OUT, since nothing else ever will.
        from concurrent.futures import Future

        finished_with_no_result: Future = Future()
        finished_with_no_result.set_result(None)
        reporter._scoring_pool.futures[_P1] = finished_with_no_result

        reporter._envelope_dispatcher.mark_remaining_timed_out()

        assert reporter._results[_P1].status == ProblemStatus.TIMED_OUT
        assert reporter._results[_P2].status == ProblemStatus.TIMED_OUT
        assert reporter._results[_P3].status == ProblemStatus.TIMED_OUT

    def test_sweep_does_not_clobber_score_written_between_snapshot_and_write(
        self, reporter
    ):
        # p1 is genuinely unscored when the sweep captures scored_ids, but a
        # worker writes its real score and finishes (future.done() becomes
        # True) right as/after has_pending_future(p1) is evaluated -- which
        # reports "not pending" (done), same as a resultless terminal future.
        # Without a final re-check under the lock at write time, the sweep
        # would overwrite the real score with TIMED_OUT/0.0. It must not.
        from concurrent.futures import Future

        real_result = ProblemResult(
            problem_id=_P1,
            category="product",
            status=ProblemStatus.SUCCESS,
            score=1.0,
        )
        finished_future: Future = Future()
        finished_future.set_result(None)
        reporter._scoring_pool.futures[_P1] = finished_future

        real_has_pending_future = reporter._scoring_pool.has_pending_future

        def racing_has_pending_future(pid):
            result = real_has_pending_future(pid)
            if pid == _P1:
                # Simulate the worker's write landing in the window between
                # the scored_ids snapshot (already taken, p1 absent) and
                # this check -- exactly the race the fix must survive.
                reporter._results[_P1] = real_result
            return result

        reporter._scoring_pool.has_pending_future = racing_has_pending_future

        reporter._envelope_dispatcher.mark_remaining_timed_out()

        assert reporter._results[_P1] is real_result
        assert reporter._results[_P1].status == ProblemStatus.SUCCESS
        assert reporter._results[_P1].score == 1.0
        assert reporter._results[_P2].status == ProblemStatus.TIMED_OUT
        assert reporter._results[_P3].status == ProblemStatus.TIMED_OUT


class TestWaitForCompletionDrain:
    """wait_for_completion() must give an in-flight score a bounded chance
    to actually land -- otherwise mark_remaining_timed_out() correctly
    refusing to write a bogus TIMED_OUT (see TestEnvelopeSweep above) just
    means the problem is silently missing from the aggregate instead of
    wrongly zeroed, with no improvement to the miner's actual score. See
    review discussion on PR #295.
    """

    def test_drain_waits_for_in_flight_score_before_finalizing(self, reporter):
        import threading

        real_result = ProblemResult(
            problem_id=_P1, category="product", status=ProblemStatus.SUCCESS, score=1.0
        )
        release = threading.Event()

        def slow_worker():
            release.wait(timeout=5)
            with reporter._lock:
                reporter._results[_P1] = real_result

        future = reporter._scoring_pool._executor.submit(slow_worker)
        reporter._scoring_pool.futures[_P1] = future

        # p2/p3 already resolved normally before wait_for_completion runs.
        reporter._results[_P2] = ProblemResult(
            problem_id=_P2, category="product", status=ProblemStatus.SUCCESS, score=1.0
        )
        reporter._results[_P3] = ProblemResult(
            problem_id=_P3, category="product", status=ProblemStatus.SUCCESS, score=1.0
        )

        # Release the "worker" well inside the drain window, simulating a
        # score that lands a moment after the monitoring loop already broke.
        threading.Timer(0.2, release.set).start()

        reporter.DRAIN_TIMEOUT = 5.0
        reporter.DRAIN_POLL_INTERVAL = 0.05
        reporter.wait_for_completion()

        assert reporter._results[_P1] is real_result
        assert reporter._results[_P1].status == ProblemStatus.SUCCESS
        assert reporter._results[_P1].score == 1.0
        reporter.backend_client.report_progress.assert_called()

    def test_drain_gives_up_after_timeout_and_marks_timed_out(self, reporter):
        import threading
        from concurrent.futures import Future

        # A future that never completes within the (shortened) drain window.
        never_done: Future = Future()
        reporter._scoring_pool.futures[_P1] = never_done

        reporter._results[_P2] = ProblemResult(
            problem_id=_P2, category="product", status=ProblemStatus.SUCCESS, score=1.0
        )
        reporter._results[_P3] = ProblemResult(
            problem_id=_P3, category="product", status=ProblemStatus.SUCCESS, score=1.0
        )

        reporter.DRAIN_TIMEOUT = 0.2
        reporter.DRAIN_POLL_INTERVAL = 0.05
        reporter.wait_for_completion()

        assert reporter._results[_P1].status == ProblemStatus.TIMED_OUT
        reporter.backend_client.report_progress.assert_called()
