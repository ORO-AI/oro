"""Validator parses envelope format from ORO-907."""

import json
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from oro_sdk.models import ProblemStatus

from validator.progress_reporter import ProgressReporter


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
    rep._scorers = {}
    return rep


class TestEnvelopeParsing:
    def test_success_envelope_dispatches_to_scoring(self, reporter, tmp_path):
        _write_envelope(tmp_path / "output.jsonl", problem_id=_P1, status="SUCCESS")
        reporter._read_and_dispatch()
        assert _P1 in reporter._scoring_futures

    def test_failure_envelope_records_without_scoring(self, reporter, tmp_path):
        _write_envelope(
            tmp_path / "output.jsonl",
            problem_id=_P1,
            status="FAILED",
            dialogue=None,
            error={"type": "RuntimeError", "message": "boom"},
        )
        reporter._read_and_dispatch()
        assert _P1 not in reporter._scoring_futures
        assert reporter._results[_P1].status == ProblemStatus.FAILED

    def test_timeout_envelope_records_without_scoring(self, reporter, tmp_path):
        _write_envelope(
            tmp_path / "output.jsonl",
            problem_id=_P1,
            status="TIMED_OUT",
            dialogue=None,
            error={"type": "TimeoutError", "message": "..."},
        )
        reporter._read_and_dispatch()
        assert _P1 not in reporter._scoring_futures
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
        reporter._read_and_dispatch()
        # Inference counts captured from envelope at dispatch time.
        meta = reporter._envelope_meta[_P1]
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
        reporter._read_and_dispatch()
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
        reporter._read_and_dispatch()
        assert reporter._results[_P1].execution_time == 42.0

    def test_malformed_line_skipped(self, reporter, tmp_path):
        out = tmp_path / "output.jsonl"
        with open(out, "a") as f:
            f.write("not json\n")
        _write_envelope(out, problem_id=_P1, status="FAILED", dialogue=None,
                        error={"type": "RuntimeError", "message": "x"})
        reporter._read_and_dispatch()
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
        reporter._read_and_dispatch()
        reporter._mark_remaining_timed_out()
        assert reporter._results[_P1].status == ProblemStatus.FAILED

    def test_sweep_marks_only_never_seen_problems(self, reporter):
        # No envelope written. Sweep should mark all three TIMED_OUT.
        reporter._mark_remaining_timed_out()
        assert reporter._results[_P1].status == ProblemStatus.TIMED_OUT
        assert reporter._results[_P2].status == ProblemStatus.TIMED_OUT
        assert reporter._results[_P3].status == ProblemStatus.TIMED_OUT
