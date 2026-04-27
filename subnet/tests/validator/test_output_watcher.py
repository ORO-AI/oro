"""OutputWatcher: tail envelope stream, yield ProblemRecord."""

import json
from pathlib import Path

import pytest

from src.agent.sandbox_status import SandboxProblemStatus
from validator.output_watcher import ErrorInfo, OutputWatcher, ProblemRecord


def _write(path: Path, **fields):
    env = {
        "problem_id": fields["problem_id"],
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
        f.write(json.dumps(env) + "\n")


@pytest.fixture
def output_path(tmp_path) -> Path:
    p = tmp_path / "output.jsonl"
    p.touch()
    return p


class TestOutputWatcher:
    def test_yields_records_in_append_order(self, output_path):
        w = OutputWatcher(output_path)
        _write(output_path, problem_id="p1", inference_total=4, execution_time=2.5)
        _write(
            output_path,
            problem_id="p2",
            status="FAILED",
            dialogue=None,
            error={"type": "RuntimeError", "message": "boom"},
        )

        records = list(w.read_new())

        assert [r.problem_id for r in records] == ["p1", "p2"]
        assert isinstance(records[0], ProblemRecord)
        assert records[0].status is SandboxProblemStatus.SUCCESS
        assert records[0].execution_time == 2.5
        assert records[0].inference_total == 4
        assert records[0].error is None
        assert records[1].status is SandboxProblemStatus.FAILED
        assert records[1].dialogue is None
        assert isinstance(records[1].error, ErrorInfo)
        assert records[1].error.type == "RuntimeError"
        assert records[1].error.message == "boom"

    def test_truncation_resets_position(self, output_path):
        w = OutputWatcher(output_path)
        # Two envelopes so the position advances well past the post-truncate
        # length of one envelope, so truncation is visibly detected.
        _write(output_path, problem_id="p1")
        _write(output_path, problem_id="p2")
        first = list(w.read_new())
        assert [r.problem_id for r in first] == ["p1", "p2"]

        # Truncate file (simulates sandbox restart writing fresh output).
        output_path.write_text("")
        _write(output_path, problem_id="p3")
        records = list(w.read_new())
        assert [r.problem_id for r in records] == ["p3"]

    def test_skips_malformed_lines(self, output_path):
        # Malformed first, valid second.
        with open(output_path, "w") as f:
            f.write("not json\n")
        _write(output_path, problem_id="p1")
        w = OutputWatcher(output_path)
        records = list(w.read_new())
        assert [r.problem_id for r in records] == ["p1"]

    def test_skips_unknown_status(self, output_path):
        # Unknown status — emit only the valid follow-up record.
        env = {
            "problem_id": "p1",
            "status": "WAT",
            "execution_time": 1.0,
            "inference_failure_count": 0,
            "inference_total": 1,
            "error": None,
            "dialogue": None,
        }
        with open(output_path, "a") as f:
            f.write(json.dumps(env) + "\n")
        _write(output_path, problem_id="p2", status="SUCCESS")

        w = OutputWatcher(output_path)
        records = list(w.read_new())
        assert [r.problem_id for r in records] == ["p2"]

    def test_returns_typed_status_enum(self, output_path):
        _write(output_path, problem_id="p1", status="SUCCESS")
        w = OutputWatcher(output_path)
        records = list(w.read_new())
        assert len(records) == 1
        # Status MUST be the typed enum so callers can use `is` comparisons.
        assert records[0].status is SandboxProblemStatus.SUCCESS
