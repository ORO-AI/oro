"""Regression tests for split_output_by_problem (ORO-988).

v1.0.60 uploaded one trajectory per run instead of ~30 because the upload
parser still expected the pre-ORO-907 list-of-steps shape while the sandbox
had switched to envelope dicts. The fallback dumped everything under
problem_ids[0]. Tests pin the post-907 envelope path.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID, uuid4

from subnet.validator.output_split import split_output_by_problem


def _envelope(pid: str) -> dict:
    """Mirror src/agent/sandbox_executor.py::_format_single_result envelope."""
    return {
        "problem_id": pid,
        "status": "SUCCESS",
        "execution_time": 1.23,
        "inference_failure_count": 0,
        "inference_total": 1,
        "error": None,
        "dialogue": [
            {"role": "assistant", "content": f"step for {pid}"},
            {"role": "tool", "content": "result"},
        ],
    }


def _write_jsonl(tmp_path: Path, lines: list[str]) -> Path:
    f = tmp_path / "output.jsonl"
    f.write_text("\n".join(lines) + "\n")
    return f


def test_three_envelopes_yield_three_entries(tmp_path: Path) -> None:
    """The regression case: 3 problems → 3 entries, each keyed by its own id,
    payload = the dialogue array (Frontend Trajectory shape)."""
    pids = [str(uuid4()) for _ in range(3)]
    f = _write_jsonl(tmp_path, [json.dumps(_envelope(p)) for p in pids])

    result = split_output_by_problem(f, [UUID(p) for p in pids])

    assert set(result) == set(pids)
    for pid, payload in result.items():
        decoded = json.loads(payload)
        assert isinstance(decoded, list) and decoded
        assert decoded[0]["content"] == f"step for {pid}"


def test_unparseable_lines_skipped(tmp_path: Path) -> None:
    good = _envelope(str(uuid4()))
    f = _write_jsonl(tmp_path, [json.dumps(good), "not json", "", "{not json"])

    result = split_output_by_problem(f, [UUID(good["problem_id"])])

    assert set(result) == {good["problem_id"]}


def test_empty_dialogue_on_success_yields_empty_list_payload(tmp_path: Path) -> None:
    """SUCCESS status with no dialogue is a real edge case (shouldn't happen
    in practice) — preserve the empty-array shape so consumers don't trip on
    a synthetic step they don't expect."""
    pid = str(uuid4())
    envelope = {**_envelope(pid), "dialogue": None}
    f = _write_jsonl(tmp_path, [json.dumps(envelope)])

    result = split_output_by_problem(f, [UUID(pid)])

    assert json.loads(result[pid]) == []


def test_abnormal_termination_synthesizes_step_with_error_metadata(tmp_path: Path) -> None:
    """ORO-1147: FAILED / TIMED_OUT envelopes have dialogue=None plus an error
    object. Previously the splitter wrote `[]` and lost the failure context.
    Now it synthesizes a single step carrying status + error so the artifact
    array shape stays valid and downstream consumers can distinguish abnormal
    termination from a zero-step trajectory."""
    pid = str(uuid4())
    envelope = {
        **_envelope(pid),
        "status": "TIMED_OUT",
        "dialogue": None,
        "error": {"type": "TimeoutError", "message": "Sandbox timeout after 300.0s"},
        "execution_time": 300.0,
    }
    f = _write_jsonl(tmp_path, [json.dumps(envelope)])

    payload = json.loads(split_output_by_problem(f, [UUID(pid)])[pid])

    assert isinstance(payload, list) and len(payload) == 1
    step = payload[0]
    assert step["role"] == "system"
    assert "TIMED_OUT" in step["content"]
    assert "TimeoutError" in step["content"]
    info = step["extra_info"]
    assert info["abnormal_termination"] is True
    assert info["status"] == "TIMED_OUT"
    assert info["error"]["type"] == "TimeoutError"
    assert info["problem_id"] == pid
    assert info["execution_time"] == 300.0


def test_abnormal_termination_with_missing_error_object(tmp_path: Path) -> None:
    """FAILED envelopes occasionally arrive with error=None (process exited
    with code N but produced no result). Still synthesize a step — don't
    crash on the missing error dict."""
    pid = str(uuid4())
    envelope = {**_envelope(pid), "status": "FAILED", "dialogue": None, "error": None}
    f = _write_jsonl(tmp_path, [json.dumps(envelope)])

    payload = json.loads(split_output_by_problem(f, [UUID(pid)])[pid])

    assert len(payload) == 1
    assert payload[0]["extra_info"]["abnormal_termination"] is True
    assert payload[0]["extra_info"]["error"]["type"] == "UnknownError"


def test_empty_file_falls_back_to_first_problem_id(tmp_path: Path) -> None:
    """Corrupt / empty output → still attach *something* under problem_ids[0]
    so the run has a forensic artifact."""
    f = tmp_path / "output.jsonl"
    f.write_text("")

    first = uuid4()
    result = split_output_by_problem(f, [first, uuid4()])

    assert str(first) in result
