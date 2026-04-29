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


def test_empty_dialogue_yields_empty_list_payload(tmp_path: Path) -> None:
    """Sandbox crash before any agent step still produces a per-problem entry
    with payload `[]` (Frontend renders 'no trajectory' for that problem)."""
    pid = str(uuid4())
    envelope = {**_envelope(pid), "dialogue": None}
    f = _write_jsonl(tmp_path, [json.dumps(envelope)])

    result = split_output_by_problem(f, [UUID(pid)])

    assert json.loads(result[pid]) == []


def test_empty_file_falls_back_to_first_problem_id(tmp_path: Path) -> None:
    """Corrupt / empty output → still attach *something* under problem_ids[0]
    so the run has a forensic artifact."""
    f = tmp_path / "output.jsonl"
    f.write_text("")

    first = uuid4()
    result = split_output_by_problem(f, [first, uuid4()])

    assert str(first) in result
