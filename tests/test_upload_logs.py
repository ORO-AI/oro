"""Regression tests for `_split_output_by_problem` (ORO-988).

v1.0.60 was uploading exactly one trajectory per run because the validator's
upload-loop parser still expected the pre-ORO-907 line format (a JSON list of
steps) while the sandbox had switched to the post-ORO-907 envelope format
({"problem_id": ..., "dialogue": [steps]}). The parser silently fell through
and dumped the whole file under problem_ids[0].

Tests exercise the parsing helper directly, decoupled from `OroValidator`
construction (which pulls in bittensor + the full SDK chain).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from uuid import UUID, uuid4


def _import_split() -> callable:
    """Import _split_output_by_problem without dragging in bittensor.

    The validator module's top-level imports are heavy. For this test we
    monkey-load just the helper by stubbing the heavy deps that fail to
    initialize in the test venv.
    """
    # Stub modules that the validator imports but we don't need here.
    for mod_name in ("bittensor", "bittensor.core.config", "bittensor.core.subtensor"):
        sys.modules.setdefault(mod_name, types.ModuleType(mod_name))

    # Read the function source directly and exec into a tiny namespace —
    # avoids a full module import and any of its side effects.
    src_path = Path(__file__).resolve().parent.parent / "subnet" / "validator" / "main.py"
    text = src_path.read_text()

    start = text.index("def _split_output_by_problem(")
    # Pull from the def through the closing of the function — find the next
    # top-level `def ` or `class ` after it.
    rest = text[start:]
    next_top = min(
        (rest.find(f"\n{kw} ", 1) for kw in ("def", "class") if rest.find(f"\n{kw} ", 1) != -1),
        default=len(rest),
    )
    body = rest[:next_top]

    ns = {
        "json": __import__("json"),
        "logging": __import__("logging"),
        "Path": Path,
        "UUID": UUID,
        "Optional": __import__("typing").Optional,
    }
    exec(compile(body, str(src_path), "exec"), ns)
    return ns["_split_output_by_problem"]


_split_output_by_problem = _import_split()


def _problem_envelope(pid: str, status: str = "SUCCESS") -> dict:
    """Mirrors src/agent/sandbox_executor.py::_format_single_result envelope."""
    return {
        "problem_id": pid,
        "status": status,
        "execution_time": 1.23,
        "inference_failure_count": 0,
        "inference_total": 1,
        "error": None,
        "dialogue": [
            {
                "role": "assistant",
                "content": f"step for {pid}",
                "extra_info": {"problem_id": pid, "step_index": 0},
            },
            {
                "role": "tool",
                "content": "result",
                "extra_info": {"problem_id": pid, "step_index": 1},
            },
        ],
    }


class TestSplitOutputEnvelopeParsing:
    """Each line in output.jsonl is one envelope; one entry per problem."""

    def test_three_envelopes_yield_three_entries(self, tmp_path: Path) -> None:
        # Use UUIDs that string-round-trip cleanly.
        pids = [str(uuid4()) for _ in range(3)]
        output_file = tmp_path / "output.jsonl"
        output_file.write_text(
            "\n".join(json.dumps(_problem_envelope(p)) for p in pids) + "\n"
        )

        result = _split_output_by_problem(output_file, [UUID(p) for p in pids])

        assert set(result.keys()) == set(pids), (
            f"Expected one entry per problem; got {set(result.keys())} vs "
            f"{set(pids)}"
        )

        # Each value is the dialogue array (Trajectory shape), not the
        # envelope.
        for pid, payload in result.items():
            decoded = json.loads(payload)
            assert isinstance(decoded, list), (
                f"Trajectory payload should be a list (Frontend Trajectory ="
                f" TrajectoryStep[]); got {type(decoded).__name__}"
            )
            assert decoded
            assert decoded[0]["extra_info"]["problem_id"] == pid

    def test_mixed_terminal_statuses_all_split(self, tmp_path: Path) -> None:
        # 1 SUCCESS + 1 FAILED + 1 TIMED_OUT — all should produce a trajectory.
        envelopes = [
            _problem_envelope(str(uuid4()), "SUCCESS"),
            _problem_envelope(str(uuid4()), "FAILED"),
            _problem_envelope(str(uuid4()), "TIMED_OUT"),
        ]
        output_file = tmp_path / "output.jsonl"
        output_file.write_text("\n".join(json.dumps(e) for e in envelopes) + "\n")

        result = _split_output_by_problem(
            output_file, [UUID(e["problem_id"]) for e in envelopes]
        )

        assert len(result) == 3

    def test_unparseable_lines_skipped_without_dropping_others(
        self, tmp_path: Path
    ) -> None:
        good = _problem_envelope(str(uuid4()))
        output_file = tmp_path / "output.jsonl"
        output_file.write_text(
            "\n".join([json.dumps(good), "not json", "", "{not json either"]) + "\n"
        )

        result = _split_output_by_problem(
            output_file, [UUID(good["problem_id"])]
        )

        assert set(result.keys()) == {good["problem_id"]}

    def test_empty_dialogue_yields_empty_list_payload(
        self, tmp_path: Path
    ) -> None:
        # Sandbox crash before agent emitted steps still produces an envelope
        # (dialogue=null/[]). Treat as a real outcome — empty trajectory, not
        # a missing entry.
        envelope = {
            "problem_id": str(uuid4()),
            "status": "FAILED",
            "execution_time": 0.5,
            "inference_failure_count": 0,
            "inference_total": 0,
            "error": {"type": "AgentImportError", "message": "boom"},
            "dialogue": None,
        }
        output_file = tmp_path / "output.jsonl"
        output_file.write_text(json.dumps(envelope) + "\n")

        result = _split_output_by_problem(
            output_file, [UUID(envelope["problem_id"])]
        )

        assert envelope["problem_id"] in result
        assert json.loads(result[envelope["problem_id"]]) == []

    def test_empty_file_falls_back_to_first_problem_id(
        self, tmp_path: Path
    ) -> None:
        # Corrupt / empty output.jsonl: keep the legacy fallback so the run
        # still has *some* artifact attached for debugging.
        output_file = tmp_path / "output.jsonl"
        output_file.write_text("")

        first = uuid4()
        result = _split_output_by_problem(output_file, [first, uuid4()])

        # Empty file → empty parse → fallback writes empty bytes under first.
        assert str(first) in result

    def test_no_problem_ids_returns_empty(self, tmp_path: Path) -> None:
        output_file = tmp_path / "output.jsonl"
        output_file.write_text("garbage\n")

        result = _split_output_by_problem(output_file, [])

        assert result == {}
