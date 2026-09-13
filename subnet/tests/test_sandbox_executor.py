"""Tests for process-based timeout enforcement in sandbox_executor."""

import json
import math
import os
import tempfile
import time
from pathlib import Path

from src.agent.sandbox_executor import (
    _read_inference_stats,
    _read_request_log,
    build_result_envelope,
    execute_single_problem,
    read_inference_stats,
)

FIXTURES = Path(__file__).parent / "fixtures"
FAST = str(FIXTURES / "fast_agent.py")
SLOW = str(FIXTURES / "slow_agent.py")
CRASH = str(FIXTURES / "crashing_agent.py")
FROZEN_DC = str(FIXTURES / "frozen_dataclass_agent.py")
LARGE_RESULT = str(FIXTURES / "large_result_agent.py")
INFERENCE_STATS = str(FIXTURES / "inference_stats_agent.py")


def test_successful_execution():
    result = execute_single_problem(
        {"query": "laptop", "id": "p1"}, timeout=10.0, agent_file=FAST
    )
    assert result.success
    assert result.result["answer"] == "hello"
    assert result.problem_id == "p1"


def test_large_result_is_received_before_child_exit():
    """A result larger than the queue pipe must not make a finished agent time out."""
    result = execute_single_problem(
        {"query": "large result", "id": "p-large"},
        timeout=2.0,
        agent_file=LARGE_RESULT,
    )

    assert result.success
    assert result.result == {"answer": "x" * 2_000_000}
    assert result.status.value == "SUCCESS"


def test_timeout_kills_process():
    """Core test: slow agent (sleeps 9999s) is terminated after 2s timeout."""
    start = time.time()
    result = execute_single_problem({"query": "slow"}, timeout=2.0, agent_file=SLOW)
    assert not result.success
    assert "timeout" in result.error.lower()
    assert time.time() - start < 15.0


def test_agent_crash_returns_error():
    result = execute_single_problem({"query": "crash"}, timeout=10.0, agent_file=CRASH)
    assert not result.success
    err_msg = result.error.message if hasattr(result.error, "message") else result.error
    assert "agent crashed on purpose" in err_msg


def test_missing_agent_file():
    result = execute_single_problem(
        {"query": "x"}, timeout=10.0, agent_file="/tmp/no_such_agent.py"
    )
    assert not result.success


def test_frozen_dataclass_with_pep563_annotations():
    """Agents using `from __future__ import annotations` + @dataclass rely on
    the loaded module being present in sys.modules — dataclasses._is_type
    reads sys.modules[cls.__module__] to resolve string-form annotations."""
    result = execute_single_problem(
        {"query": "anything", "id": "p-frozen"}, timeout=10.0, agent_file=FROZEN_DC
    )
    assert result.success
    assert result.result["answer"] == "hello-frozen"


def test_agent_process_snapshots_episode_inference_stats(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_OUTPUT_FILE", str(tmp_path / "output.jsonl"))
    result = execute_single_problem(
        {
            "query": "test",
            "problem_id": "episode-1",
            "category": "generated_environment",
        },
        timeout=10.0,
        agent_file=INFERENCE_STATS,
    )

    assert result.success
    usage = read_inference_stats(str(tmp_path / "inference_stats.jsonl"))
    assert usage["episode-1"] == {
        "problem_id": "episode-1",
        "inference_success": 1,
        "inference_failed": 0,
        "inference_total": 1,
        "inference_cost_usd": 0.25,
        "inference_cost_missing": 0,
        "prompt_tokens": 10,
        "completion_tokens": 2,
    }
    assert result.inference_usage == usage["episode-1"]
    output = tmp_path / "runner-output.jsonl"
    envelope = build_result_envelope(result)
    assert envelope["_shadow_inference_usage"] == usage["episode-1"]
    envelope["_shadow_inference_usage"] = {
        **envelope["_shadow_inference_usage"],
        "problem_id": "spoofed",
    }
    output.write_text(json.dumps(envelope) + "\n")
    assert read_inference_stats(str(output)) == {"episode-1": usage["episode-1"]}


def test_legacy_result_envelope_does_not_include_shadow_inference_usage():
    result = execute_single_problem(
        {"query": "test", "problem_id": "legacy"},
        timeout=10.0,
        agent_file=FAST,
    )

    assert "_shadow_inference_usage" not in build_result_envelope(result)


class TestReadInferenceStats:
    def test_reads_matching_problem(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            f.write(json.dumps({"problem_id": "p1", "inference_failed": 2, "inference_total": 8}) + "\n")
            f.write(json.dumps({"problem_id": "p2", "inference_failed": 0, "inference_total": 5}) + "\n")
            path = f.name
        try:
            failures, total = _read_inference_stats(path, "p1")
            assert failures == 2
            assert total == 8
        finally:
            os.unlink(path)

    def test_missing_problem_returns_zeros(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            f.write(json.dumps({"problem_id": "p1", "inference_failed": 1, "inference_total": 3}) + "\n")
            path = f.name
        try:
            failures, total = _read_inference_stats(path, "p99")
            assert failures == 0
            assert total == 0
        finally:
            os.unlink(path)

    def test_missing_file_returns_zeros(self):
        failures, total = _read_inference_stats("/tmp/nonexistent.jsonl", "p1")
        assert failures == 0
        assert total == 0

    def test_merges_latest_cumulative_stats_from_each_source(self):
        agent_entries = [
            {
                "problem_id": "p1",
                "inference_total": 1,
                "inference_cost_usd": 0.1,
            },
            {
                "problem_id": "p1",
                "inference_total": 3,
                "inference_cost_usd": 0.3,
            },
        ]
        simulator_entry = {
            "problem_id": "p1",
            "inference_total": 2,
            "inference_cost_usd": 0.2,
        }
        with (
            tempfile.NamedTemporaryFile(
                suffix=".jsonl", delete=False, mode="w"
            ) as agent_file,
            tempfile.NamedTemporaryFile(
                suffix=".jsonl", delete=False, mode="w"
            ) as simulator_file,
        ):
            for entry in agent_entries:
                agent_file.write(json.dumps(entry) + "\n")
            simulator_file.write(json.dumps(simulator_entry) + "\n")
            paths = [agent_file.name, simulator_file.name]
        try:
            usage = read_inference_stats(paths)["p1"]
            assert usage["inference_failed"] == 0
            assert usage["inference_total"] == 5
            assert usage["inference_cost_usd"] == 0.5
        finally:
            for path in paths:
                os.unlink(path)

    def test_malformed_line_does_not_hide_later_snapshot(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            f.write('{"problem_id":"p1"\n')
            f.write(json.dumps({"problem_id": "p1", "inference_total": 2}) + "\n")
            path = f.name
        try:
            assert read_inference_stats(path)["p1"]["inference_total"] == 2
        finally:
            os.unlink(path)

    def test_ignores_snapshot_with_no_valid_counters(self, tmp_path):
        for index, value in enumerate(("bad", math.nan, math.inf, -1)):
            path = tmp_path / f"stats-{index}.jsonl"
            path.write_text(
                json.dumps({"problem_id": "p1", "inference_total": value})
            )

            assert read_inference_stats(str(path)) == {}


class TestReadRequestLog:
    def test_reads_all_entries(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            f.write(json.dumps({"method": "GET", "path": "/a"}) + "\n")
            f.write(json.dumps({"method": "POST", "path": "/b"}) + "\n")
            path = f.name
        try:
            entries = _read_request_log(path)
            assert len(entries) == 2
            assert entries[0]["path"] == "/a"
            assert entries[1]["path"] == "/b"
        finally:
            os.unlink(path)

    def test_missing_file_returns_empty(self):
        entries = _read_request_log("/tmp/nonexistent_request_log.jsonl")
        assert entries == []

    def test_skips_blank_lines(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            f.write(json.dumps({"method": "GET", "path": "/a"}) + "\n")
            f.write("\n")
            f.write(json.dumps({"method": "GET", "path": "/b"}) + "\n")
            path = f.name
        try:
            entries = _read_request_log(path)
            assert len(entries) == 2
        finally:
            os.unlink(path)
