"""Tests for process-based timeout enforcement in sandbox_executor."""

import json
import os
import tempfile
import time
from pathlib import Path

from src.agent.sandbox_executor import execute_single_problem, _read_inference_stats

FIXTURES = Path(__file__).parent / "fixtures"
FAST = str(FIXTURES / "fast_agent.py")
SLOW = str(FIXTURES / "slow_agent.py")
CRASH = str(FIXTURES / "crashing_agent.py")


def test_successful_execution():
    result = execute_single_problem(
        {"query": "laptop", "id": "p1"}, timeout=10.0, agent_file=FAST
    )
    assert result.success
    assert result.result["answer"] == "hello"
    assert result.problem_id == "p1"


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
    assert "agent crashed on purpose" in result.error


def test_missing_agent_file():
    result = execute_single_problem(
        {"query": "x"}, timeout=10.0, agent_file="/tmp/no_such_agent.py"
    )
    assert not result.success


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
