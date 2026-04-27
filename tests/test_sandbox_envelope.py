"""Envelope format tests for ORO-907 sandbox->validator IPC."""

import json
from unittest.mock import MagicMock, patch

from src.agent import sandbox_executor
from src.agent.sandbox_executor import ErrorInfo, ExecutionResult, _format_single_result
from src.agent.sandbox_status import SandboxProblemStatus


# Mirrors Backend/app/models/schemas/common.py::ProblemStatus.
# Sandbox cannot import from Backend repo. CI guard catches drift.
BACKEND_PROBLEM_STATUS_VALUES = frozenset(
    {"PENDING", "RUNNING", "SUCCESS", "FAILED", "SKIPPED", "TIMED_OUT"}
)


class TestSandboxProblemStatus:
    def test_values_subset_of_backend(self):
        sandbox_values = {s.value for s in SandboxProblemStatus}
        assert sandbox_values <= BACKEND_PROBLEM_STATUS_VALUES, (
            f"SandboxProblemStatus has values not in Backend ProblemStatus: "
            f"{sandbox_values - BACKEND_PROBLEM_STATUS_VALUES}. "
            f"Update Backend/app/models/schemas/common.py::ProblemStatus or "
            f"narrow sandbox emissions."
        )

    def test_required_values_present(self):
        values = {s.value for s in SandboxProblemStatus}
        assert {"SUCCESS", "FAILED", "TIMED_OUT"} <= values


class TestExecutionResultStatus:
    def test_default_status_is_failed(self):
        # Constructed without status -> FAILED (safe default).
        r = ExecutionResult(query="q", success=False, error="x")
        assert r.status == SandboxProblemStatus.FAILED

    def test_success_status_when_explicitly_set(self):
        r = ExecutionResult(
            query="q", success=True, result=[], status=SandboxProblemStatus.SUCCESS
        )
        assert r.status == SandboxProblemStatus.SUCCESS

    def test_timed_out_status_when_explicitly_set(self):
        r = ExecutionResult(
            query="q",
            success=False,
            error="t",
            status=SandboxProblemStatus.TIMED_OUT,
        )
        assert r.status == SandboxProblemStatus.TIMED_OUT


class TestExecuteSingleProblemStatus:
    def test_timed_out_when_process_alive_after_join(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SANDBOX_OUTPUT_FILE", str(tmp_path / "output.jsonl"))

        # Stub Process to simulate timeout: is_alive() True after join.
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True
        mock_proc.pid = 1
        mock_proc.exitcode = None
        with patch.object(
            sandbox_executor.multiprocessing, "Process", return_value=mock_proc
        ):
            result = sandbox_executor.execute_single_problem(
                {"query": "q", "problem_id": "p1"}, timeout=0.01
            )
        assert result.status == SandboxProblemStatus.TIMED_OUT
        assert result.success is False

    def test_failed_status_when_no_result_in_queue(self, tmp_path, monkeypatch):
        """Process exits cleanly but queue is empty -> FAILED."""
        monkeypatch.setenv("SANDBOX_OUTPUT_FILE", str(tmp_path / "output.jsonl"))

        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = False
        mock_proc.pid = 2
        mock_proc.exitcode = 1
        with patch.object(
            sandbox_executor.multiprocessing, "Process", return_value=mock_proc
        ):
            result = sandbox_executor.execute_single_problem(
                {"query": "q", "problem_id": "p2"}, timeout=0.01
            )
        assert result.status == SandboxProblemStatus.FAILED
        assert result.success is False


def _parse(line: str) -> dict:
    obj = json.loads(line)
    assert isinstance(obj, dict), f"Envelope must be a dict, got {type(obj)}"
    return obj


class TestFormatSingleResultEnvelope:
    def test_success_envelope(self):
        result = ExecutionResult(
            query="q",
            success=True,
            result=[{"role": "user", "content": "hi", "extra_info": {"step": 1}}],
            execution_time=2.5,
            problem_id="p1",
            inference_failure_count=0,
            inference_total=4,
            status=SandboxProblemStatus.SUCCESS,
        )
        env = _parse(_format_single_result(result))
        assert env["problem_id"] == "p1"
        assert env["status"] == "SUCCESS"
        assert env["execution_time"] == 2.5
        assert env["inference_failure_count"] == 0
        assert env["inference_total"] == 4
        assert env["error"] is None
        assert isinstance(env["dialogue"], list) and len(env["dialogue"]) == 1

    def test_failure_envelope_emits_with_null_dialogue(self):
        result = ExecutionResult(
            query="q",
            success=False,
            error="boom",
            execution_time=0.5,
            problem_id="p2",
            inference_failure_count=1,
            inference_total=2,
            status=SandboxProblemStatus.FAILED,
        )
        env = _parse(_format_single_result(result))
        assert env["problem_id"] == "p2"
        assert env["status"] == "FAILED"
        assert env["dialogue"] is None
        assert env["error"]["message"] == "boom"
        assert env["error"]["type"]  # non-empty

    def test_failure_envelope_uses_structured_error_type_from_source(self):
        """When error is captured at the exception site as ErrorInfo, the
        type is the actual exception class name, not a regex on the message."""
        result = ExecutionResult(
            query="q",
            success=False,
            error=ErrorInfo(type="ValueError", message="random text without colon"),
            execution_time=0.5,
            problem_id="p_struct",
            status=SandboxProblemStatus.FAILED,
        )
        env = _parse(_format_single_result(result))
        assert env["error"]["type"] == "ValueError"
        assert env["error"]["message"] == "random text without colon"

    def test_run_in_process_captures_exception_type_at_source(self, tmp_path):
        """Child process side: exception type name is on the queue as a dict."""
        import multiprocessing as mp

        agent_file = tmp_path / "raising_agent.py"
        agent_file.write_text(
            "def agent_main(problem):\n"
            "    raise KeyError('missing-key')\n"
        )
        q: mp.Queue = mp.Queue()
        sandbox_executor._run_in_process(
            {"query": "q", "problem_id": "x"}, str(agent_file), q
        )
        kind, data = q.get(timeout=2)
        assert kind == "error"
        assert isinstance(data, dict)
        assert data["type"] == "KeyError"
        assert "missing-key" in data["message"]

    def test_timeout_envelope_marks_status_and_records_partial_counts(self):
        result = ExecutionResult(
            query="q",
            success=False,
            error="Execution exceeded timeout of 300s",
            execution_time=300.0,
            problem_id="p3",
            inference_failure_count=0,
            inference_total=3,
            status=SandboxProblemStatus.TIMED_OUT,
        )
        env = _parse(_format_single_result(result))
        assert env["status"] == "TIMED_OUT"
        assert env["dialogue"] is None
        assert env["inference_total"] == 3
        assert env["error"]["type"] == "TimeoutError"
