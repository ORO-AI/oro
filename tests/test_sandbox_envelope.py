"""Envelope format tests for ORO-907 sandbox->validator IPC."""

from unittest.mock import MagicMock, patch

from src.agent import sandbox_executor
from src.agent.sandbox_executor import ExecutionResult
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
