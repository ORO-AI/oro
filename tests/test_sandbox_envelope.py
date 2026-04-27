"""Envelope format tests for ORO-907 sandbox->validator IPC."""

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
