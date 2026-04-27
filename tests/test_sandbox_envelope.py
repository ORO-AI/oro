"""Envelope format tests for ORO-907 sandbox->validator IPC."""

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
