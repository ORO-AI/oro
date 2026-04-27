"""Sandbox-side problem outcome status enum.

Emitted in the per-problem envelope written to output.jsonl. Validator's
ProgressReporter reads these values and forwards to Backend's ProblemStatus.

Values must be a subset of Backend/app/models/schemas/common.py::ProblemStatus.
The CI guard test in tests/test_sandbox_envelope.py asserts this subset
relation against a literal mirror of the Backend enum values.
"""

from enum import Enum


class SandboxProblemStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
