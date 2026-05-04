"""Unit tests for _format_single_result envelope output (ORO-907)."""

import json
from typing import List, Optional

from src.agent.sandbox_executor import ExecutionResult, _format_single_result
from src.agent.sandbox_status import SandboxProblemStatus


def _make_result(
    success: bool,
    dialogue: Optional[List],
    execution_time: float,
    status: SandboxProblemStatus = SandboxProblemStatus.SUCCESS,
    error: Optional[str] = None,
) -> ExecutionResult:
    return ExecutionResult(
        query="find a laptop",
        success=success,
        result=dialogue,
        error=error,
        execution_time=execution_time,
        problem_id="11111111-1111-1111-1111-111111111111",
        status=status,
    )


class TestFormatSingleResultExecutionTime:
    def test_execution_time_stamped_on_first_dialogue_step(self):
        dialogue = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]
        line = _format_single_result(
            _make_result(success=True, dialogue=dialogue, execution_time=4.25)
        )
        envelope = json.loads(line)
        steps = envelope["dialogue"]
        assert steps[0]["extra_info"]["execution_time"] == 4.25

    def test_execution_time_not_duplicated_on_later_steps(self):
        dialogue = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]
        line = _format_single_result(
            _make_result(success=True, dialogue=dialogue, execution_time=4.25)
        )
        envelope = json.loads(line)
        steps = envelope["dialogue"]
        assert "execution_time" not in steps[1].get("extra_info", {})

    def test_failure_emits_envelope_with_null_dialogue(self):
        line = _format_single_result(
            _make_result(
                success=False,
                dialogue=None,
                execution_time=1.0,
                status=SandboxProblemStatus.FAILED,
                error="boom",
            )
        )
        envelope = json.loads(line)
        assert isinstance(envelope, dict)
        assert envelope["status"] == "FAILED"
        assert envelope["dialogue"] is None
