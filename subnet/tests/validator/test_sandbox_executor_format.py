"""Unit tests for _format_single_result output JSONL shape."""

import json
import sys
from pathlib import Path
from typing import List, Optional

# Allow `src.agent` imports from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.agent.sandbox_executor import ExecutionResult, _format_single_result


def _make_result(success: bool, dialogue: Optional[List], execution_time: float) -> ExecutionResult:
    return ExecutionResult(
        query="find a laptop",
        success=success,
        result=dialogue,
        execution_time=execution_time,
        problem_id="11111111-1111-1111-1111-111111111111",
    )


class TestFormatSingleResultExecutionTime:
    def test_execution_time_stamped_on_first_step(self):
        dialogue = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
        line = _format_single_result(_make_result(success=True, dialogue=dialogue, execution_time=4.25))
        assert line is not None
        steps = json.loads(line)
        assert steps[0]["extra_info"]["execution_time"] == 4.25

    def test_execution_time_not_duplicated_on_later_steps(self):
        dialogue = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
        line = _format_single_result(_make_result(success=True, dialogue=dialogue, execution_time=4.25))
        steps = json.loads(line)
        assert "execution_time" not in steps[1].get("extra_info", {})

    def test_failure_still_returns_none(self):
        line = _format_single_result(_make_result(success=False, dialogue=None, execution_time=1.0))
        assert line is None
