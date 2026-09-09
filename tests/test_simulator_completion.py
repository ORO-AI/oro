from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from validator.simulator_completion import SimulatorCompletion


def test_forwards_user_simulator_request_through_inference_proxy() -> None:
    client = MagicMock()
    client.post.return_value = {
        "choices": [
            {
                "message": {"content": '{"action":"no_op","content":""}'},
                "finish_reason": "stop",
            }
        ]
    }
    completion = SimulatorCompletion("miner-token", client=client)
    messages = [{"role": "user", "content": "continue?"}]

    result = asyncio.run(
        completion(
            "mistralai/mistral-small-2603",
            messages,
            max_tokens=400,
            temperature=0.0,
        )
    )

    client.post.assert_called_once_with(
        "/inference/chat/completions",
        json_data={
            "model": "mistralai/mistral-small-2603",
            "messages": messages,
            "max_tokens": 400,
            "temperature": 0.0,
            "stream": False,
        },
    )
    assert result == {
        "text": '{"action":"no_op","content":""}',
        "tool_calls": [],
        "finish_reason": "stop",
    }


@pytest.mark.parametrize("response", [None, {}, {"choices": []}])
def test_rejects_missing_completion(response) -> None:  # noqa: ANN001
    client = MagicMock()
    client.post.return_value = response
    completion = SimulatorCompletion("miner-token", client=client)

    with pytest.raises(RuntimeError, match="returned no completion"):
        asyncio.run(completion("model", []))
