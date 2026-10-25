"""Integration coverage for the world-session example agent."""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.agent import environment_agent
from subnet.tests.world_pack_fixture import StubModelClient, StubSearchClient, load_world_pack
from validator.session_registry import SessionRegistry
from validator.session_service import SessionRuntime


def _tool_call(n: int, name: str, args: dict) -> dict:
    return {"id": f"model-call-{n}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_example_agent_takes_one_step_per_tool_call_until_finished(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    pack = load_world_pack(tmp_path)
    runtime = SessionRuntime()
    replies = [
        {"content": "Thinking."},  # no tool call: the agent nudges and retries
        {"content": "", "tool_calls": [_tool_call(1, "find", {"query": "red"}), _tool_call(2, "ask", {"text": "this one?"})]},
        {"content": "", "tool_calls": [_tool_call(3, "buy", {"id": "a"}), _tool_call(4, "find", {"query": "x"})]},
    ]
    inference: list[dict[str, Any]] = []
    environment: list[dict[str, Any]] = []

    def post(path: str, json_data: dict[str, Any]) -> dict[str, Any]:
        if path == "/inference/chat/completions":
            inference.append(json_data)
            return {"choices": [{"message": replies[len(inference) - 1]}]}
        environment.append(json_data)
        return runtime.call(json_data)

    monkeypatch.setattr(environment_agent._proxy, "post", post)
    with SessionRegistry(pack, episode_nonces={"t1": "n1"}, model_client=StubModelClient("Yes, red."),
                         search_client=StubSearchClient()) as registry:
        runtime.install(registry)
        started = registry.start("t1")
        problem = {"problem_id": "p1", "environment": {"binding": started["binding"], "policy_view": started["policy_view"]}}

        dialogue = environment_agent.agent_main(problem)
        [record] = registry.terminal_results()

    assert inference[0]["tools"] == started["policy_view"]["tools"]
    assert "hello" in inference[0]["messages"][1]["content"]  # the opening reaches the model
    assert [(e["step"], e["action"]["name"]) for e in environment] == [(1, "find"), (2, "ask"), (3, "buy")]
    assert all(e["session_id"] == started["session_id"] for e in environment)
    tool_messages = [m for m in inference[2]["messages"] if m["role"] == "tool"]
    assert json.loads(tool_messages[1]["content"]) == {"reply": "Yes, red."}  # the shopper reply is an observation
    assert dialogue[-1]["environment_result"]["finished"] is True
    assert record["verdict"]["score"] == 1.0
