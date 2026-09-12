"""Integration coverage for the generated-environment example agent."""

from __future__ import annotations

import json as jsonlib
from pathlib import Path
from typing import Any

import pytest

from src.agent import environment_agent
from validator.env_pack_loader import LoadedPack
from validator.generated_evaluation import write_problem_file
from validator.session_registry import SessionRegistry
from validator.session_service import SessionRuntime

pytest_plugins = ("tests.compat_fixture",)


def test_example_agent_uses_dynamic_tools_to_complete_a_runtime_session(
    loaded_pack: LoadedPack,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = SessionRuntime()
    candidate = loaded_pack.task_specs[0].gold_set[0]
    planned_actions = [
        {
            "name": "add_to_cart",
            "arguments": {
                "product_id": candidate.product_id,
                "sku": candidate.sku,
            },
        },
        {"name": "place_test_order", "arguments": {}},
    ]
    inference_requests: list[dict[str, Any]] = []
    environment_requests: list[dict[str, Any]] = []

    def post(path: str, json_data: dict[str, Any]) -> dict[str, Any]:
        if path == "/inference/chat/completions":
            inference_requests.append(json_data)
            if len(inference_requests) == 1:
                return {"choices": [{"message": {"content": "Finished."}}]}
            action = planned_actions[len(inference_requests) - 2]
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": f"model-call-{len(inference_requests)}",
                                    "type": "function",
                                    "function": {
                                        "name": action["name"],
                                        "arguments": jsonlib.dumps(
                                            action["arguments"]
                                        ),
                                    }
                                }
                            ],
                        }
                    }
                ]
            }
        environment_requests.append(json_data)
        return runtime.call(json_data)

    monkeypatch.setattr(environment_agent._proxy, "post", post)

    with SessionRegistry(loaded_pack) as registry:
        runtime.install(registry)
        bootstrap = registry.start(
            evaluation_run_id="private-evaluation-id",
            agent_version_id="private-agent-version-id",
            task_id=loaded_pack.task_ids[0],
            session_id="example-session",
        )
        problem_path = tmp_path / "problems.jsonl"
        write_problem_file(problem_path, [bootstrap])
        problem = jsonlib.loads(problem_path.read_text(encoding="utf-8"))

        dialogue = environment_agent.agent_main(problem)

    assert len(dialogue) == 3
    assert len(inference_requests) == 3
    assert any(
        message.get("content")
        == "The environment has not reported done=true. "
        "Choose one of the supplied tools to continue."
        for message in inference_requests[1]["messages"]
    )
    assert inference_requests[0]["tools"] == problem["environment"]["policy_view"][
        "tools"
    ]
    assert [request["turn"] for request in environment_requests] == [1, 2]
    assert all(
        request["idempotency_key"] == request["call_id"]
        for request in environment_requests
    )
    assert dialogue[-1]["environment_result"]["calls"][0]["observation"][
        "done"
    ] is True
