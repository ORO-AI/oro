"""Minimal agent example for validator-owned generated environments."""

from __future__ import annotations

import json
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


_DEFAULT_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}
_proxy = ProxyClient(timeout=120, max_retries=2)


def _model() -> str:
    provider = getenv("INFERENCE_PROVIDER", "chutes")
    return getenv("SANDBOX_MODEL") or _DEFAULT_MODELS.get(
        provider, _DEFAULT_MODELS["chutes"]
    )


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Let the model choose actions from the environment's dynamic tool list."""

    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = int(policy.get("max_calls_per_turn", 1))

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "Use the supplied shopping tools to satisfy the shopper. "
                "Continue until an observation reports done=true."
            ),
        },
        {"role": "user", "content": policy["query"]},
    ]
    dialogue: list[dict[str, Any]] = []

    for turn in range(1, int(policy["max_steps"]) + 1):
        for _attempt in range(2):
            inference = _proxy.post(
                "/inference/chat/completions",
                json_data={
                    "model": _model(),
                    "messages": messages,
                    "tools": policy["tools"],
                    "tool_choice": "required",
                    "temperature": 0,
                },
            )
            if inference is None:
                raise RuntimeError("inference request failed")
            assistant = inference["choices"][0]["message"]
            assistant_content = assistant.get("content") or ""
            tool_calls = (assistant.get("tool_calls") or [])[:max_calls]
            if tool_calls:
                break
            dialogue.append(
                {"role": "assistant", "content": assistant_content}
            )
            messages.extend(
                [
                    {"role": "assistant", "content": assistant_content},
                    {
                        "role": "user",
                        "content": (
                            "The environment has not reported done=true. "
                            "Choose one of the supplied tools to continue."
                        ),
                    },
                ]
            )
        else:
            raise RuntimeError("model returned no tool call before completion")

        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
        )
        group_id = f"{problem_id}-turn-{turn}"
        envelope = {
            **binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": turn,
            "calls": [
                {
                    "call_id": f"{group_id}-{index}",
                    "action": {
                        "name": call["function"]["name"],
                        "args": _arguments(call),
                    },
                }
                for index, call in enumerate(tool_calls, start=1)
            ],
        }
        result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            raise RuntimeError("environment call failed")
        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
                "environment_result": result,
            }
        )

        for tool_call, call_result in zip(tool_calls, result["calls"], strict=True):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(call_result["observation"]),
                }
            )
        if result.get("user_message"):
            messages.append(
                {"role": "user", "content": result["user_message"]["content"]}
            )
        if any(call["observation"]["done"] for call in result["calls"]):
            break

    return dialogue


__all__ = ["agent_main"]
