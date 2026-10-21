"""Minimal agent example for validator-owned world sessions.

Each tool call is one environment step. Its observation is the tool result,
including the shopper's reply to a message. The episode ends when a step
reports ``finished``.
"""

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
    """Let the model choose actions from the environment's tool list, one step per call."""

    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]

    request = policy["query"]
    if policy.get("opening") is not None:
        request += "\n\n" + json.dumps(policy["opening"])
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "Use the supplied tools to serve the shopper. Message the shopper "
                "through its tool; the reply comes back as the tool result. "
                "Continue until a result reports finished=true."
            ),
        },
        {"role": "user", "content": request},
    ]
    dialogue: list[dict[str, Any]] = []
    step = idle = 0
    finished = bool(policy.get("finished"))

    while not finished and step < int(policy["max_steps"]):
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
        content = assistant.get("content") or ""
        tool_calls = assistant.get("tool_calls") or []
        if not tool_calls:
            idle += 1
            if idle > 2:
                raise RuntimeError("model returned no tool call before the episode finished")
            messages += [
                {"role": "assistant", "content": content},
                {"role": "user", "content": "Choose one of the supplied tools to continue."},
            ]
            dialogue.append({"role": "assistant", "content": content})
            continue
        idle = 0
        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        for tool_call in tool_calls:
            if finished:
                result = {"observation": {"error": "the episode has finished"}}
            else:
                step += 1
                result = _proxy.post(
                    "/environment/call",
                    json_data={
                        **binding,
                        "step": step,
                        "action": {"name": tool_call["function"]["name"], "args": _arguments(tool_call)},
                    },
                )
                if result is None:
                    raise RuntimeError("environment call failed")
                finished = bool(result["finished"])
                dialogue.append({"role": "assistant", "content": content, "tool_call": tool_call, "environment_result": result})
            messages.append(
                {"role": "tool", "tool_call_id": tool_call["id"], "content": json.dumps(result["observation"])}
            )

    return dialogue


__all__ = ["agent_main"]
