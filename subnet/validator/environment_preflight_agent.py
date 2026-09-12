"""Deterministic validator-owned policy used by the environment preflight."""

from __future__ import annotations

import os
from typing import Any

import requests


POLICY_PROTOCOL_VERSION = "oro.environment_preflight_policy.v1"


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute validator-supplied actions through the sandbox's public proxy."""

    environment = problem_data["environment"]
    binding = environment["binding"]
    proxy_url = os.environ.get("SANDBOX_PROXY_URL", "http://proxy:80").rstrip("/")
    dialogue: list[dict[str, Any]] = []
    problem_id = str(problem_data["problem_id"])

    for turn, actions in enumerate(environment["action_groups"], start=1):
        group_id = f"{problem_id}-{turn}"
        response = requests.post(
            f"{proxy_url}/environment/call",
            json={
                **binding,
                "call_id": group_id,
                "idempotency_key": group_id,
                "turn": turn,
                "calls": [
                    {
                        "call_id": f"{group_id}-{index}",
                        "action": action,
                    }
                    for index, action in enumerate(actions, start=1)
                ],
            },
            timeout=70,
        )
        response.raise_for_status()
        result = response.json()
        if turn == 1 and not result.get("user_message", {}).get("content"):
            raise RuntimeError("environment preflight did not receive a shopper reply")
        dialogue.append(
            {
                "role": "assistant",
                "content": f"completed environment preflight turn {turn}",
                "environment_result": result,
            }
        )

    return dialogue


__all__ = ["POLICY_PROTOCOL_VERSION", "agent_main"]
