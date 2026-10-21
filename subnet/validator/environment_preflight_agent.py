"""Deterministic validator-owned policy used by the environment preflight."""

from __future__ import annotations

import os
from typing import Any

import requests


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute validator-supplied actions, one environment step each, through the sandbox's proxy."""

    environment = problem_data["environment"]
    binding = environment["binding"]
    actions = environment["actions"]
    proxy_url = os.environ.get("SANDBOX_PROXY_URL", "http://proxy:80").rstrip("/")
    dialogue: list[dict[str, Any]] = []

    for step, action in enumerate(actions, start=1):
        response = requests.post(
            f"{proxy_url}/environment/call",
            json={**binding, "step": step, "action": action},
            timeout=70,
        )
        response.raise_for_status()
        result = response.json()
        if bool(result["finished"]) != (step == len(actions)):
            raise RuntimeError(f"environment preflight step {step} finished={result['finished']}")
        dialogue.append(
            {
                "role": "assistant",
                "content": f"completed environment preflight step {step}",
                "environment_result": result,
            }
        )

    return dialogue


__all__ = ["agent_main"]
