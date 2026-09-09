"""Miner-funded completion adapter for validator-owned user simulation."""

from __future__ import annotations

from typing import Any

from src.agent.proxy_client import ProxyClient


class SimulatorCompletion:
    """Expose the validator proxy as an ``oro-env-runtime`` completion callable."""

    def __init__(
        self,
        access_token: str,
        *,
        proxy_url: str = "http://proxy:80",
        timeout_s: int = 55,
        client: ProxyClient | None = None,
    ) -> None:
        if not access_token:
            raise ValueError("miner inference access token is required")
        self._client = client or ProxyClient(
            proxy_url=proxy_url,
            api_key=access_token,
            timeout=timeout_s,
            max_retries=1,
        )

    async def __call__(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
        temperature: float | None = 0.0,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if temperature is not None:
            request["temperature"] = temperature
        if tools:
            request["tools"] = tools

        response = self._client.post(
            "/inference/chat/completions",
            json_data=request,
        )
        choices = response.get("choices") if isinstance(response, dict) else None
        choice = choices[0] if isinstance(choices, list) and choices else None
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            raise RuntimeError("user simulator inference returned no completion")

        return {
            "text": message.get("content") or "",
            "tool_calls": message.get("tool_calls") or [],
            "finish_reason": choice.get("finish_reason"),
        }


__all__ = ["SimulatorCompletion"]
