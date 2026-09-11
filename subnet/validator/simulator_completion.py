"""Miner-funded completion adapter for validator-owned user simulation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.agent.proxy_client import ProxyClient

logger = logging.getLogger(__name__)

# The shopper-simulator LLM call is a single-shot dependency in the middle
# of a task episode; a lone transient (429, 5xx, socket reset, empty body)
# used to trip ``environment_error`` on the task, and once eight of a
# 90-task race pack tripped, ``generated_evaluation.aggregate_results``
# sank the whole run. Prod on 2026-09-10 lost dozens of pack runs to this
# pattern (see ORO-2189). Retry a bounded number of times with
# exponential backoff before we escalate — the provider almost always
# recovers on the next attempt.
_MAX_ATTEMPTS = 3
_INITIAL_BACKOFF_S = 0.5
_BACKOFF_MULTIPLIER = 2.0


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

        backoff_s = _INITIAL_BACKOFF_S
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            response = self._client.post(
                "/inference/chat/completions",
                json_data=request,
            )
            choices = response.get("choices") if isinstance(response, dict) else None
            choice = choices[0] if isinstance(choices, list) and choices else None
            message = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(message, dict):
                if attempt > 1:
                    logger.warning(
                        "user simulator inference recovered on attempt %d/%d",
                        attempt,
                        _MAX_ATTEMPTS,
                    )
                return {
                    "text": message.get("content") or "",
                    "tool_calls": message.get("tool_calls") or [],
                    "finish_reason": choice.get("finish_reason"),
                }

            # Distinguish "proxy swallowed a non-200" from "provider returned
            # 200 with a malformed body" for post-hoc diagnosis. On the
            # non-200 branch include the upstream provider's actual status
            # and error body via ProxyClient.last_error (ORO-2191) so the
            # ledger's error_detail carries the reason (rate_limit_exceeded
            # vs insufficient_credits vs invalid_api_key vs model_unavailable),
            # not just a generic RuntimeError. Both branches are retriable
            # since the shopper-simulator turn is idempotent.
            if response is None:
                last_error = getattr(self._client, "last_error", None) or {}
                upstream_status = last_error.get("status")
                upstream_body = (last_error.get("body") or "").strip()
                detail = (
                    f"proxy non-200 status={upstream_status} body={upstream_body!r}"
                )
            else:
                detail = "malformed body"
            failure = f"user simulator inference returned no completion ({detail})"
            if attempt >= _MAX_ATTEMPTS:
                logger.error(
                    "user simulator inference failed after %d attempts (%s)",
                    _MAX_ATTEMPTS,
                    failure,
                )
                raise RuntimeError(failure)
            logger.warning(
                "user simulator inference failed on attempt %d/%d (%s); retrying in %.2fs",
                attempt,
                _MAX_ATTEMPTS,
                failure,
                backoff_s,
            )
            await asyncio.sleep(backoff_s)
            backoff_s *= _BACKOFF_MULTIPLIER


__all__ = ["SimulatorCompletion"]
