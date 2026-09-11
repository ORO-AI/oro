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


class InferenceProviderError(RuntimeError):
    """A user-simulator inference call failed with an upstream error body.

    Carries the provider's status + truncated body so ``session_registry``
    can safely surface a trusted, provider-sourced message into the
    episode ledger's ``error_detail`` without also whitelisting arbitrary
    ``str(exc)`` from every simulator failure (which could leak private
    task material if a bespoke simulator raised an attacker-influenced
    message — see ORO-1866 disclosure canary).
    """

    def __init__(self, status: int | None, body: str) -> None:
        self.status = status
        self.body = body
        super().__init__(
            f"user simulator inference returned no completion "
            f"(status={status} body={body!r})"
        )


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

        # Use ``post_verbose`` — its return value carries the upstream
        # status+body directly. Callers must not rely on a shared client
        # attribute (see ORO-2191 review): a single ProxyClient shared by
        # concurrent sessions would let session B's failure overwrite
        # session A's just before A reads it, corrupting A's ledger.
        backoff_s = _INITIAL_BACKOFF_S
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            result = self._client.post_verbose(
                "/inference/chat/completions",
                json_data=request,
            )
            data = result.data
            choices = data.get("choices") if isinstance(data, dict) else None
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

            # Extract the upstream signal from this call's own return value —
            # never from a shared attribute. ``result.error`` describes the
            # network vs upstream distinction so log labels can be honest
            # ("network" vs "proxy non-200") instead of collapsing both.
            error = result.error or {}
            kind = error.get("kind", "unknown")
            upstream_status = error.get("status")
            upstream_body = (error.get("body") or "").strip()
            if data is None and result.error is None:
                # ProxyClient returned no data but no error info — treat as
                # a malformed 200 body (see ORO-2191 non-blocker: a JSON
                # ``null`` body decodes to None on a 200 response).
                kind, upstream_status, upstream_body = "malformed", 200, ""
            failure_label = (
                "network error"
                if kind == "network"
                else "malformed body"
                if kind == "malformed"
                else "proxy non-200"
            )
            log_detail = (
                f"{failure_label} status={upstream_status} body={upstream_body!r}"
            )
            if attempt >= _MAX_ATTEMPTS:
                logger.error(
                    "user simulator inference failed after %d attempts (%s)",
                    _MAX_ATTEMPTS,
                    log_detail,
                )
                raise InferenceProviderError(
                    status=upstream_status,
                    body=upstream_body,
                )
            logger.warning(
                "user simulator inference failed on attempt %d/%d (%s); retrying in %.2fs",
                attempt,
                _MAX_ATTEMPTS,
                log_detail,
                backoff_s,
            )
            await asyncio.sleep(backoff_s)
            backoff_s *= _BACKOFF_MULTIPLIER


__all__ = ["InferenceProviderError", "SimulatorCompletion"]
