"""Miner-funded completion adapter for validator-owned user simulation."""

from __future__ import annotations

import asyncio
import logging
import math
import secrets
import threading
import time
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from src.agent.proxy_client import ProxyClient

# Marks the validator's own proxy calls (the disclosure reader's decisions), which the
# agent sandbox may not make. Minted per process and never given to the sandbox: the
# agent holds the same provider key, so the key alone cannot tell the two apart.
VALIDATOR_CALLER_HEADER = "X-ORO-Validator"
VALIDATOR_CALLER_SECRET = secrets.token_urlsafe(32)

logger = logging.getLogger(__name__)

# Simulator calls (the shopper's chat and the disclosure Reader's decisions)
# are single-shot dependencies in the middle of a task episode; a lone
# transient (429, an in-flight credit 402, 5xx, socket reset, empty body)
# used to trip ``environment_error`` on the task. Retry a bounded number
# of times before we escalate — the provider usually recovers.
#
# Backoff shape (5s, 10s, 20s) is sized to step through a real provider
# rate-limit window (typically 30-60s) so the last attempt lands outside
# it. Shorter waits would repeatedly hit the same closed window.
#
# Wall budget: request latency counts against ``simulator_timeout_s``
# (60s at session_registry), not just the sleeps, and every phase of a
# simulator turn shares that deadline (``SIMULATOR_DEADLINE``). Skip
# further retries once the projected wall (elapsed + next backoff + a
# safety margin for the next request) would pass this call's budget or
# the turn's deadline, so we surface a clean ``InferenceProviderError``
# instead of getting quarantined mid-sleep by the outer ``TimeoutError``.
SIMULATOR_DEADLINE: ContextVar[float] = ContextVar("simulator_deadline")
_MAX_ATTEMPTS = 4
_RETRY_BACKOFFS_S = (5.0, 10.0, 20.0)
_WALL_BUDGET_S = 55.0
_NEXT_ATTEMPT_HEADROOM_S = 3.0


class InferenceProviderError(RuntimeError):
    """A user-simulator inference call failed with an upstream error body.

    Carries the provider's status + truncated body so ``session_registry``
    can safely surface a trusted, provider-sourced message into the
    episode ledger's ``error_detail`` without also whitelisting arbitrary
    ``str(exc)`` from every simulator failure (which could leak private
    task material if a bespoke simulator raised an attacker-influenced
    message — see ORO-1866 disclosure canary).
    """

    def __init__(self, status: int | None, body: str, *, network: bool = False) -> None:
        self.status = status
        self.body = body
        self.network = network
        super().__init__(
            f"user simulator inference returned no completion "
            f"(status={status} body={body!r})"
        )

    @property
    def key_exhausted(self) -> bool:
        """The run key's limit (403) or the miner account's credit (402) is spent.

        The in-flight reservation 402 clears as other calls finish, so it doesn't count.
        """
        body = self.body.lower()
        if self.status == 403:
            return "key limit exceeded" in body
        return self.status == 402 and (
            "insufficient credits" in body or "requires more credits" in body
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
        inference_stats_file: str | None = None,
        episode_id: str | None = None,
        key_exhausted_event: threading.Event | None = None,
    ) -> None:
        if not access_token:
            raise ValueError("inference access token is required")
        self._access_token = access_token
        self._key_exhausted_event = key_exhausted_event
        # Configured Readers require Jev's decisions endpoint, which only OpenRouter
        # provides; runtime setup rejects them when this callable is unavailable.
        if not access_token.startswith("sk-or-"):
            self.decide = None
        self._client = client or ProxyClient(
            proxy_url=proxy_url,
            api_key=access_token,
            timeout=timeout_s,
            max_retries=1,
            inference_stats_file=inference_stats_file,
            inference_stats_problem_id=episode_id,
        )
        self._client.headers = {VALIDATOR_CALLER_HEADER: VALIDATOR_CALLER_SECRET}

    async def __call__(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
        temperature: float | None = 0.0,
        reasoning: bool | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
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
        if reasoning is not None:
            request["reasoning"] = (
                reasoning if isinstance(reasoning, dict) else {"enabled": reasoning}
            )
            if not self._access_token.startswith("sk-or-"):
                preference = request["reasoning"]
                if model != "Qwen/Qwen3.5-397B-A17B-TEE" or preference.get("enabled") is not False or len(preference) != 1:
                    raise ValueError("Unsupported simulator reasoning preference for Chutes")
                # Chutes' Qwen non-thinking mode uses the vLLM template switch.
                request["chat_template_kwargs"] = {"enable_thinking": False}
                del request["reasoning"]
        if response_format is not None:
            request["response_format"] = response_format

        def completion(data: Any) -> dict[str, Any] | None:
            choices = data.get("choices") if isinstance(data, dict) else None
            choice = choices[0] if isinstance(choices, list) and choices else None
            message = choice.get("message") if isinstance(choice, dict) else None
            if not isinstance(message, dict):
                return None
            return {
                "text": (message.get("content") or "").replace(self._access_token, "[REDACTED]"),
                "tool_calls": message.get("tool_calls") or [],
                "finish_reason": choice.get("finish_reason"),
            }

        return await self._post("/inference/chat/completions", request, completion)

    async def decide(
        self, model: str, state: dict[str, Any], questions: dict[str, Any]
    ) -> dict[str, Any]:
        """Typed answers from a decisions model, on the same run key and retries.

        A configured Reader has no other judge to fall back to when this raises.
        """
        try:
            return await self._post(
                "/inference/alpha/decisions",
                {"model": model, "state": state, "questions": questions},
                lambda data: data if isinstance(data, dict) else None,
            )
        except InferenceProviderError as exc:
            # No response is an outage: a situation without a Reader asks its next judge.
            if exc.network:
                raise ConnectionError(exc.body or "no response") from exc
            raise

    async def _post(
        self, path: str, json_data: dict[str, Any], parse: Callable[[Any], Any]
    ) -> Any:
        """POST with bounded retries; ``parse`` returns the result, or None if malformed."""
        deadline = min(time.monotonic() + _WALL_BUDGET_S, SIMULATOR_DEADLINE.get(math.inf))
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            # Use ``post_verbose_async`` — its return value carries the upstream
            # status+body directly. Callers must not rely on a shared client
            # attribute (see ORO-2191 review): a single ProxyClient shared by
            # concurrent sessions would let session B's failure overwrite
            # session A's just before A reads it, corrupting A's ledger.
            result = await self._client.post_verbose_async(path, json_data=json_data)
            parsed = parse(result.data)
            if parsed is not None:
                if attempt > 1:
                    logger.warning(
                        "user simulator inference %s recovered on attempt %d/%d",
                        path,
                        attempt,
                        _MAX_ATTEMPTS,
                    )
                return parsed

            # Extract the upstream signal from this call's own return value —
            # never from a shared attribute. ``result.error`` describes the
            # network vs upstream distinction so log labels can be honest
            # ("network" vs "proxy non-200") instead of collapsing both.
            error = result.error or {}
            kind = error.get("kind", "unknown")
            upstream_status = error.get("status")
            upstream_body = (error.get("body") or "").strip().replace(self._access_token, "[REDACTED]")
            if result.data is None and result.error is None:
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
                f"{path} {failure_label} status={upstream_status} body={upstream_body!r}"
            )
            provider_error = InferenceProviderError(
                upstream_status, upstream_body, network=kind == "network"
            )
            if provider_error.key_exhausted:
                if self._key_exhausted_event is not None:
                    self._key_exhausted_event.set()
                logger.warning("inference key budget exhausted during user simulation")
                raise provider_error
            if attempt >= _MAX_ATTEMPTS:
                logger.error(
                    "user simulator inference failed after %d attempts (%s)",
                    _MAX_ATTEMPTS,
                    log_detail,
                )
                raise provider_error
            backoff_s = _RETRY_BACKOFFS_S[attempt - 1]
            remaining_s = deadline - time.monotonic()
            if backoff_s + _NEXT_ATTEMPT_HEADROOM_S > remaining_s:
                logger.error(
                    "user simulator inference wall budget exhausted "
                    "after %d attempts (%s); remaining=%.2fs",
                    attempt,
                    log_detail,
                    remaining_s,
                )
                raise provider_error
            logger.warning(
                "user simulator inference failed on attempt %d/%d (%s); "
                "retrying in %.2fs",
                attempt,
                _MAX_ATTEMPTS,
                log_detail,
                backoff_s,
            )
            await asyncio.sleep(backoff_s)

__all__ = ["InferenceProviderError", "SimulatorCompletion"]
