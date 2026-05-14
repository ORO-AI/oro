"""Smoke-test miner-supplied inference tokens before consuming a run slot.

Catches invalid tokens (401) and zero-balance accounts (402) cheaply against
any OpenAI-compatible chat/completions endpoint. Transient errors (5xx, 429,
timeouts) return success so we don't fail runs for upstream provider blips.
"""

from __future__ import annotations

import requests
from bittensor.utils.btlogging import logging


def validation_model_for(provider: str) -> str:
    """Pick a small model present on each provider for the smoke-test."""
    if provider == "chutes":
        return "Qwen/Qwen3-32B-TEE"
    if provider == "openrouter":
        return "openai/gpt-oss-20b"
    raise ValueError(f"unknown inference provider: {provider}")


def validate_inference_token(
    access_token: str, base_url: str, model: str
) -> tuple[bool, str]:
    """Make a 1-token completion against `base_url`. Returns (ok, reason)."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    try:
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            },
            timeout=15,
        )
        if resp.status_code == 200:
            return True, ""
        if resp.status_code == 401:
            return False, "Inference token invalid or expired (HTTP 401)"
        if resp.status_code == 402:
            detail = resp.json().get("detail", {})
            msg = (
                detail.get("message", str(detail))
                if isinstance(detail, dict)
                else str(detail)
            )
            return False, f"Inference account has no credits ({msg})"
        if resp.status_code == 429:
            return True, ""
        logging.warning(
            "Inference token validation inconclusive: status=%s url=%s",
            resp.status_code,
            url,
        )
        return True, ""
    except Exception as exc:
        logging.warning("Inference token validation error against %s: %s", url, exc)
        return True, ""
