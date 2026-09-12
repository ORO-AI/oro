"""Provider credential selection shared by local evaluation runners."""

import os

_CHUTES_INFERENCE_BASE_URL = "https://llm.chutes.ai/v1"
_OPENROUTER_INFERENCE_BASE_URL = "https://openrouter.ai/api/v1"


def resolve_inference_credentials() -> tuple[str | None, str | None, str | None]:
    """Resolve (api_key, provider, base_url) for the local test rig.

    Honors an explicit INFERENCE_PROVIDER override; otherwise infers from
    which key env var is set. Returns (None, None, None) if neither is set.
    """
    or_key = os.environ.get("OPENROUTER_API_KEY")
    chutes_key = os.environ.get("CHUTES_API_KEY")

    explicit = os.environ.get("INFERENCE_PROVIDER")
    if explicit == "openrouter" and or_key:
        return or_key, "openrouter", _OPENROUTER_INFERENCE_BASE_URL
    if explicit == "chutes" and chutes_key:
        return chutes_key, "chutes", _CHUTES_INFERENCE_BASE_URL

    if or_key:
        return or_key, "openrouter", _OPENROUTER_INFERENCE_BASE_URL
    if chutes_key:
        return chutes_key, "chutes", _CHUTES_INFERENCE_BASE_URL
    return None, None, None
