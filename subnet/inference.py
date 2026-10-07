"""Provider credential selection shared by local evaluation runners."""

import os

_OPENROUTER_INFERENCE_BASE_URL = "https://openrouter.ai/api/v1"


def resolve_inference_credentials() -> tuple[str | None, str | None, str | None]:
    """Resolve (api_key, provider, base_url) for the local test rig.

    Accepts only OpenRouter credentials and rejects unsupported providers
    before a local evaluation starts. Returns (None, None, None) if no key is set.
    """
    or_key = os.environ.get("OPENROUTER_API_KEY")
    explicit = os.environ.get("INFERENCE_PROVIDER")
    if explicit and explicit != "openrouter":
        raise ValueError("INFERENCE_PROVIDER must be 'openrouter'")
    if or_key:
        return or_key, "openrouter", _OPENROUTER_INFERENCE_BASE_URL
    return None, None, None
