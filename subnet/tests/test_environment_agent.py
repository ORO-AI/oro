"""Focused coverage for the submit-compatible environment example agent."""

import pytest

from src.agent import environment_agent


@pytest.mark.parametrize(
    ("provider", "expected_model"),
    [
        ("chutes", "deepseek-ai/DeepSeek-V3.2-TEE"),
        ("openrouter", "deepseek/deepseek-v3.2"),
        ("unknown", "deepseek-ai/DeepSeek-V3.2-TEE"),
    ],
)
def test_selects_a_provider_compatible_model(
    provider: str,
    expected_model: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFERENCE_PROVIDER", provider)
    monkeypatch.delenv("SANDBOX_MODEL", raising=False)

    assert environment_agent._model() == expected_model


def test_honors_an_explicit_model_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFERENCE_PROVIDER", "chutes")
    monkeypatch.setenv("SANDBOX_MODEL", "custom/model")

    assert environment_agent._model() == "custom/model"
