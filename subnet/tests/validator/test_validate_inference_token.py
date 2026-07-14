"""Tests for Validator._validate_inference_token and _validation_model_for."""

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def _stub_bittensor_core():
    """Stub out bittensor.core.subtensor so validator.main can be imported."""
    if "bittensor.core.subtensor" not in sys.modules:
        bt_core = sys.modules.setdefault(
            "bittensor.core", types.ModuleType("bittensor.core")
        )
        subtensor_mod = types.ModuleType("bittensor.core.subtensor")
        subtensor_mod.Subtensor = MagicMock()
        sys.modules["bittensor.core.subtensor"] = subtensor_mod
        bt_core.subtensor = subtensor_mod


_stub_bittensor_core()

from validator.main import Validator  # noqa: E402


class TestValidationModelFor:
    def test_chutes_returns_expected_model(self):
        assert Validator._validation_model_for("chutes") == "Qwen/Qwen3-32B-TEE"

    def test_openrouter_returns_expected_model(self):
        assert Validator._validation_model_for("openrouter") == "openai/gpt-oss-20b"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="unknown inference provider"):
            Validator._validation_model_for("unknown_provider")


class TestValidateInferenceToken:
    def _mock_resp(self, status_code: int, json_body: dict | None = None):
        mock = MagicMock()
        mock.status_code = status_code
        if json_body is not None:
            mock.json.return_value = json_body
        return mock

    def test_200_returns_valid(self):
        with patch("validator.main.requests.post", return_value=self._mock_resp(200)):
            ok, reason = Validator._validate_inference_token(
                "tok", "https://llm.chutes.ai/v1", "Qwen/Qwen3-32B-TEE"
            )
        assert ok is True
        assert reason == ""

    def test_401_persists_fails_after_retries(self):
        # A genuinely-bad key stays 401 through every attempt → fails fast.
        # Retries the token smoke-test on 401 to absorb key-propagation lag.
        with (
            patch("validator.main.time.sleep") as mock_sleep,
            patch(
                "validator.main.requests.post", return_value=self._mock_resp(401)
            ) as mock_post,
        ):
            ok, reason = Validator._validate_inference_token(
                "tok", "https://llm.chutes.ai/v1", "Qwen/Qwen3-32B-TEE"
            )
        assert ok is False
        assert "401" in reason
        assert mock_post.call_count == 6  # 1 initial + 5 retries
        assert mock_sleep.call_count == 5

    def test_401_then_200_retries_and_passes(self):
        # A freshly-minted key that 401s from propagation lag, then goes live.
        with (
            patch("validator.main.time.sleep") as mock_sleep,
            patch(
                "validator.main.requests.post",
                side_effect=[self._mock_resp(401), self._mock_resp(200)],
            ) as mock_post,
        ):
            ok, reason = Validator._validate_inference_token(
                "tok", "https://llm.chutes.ai/v1", "Qwen/Qwen3-32B-TEE"
            )
        assert ok is True
        assert reason == ""
        assert mock_post.call_count == 2
        assert mock_sleep.call_count == 1

    def test_401_retry_uses_exponential_backoff_with_jitter(self):
        # Verify per-attempt sleep grows linearly with attempt number
        # under the jittered exponential curve. Assert the invariant
        # (sleep[i] == base * (i + 1) at jitter=1.0) rather than the
        # literal values, so retunes only touch the source constant.
        sleeps: list[float] = []
        with (
            patch("validator.main.time.sleep", side_effect=sleeps.append),
            patch("validator.main.random.uniform", return_value=1.0),
            patch(
                "validator.main.requests.post", return_value=self._mock_resp(401)
            ),
        ):
            Validator._validate_inference_token(
                "tok", "https://llm.chutes.ai/v1", "Qwen/Qwen3-32B-TEE"
            )
        base = float(os.environ.get("ORO_TOKEN_401_BACKOFF_BASE", "5.0"))
        assert sleeps == [base * (i + 1) for i in range(5)]

    def test_401_backoff_base_env_override(self):
        # Verify ORO_TOKEN_401_BACKOFF_BASE lets on-call retune without
        # a code change. Pin jitter to 1.0 so sleeps map 1:1 onto the
        # base * (i + 1) curve for whatever base the env sets.
        sleeps: list[float] = []
        with (
            patch.dict(os.environ, {"ORO_TOKEN_401_BACKOFF_BASE": "2.0"}),
            patch("validator.main.time.sleep", side_effect=sleeps.append),
            patch("validator.main.random.uniform", return_value=1.0),
            patch(
                "validator.main.requests.post", return_value=self._mock_resp(401)
            ),
        ):
            Validator._validate_inference_token(
                "tok", "https://llm.chutes.ai/v1", "Qwen/Qwen3-32B-TEE"
            )
        assert sleeps == [2.0, 4.0, 6.0, 8.0, 10.0]

    def test_401_retry_jitter_bounds(self):
        # Verify jitter multiplier is drawn from [0.5, 1.5]. A
        # too-narrow range defeats the point of spreading the mint burst.
        with (
            patch("validator.main.time.sleep"),
            patch("validator.main.random.uniform") as mock_uniform,
            patch(
                "validator.main.requests.post", return_value=self._mock_resp(401)
            ),
        ):
            mock_uniform.return_value = 1.0
            Validator._validate_inference_token(
                "tok", "https://llm.chutes.ai/v1", "Qwen/Qwen3-32B-TEE"
            )
        for call in mock_uniform.call_args_list:
            low, high = call.args
            assert 0.5 <= low < high <= 1.5

    def test_402_returns_invalid_with_message(self):
        json_body = {"detail": {"message": "insufficient balance"}}
        with patch(
            "validator.main.requests.post",
            return_value=self._mock_resp(402, json_body),
        ):
            ok, reason = Validator._validate_inference_token(
                "tok", "https://llm.chutes.ai/v1", "Qwen/Qwen3-32B-TEE"
            )
        assert ok is False
        assert "insufficient balance" in reason

    def test_429_returns_valid(self):
        with patch("validator.main.requests.post", return_value=self._mock_resp(429)):
            ok, reason = Validator._validate_inference_token(
                "tok", "https://llm.chutes.ai/v1", "Qwen/Qwen3-32B-TEE"
            )
        assert ok is True
        assert reason == ""

    def test_5xx_returns_valid(self):
        with patch("validator.main.requests.post", return_value=self._mock_resp(503)):
            ok, reason = Validator._validate_inference_token(
                "tok", "https://llm.chutes.ai/v1", "Qwen/Qwen3-32B-TEE"
            )
        assert ok is True
        assert reason == ""

    def test_exception_returns_valid(self):
        with patch(
            "validator.main.requests.post", side_effect=Exception("connection refused")
        ):
            ok, reason = Validator._validate_inference_token(
                "tok", "https://llm.chutes.ai/v1", "Qwen/Qwen3-32B-TEE"
            )
        assert ok is True
        assert reason == ""

    def test_url_built_from_base_url(self):
        mock_resp = self._mock_resp(200)
        with patch("validator.main.requests.post", return_value=mock_resp) as mock_post:
            Validator._validate_inference_token(
                "tok", "https://llm.chutes.ai/v1", "Qwen/Qwen3-32B-TEE"
            )
        args, kwargs = mock_post.call_args
        assert args[0] == "https://llm.chutes.ai/v1/chat/completions"

    def test_url_strips_trailing_slash(self):
        mock_resp = self._mock_resp(200)
        with patch("validator.main.requests.post", return_value=mock_resp) as mock_post:
            Validator._validate_inference_token(
                "tok", "https://llm.chutes.ai/v1/", "Qwen/Qwen3-32B-TEE"
            )
        args, kwargs = mock_post.call_args
        assert args[0] == "https://llm.chutes.ai/v1/chat/completions"

    def test_validate_inference_token_against_openrouter(self):
        """Verify the validator can smoke-test an OR-style endpoint."""
        mock_resp = self._mock_resp(200)
        with patch("validator.main.requests.post", return_value=mock_resp) as mock_post:
            ok, reason = Validator._validate_inference_token(
                "sk-or-test", "https://openrouter.ai/api/v1", "openai/gpt-oss-20b"
            )

        assert ok is True
        assert reason == ""
        args, kwargs = mock_post.call_args
        assert args[0] == "https://openrouter.ai/api/v1/chat/completions"
        assert kwargs["json"]["model"] == "openai/gpt-oss-20b"
