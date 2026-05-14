"""Tests for validator.inference_token (validate_inference_token, validation_model_for)."""

from unittest.mock import MagicMock, patch

import pytest

from validator.inference_token import validate_inference_token, validation_model_for


class TestValidationModelFor:
    def test_chutes_returns_expected_model(self):
        assert validation_model_for("chutes") == "Qwen/Qwen3-32B-TEE"

    def test_openrouter_returns_expected_model(self):
        assert validation_model_for("openrouter") == "openai/gpt-oss-20b"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="unknown inference provider"):
            validation_model_for("unknown_provider")


class TestValidateInferenceToken:
    def _mock_resp(self, status_code: int, json_body: dict | None = None):
        mock = MagicMock()
        mock.status_code = status_code
        if json_body is not None:
            mock.json.return_value = json_body
        return mock

    def test_200_returns_valid(self):
        with patch(
            "validator.inference_token.requests.post",
            return_value=self._mock_resp(200),
        ):
            ok, reason = validate_inference_token(
                "tok", "https://llm.chutes.ai/v1", "Qwen/Qwen3-32B-TEE"
            )
        assert ok is True
        assert reason == ""

    def test_401_returns_invalid(self):
        with patch(
            "validator.inference_token.requests.post",
            return_value=self._mock_resp(401),
        ):
            ok, reason = validate_inference_token(
                "tok", "https://llm.chutes.ai/v1", "Qwen/Qwen3-32B-TEE"
            )
        assert ok is False
        assert "401" in reason

    def test_402_returns_invalid_with_message(self):
        json_body = {"detail": {"message": "insufficient balance"}}
        with patch(
            "validator.inference_token.requests.post",
            return_value=self._mock_resp(402, json_body),
        ):
            ok, reason = validate_inference_token(
                "tok", "https://llm.chutes.ai/v1", "Qwen/Qwen3-32B-TEE"
            )
        assert ok is False
        assert "insufficient balance" in reason

    def test_429_returns_valid(self):
        with patch(
            "validator.inference_token.requests.post",
            return_value=self._mock_resp(429),
        ):
            ok, reason = validate_inference_token(
                "tok", "https://llm.chutes.ai/v1", "Qwen/Qwen3-32B-TEE"
            )
        assert ok is True
        assert reason == ""

    def test_5xx_returns_valid(self):
        with patch(
            "validator.inference_token.requests.post",
            return_value=self._mock_resp(503),
        ):
            ok, reason = validate_inference_token(
                "tok", "https://llm.chutes.ai/v1", "Qwen/Qwen3-32B-TEE"
            )
        assert ok is True
        assert reason == ""

    def test_exception_returns_valid(self):
        with patch(
            "validator.inference_token.requests.post",
            side_effect=Exception("connection refused"),
        ):
            ok, reason = validate_inference_token(
                "tok", "https://llm.chutes.ai/v1", "Qwen/Qwen3-32B-TEE"
            )
        assert ok is True
        assert reason == ""

    def test_url_built_from_base_url(self):
        mock_resp = self._mock_resp(200)
        with patch(
            "validator.inference_token.requests.post", return_value=mock_resp
        ) as mock_post:
            validate_inference_token(
                "tok", "https://llm.chutes.ai/v1", "Qwen/Qwen3-32B-TEE"
            )
        args, kwargs = mock_post.call_args
        assert args[0] == "https://llm.chutes.ai/v1/chat/completions"

    def test_url_strips_trailing_slash(self):
        mock_resp = self._mock_resp(200)
        with patch(
            "validator.inference_token.requests.post", return_value=mock_resp
        ) as mock_post:
            validate_inference_token(
                "tok", "https://llm.chutes.ai/v1/", "Qwen/Qwen3-32B-TEE"
            )
        args, kwargs = mock_post.call_args
        assert args[0] == "https://llm.chutes.ai/v1/chat/completions"

    def test_validate_inference_token_against_openrouter(self):
        """Verify the validator can smoke-test an OR-style endpoint."""
        mock_resp = self._mock_resp(200)
        with patch(
            "validator.inference_token.requests.post", return_value=mock_resp
        ) as mock_post:
            ok, reason = validate_inference_token(
                "sk-or-test", "https://openrouter.ai/api/v1", "openai/gpt-oss-20b"
            )

        assert ok is True
        assert reason == ""
        args, kwargs = mock_post.call_args
        assert args[0] == "https://openrouter.ai/api/v1/chat/completions"
        assert kwargs["json"]["model"] == "openai/gpt-oss-20b"
