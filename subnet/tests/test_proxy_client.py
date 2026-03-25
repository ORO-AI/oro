"""Tests for proxy_client Authorization header injection and inference stats."""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock


from src.agent.proxy_client import InferenceStats, ProxyClient


class TestProxyClientAuth:
    """Tests for Authorization header on inference requests."""

    def _mock_response(self, status_code=200, json_data=None):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data or {"content": "ok"}
        return resp

    @patch("src.agent.proxy_client.requests")
    def test_inference_post_includes_auth_header(self, mock_requests):
        mock_requests.post.return_value = self._mock_response()
        client = ProxyClient(proxy_url="http://proxy:80", api_key="test-token")

        client.post("/inference/chat/completions", json_data={"model": "test"})

        _, kwargs = mock_requests.post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer test-token"

    @patch("src.agent.proxy_client.requests")
    def test_non_inference_post_omits_auth_header(self, mock_requests):
        mock_requests.post.return_value = self._mock_response()
        client = ProxyClient(proxy_url="http://proxy:80", api_key="test-token")

        client.post("/search/find_product", json_data={"q": "laptop"})

        _, kwargs = mock_requests.post.call_args
        assert "Authorization" not in kwargs.get("headers", {})

    @patch("src.agent.proxy_client.requests")
    def test_no_api_key_omits_auth_header(self, mock_requests):
        mock_requests.post.return_value = self._mock_response()
        client = ProxyClient(proxy_url="http://proxy:80", api_key=None)

        client.post("/inference/chat/completions", json_data={"model": "test"})

        _, kwargs = mock_requests.post.call_args
        assert "Authorization" not in kwargs.get("headers", {})

    def test_api_key_from_env(self):
        with patch.dict("os.environ", {"CHUTES_ACCESS_TOKEN": "env-token"}):
            client = ProxyClient(proxy_url="http://proxy:80")
            assert client.api_key == "env-token"

    def test_explicit_api_key_overrides_env(self):
        with patch.dict("os.environ", {"CHUTES_ACCESS_TOKEN": "env-token"}):
            client = ProxyClient(proxy_url="http://proxy:80", api_key="explicit")
            assert client.api_key == "explicit"


class TestInferenceStats:
    def test_to_dict(self):
        stats = InferenceStats()
        stats.record_success()
        stats.record_success()
        stats.record_failure()
        assert stats.to_dict() == {
            "inference_success": 2,
            "inference_failed": 1,
            "inference_total": 3,
        }

    def test_append_to_file(self):
        stats = InferenceStats()
        stats.record_success()
        stats.record_failure()
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            with patch.dict("os.environ", {"PROBLEM_DATA": '{"problem_id": "p-123"}'}):
                stats.append_to_file(path)
            with open(path) as f:
                entry = json.loads(f.readline())
            assert entry["problem_id"] == "p-123"
            assert entry["inference_success"] == 1
            assert entry["inference_failed"] == 1
            assert entry["inference_total"] == 2
        finally:
            os.unlink(path)

    def test_append_multiple_problems(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            for pid, successes, failures in [("a", 3, 1), ("b", 0, 2)]:
                stats = InferenceStats()
                for _ in range(successes):
                    stats.record_success()
                for _ in range(failures):
                    stats.record_failure()
                with patch.dict("os.environ", {"PROBLEM_DATA": json.dumps({"problem_id": pid})}):
                    stats.append_to_file(path)
            with open(path) as f:
                lines = [json.loads(line) for line in f if line.strip()]
            assert len(lines) == 2
            assert lines[0]["problem_id"] == "a"
            assert lines[1]["problem_id"] == "b"
        finally:
            os.unlink(path)
