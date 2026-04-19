"""Tests for proxy_client Authorization header injection and inference stats."""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock


from src.agent.proxy_client import InferenceStats, ProxyClient, RequestLog


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
    """Tests for incremental inference stats writing."""

    def test_writes_cumulative_after_each_call(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            with patch.dict("os.environ", {"PROBLEM_DATA": '{"problem_id": "p-1"}'}):
                stats = InferenceStats(stats_file=path)
                stats.record_success()
                stats.record_failure()
                stats.record_success()

            with open(path) as f:
                lines = [json.loads(line) for line in f if line.strip()]

            assert len(lines) == 3
            assert lines[0]["inference_total"] == 1
            assert lines[1]["inference_total"] == 2
            assert lines[2] == {
                "problem_id": "p-1",
                "inference_success": 2,
                "inference_failed": 1,
                "inference_total": 3,
            }
        finally:
            os.unlink(path)

    def test_no_file_does_not_crash(self):
        stats = InferenceStats(stats_file=None)
        stats.record_success()
        stats.record_failure()

    def test_problem_id_from_env(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            with patch.dict(
                "os.environ", {"PROBLEM_DATA": '{"problem_id": "uuid-123"}'}
            ):
                stats = InferenceStats(stats_file=path)
                stats.record_success()

            with open(path) as f:
                entry = json.loads(f.readline())
            assert entry["problem_id"] == "uuid-123"
        finally:
            os.unlink(path)

    def test_missing_problem_data_uses_unknown(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            env = os.environ.copy()
            env.pop("PROBLEM_DATA", None)
            with patch.dict("os.environ", env, clear=True):
                stats = InferenceStats(stats_file=path)
                stats.record_failure()

            with open(path) as f:
                entry = json.loads(f.readline())
            assert entry["problem_id"] == "unknown"
        finally:
            os.unlink(path)


class TestRequestLog:
    """Tests for proxy call request logging."""

    def test_records_get_call(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            log = RequestLog(log_file=path)
            log.record(
                method="GET",
                path="/search/find_product",
                params={"q": "laptop", "page": 1},
                status_code=200,
                response_body=[{"product_id": "123", "title": "Laptop"}],
                duration_ms=150.5,
            )

            with open(path) as f:
                entry = json.loads(f.readline())

            assert entry["method"] == "GET"
            assert entry["path"] == "/search/find_product"
            assert entry["params"] == {"q": "laptop", "page": 1}
            assert entry["status_code"] == 200
            assert entry["duration_ms"] == 150.5
            assert entry["response"] == [{"product_id": "123", "title": "Laptop"}]
            assert "timestamp" in entry
        finally:
            os.unlink(path)

    def test_records_post_call(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            log = RequestLog(log_file=path)
            log.record(
                method="POST",
                path="/search/view_product_information",
                json_data={"product_id": "456"},
                status_code=200,
                duration_ms=80.0,
            )

            with open(path) as f:
                entry = json.loads(f.readline())

            assert entry["method"] == "POST"
            assert entry["json_data"] == {"product_id": "456"}
        finally:
            os.unlink(path)

    def test_inference_body_omits_messages(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            log = RequestLog(log_file=path)
            log.record(
                method="POST",
                path="/inference/chat/completions",
                json_data={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "very long prompt..."}],
                    "temperature": 0.7,
                },
                status_code=200,
                duration_ms=2000.0,
            )

            with open(path) as f:
                entry = json.loads(f.readline())

            assert "messages" not in entry["json_data"]
            assert entry["json_data"]["model"] == "gpt-4"
            assert entry["json_data"]["temperature"] == 0.7
        finally:
            os.unlink(path)

    def test_large_response_truncated(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            log = RequestLog(log_file=path)
            large_body = {"data": "x" * 3000}
            log.record(
                method="GET",
                path="/search/find_product",
                status_code=200,
                response_body=large_body,
                duration_ms=100.0,
            )

            with open(path) as f:
                entry = json.loads(f.readline())

            assert entry["response_truncated"] is True
            assert entry["response_length"] > 2000
            assert "response" not in entry
        finally:
            os.unlink(path)

    def test_no_file_does_not_crash(self):
        log = RequestLog(log_file=None)
        log.record(method="GET", path="/search/find_product", status_code=200)

    def test_none_params_omitted(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            log = RequestLog(log_file=path)
            log.record(method="GET", path="/health", status_code=200, duration_ms=5.0)

            with open(path) as f:
                entry = json.loads(f.readline())

            assert "params" not in entry
            assert "json_data" not in entry
        finally:
            os.unlink(path)

    def test_multiple_calls_append(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            log = RequestLog(log_file=path)
            log.record(method="GET", path="/a", status_code=200, duration_ms=1.0)
            log.record(method="GET", path="/b", status_code=200, duration_ms=2.0)
            log.record(method="POST", path="/c", status_code=200, duration_ms=3.0)

            with open(path) as f:
                lines = [json.loads(line) for line in f if line.strip()]

            assert len(lines) == 3
            assert [e["path"] for e in lines] == ["/a", "/b", "/c"]
        finally:
            os.unlink(path)
