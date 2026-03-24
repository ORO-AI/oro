"""Tests for proxy_client Authorization header injection."""

from unittest.mock import patch, MagicMock


from src.agent.proxy_client import ProxyClient


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
