"""Tests for ProxyClient — covers per-attempt request logging."""

import json
from unittest.mock import patch, MagicMock

import pytest
import requests

from src.agent.proxy_client import ProxyClient, RequestLog


def _read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture
def log_path(tmp_path):
    return str(tmp_path / "request_log.jsonl")


@pytest.fixture
def client(log_path, monkeypatch):
    monkeypatch.setenv("REQUEST_LOG_FILE", log_path)
    return ProxyClient(proxy_url="http://proxy:80", max_retries=3, retry_delay=0)


def _ok_response(payload=None):
    r = MagicMock(spec=requests.Response)
    r.status_code = 200
    r.json.return_value = payload or {"ok": True}
    return r


def _status_response(code: int):
    r = MagicMock(spec=requests.Response)
    r.status_code = code
    r.json.return_value = {}
    return r


def test_post_logs_one_attempt_on_success(client, log_path):
    with patch("src.agent.proxy_client.requests.post", return_value=_ok_response()):
        client.post("/inference/chat", json_data={"messages": []})

    entries = _read_jsonl(log_path)
    attempts = [e for e in entries if e["kind"] == "attempt"]
    summaries = [e for e in entries if e["kind"] == "summary"]

    assert len(attempts) == 1
    assert attempts[0]["attempt"] == 0
    assert attempts[0]["status_code"] == 200
    assert attempts[0]["method"] == "POST"
    assert attempts[0]["path"] == "/inference/chat"
    assert "error_class" not in attempts[0]
    assert len(summaries) == 1


def test_post_logs_attempt_per_retry_on_5xx(client, log_path):
    responses = [_status_response(503), _status_response(503), _ok_response()]
    with patch("src.agent.proxy_client.requests.post", side_effect=responses), \
         patch("src.agent.proxy_client.time.sleep"):
        client.post("/inference/chat", json_data={"messages": []})

    attempts = [e for e in _read_jsonl(log_path) if e["kind"] == "attempt"]
    assert [a["attempt"] for a in attempts] == [0, 1, 2]
    assert [a["status_code"] for a in attempts] == [503, 503, 200]


def test_post_records_error_class_on_timeout(client, log_path):
    """A hung HTTP call surfaces as an attempt with error_class=ReadTimeout."""
    with patch(
        "src.agent.proxy_client.requests.post",
        side_effect=requests.exceptions.ReadTimeout("read timed out"),
    ), patch("src.agent.proxy_client.time.sleep"):
        result = client.post("/inference/chat", json_data={"messages": []})

    assert result is None
    attempts = [e for e in _read_jsonl(log_path) if e["kind"] == "attempt"]
    assert len(attempts) == 3
    for a in attempts:
        assert a["status_code"] is None
        assert a["error_class"] == "ReadTimeout"
        assert "duration_ms" in a


def test_attempt_duration_ms_present_per_call(client, log_path):
    """Each attempt entry carries its own duration_ms (not a cumulative roll-up)."""
    with patch(
        "src.agent.proxy_client.requests.post",
        side_effect=[_status_response(503), _ok_response()],
    ), patch("src.agent.proxy_client.time.sleep"):
        client.post("/search/find_product")

    attempts = [e for e in _read_jsonl(log_path) if e["kind"] == "attempt"]
    assert len(attempts) == 2
    for a in attempts:
        assert isinstance(a["duration_ms"], (int, float))
        assert a["duration_ms"] >= 0


def test_get_logs_attempts_too(client, log_path):
    with patch("src.agent.proxy_client.requests.get", return_value=_ok_response()):
        client.get("/search/find_product", params={"q": "shoes"})

    attempts = [e for e in _read_jsonl(log_path) if e["kind"] == "attempt"]
    assert len(attempts) == 1
    assert attempts[0]["method"] == "GET"
    assert attempts[0]["path"] == "/search/find_product"


def test_summary_entry_still_written(client, log_path):
    """Existing per-call summary entry is preserved (judge consumes it)."""
    with patch("src.agent.proxy_client.requests.post", return_value=_ok_response()):
        client.post("/search/find_product")

    summaries = [e for e in _read_jsonl(log_path) if e["kind"] == "summary"]
    assert len(summaries) == 1
    assert summaries[0]["path"] == "/search/find_product"
    assert summaries[0]["status_code"] == 200


def test_request_log_disabled_when_no_file(monkeypatch):
    """No env var -> attempts and summaries are silently skipped."""
    monkeypatch.delenv("REQUEST_LOG_FILE", raising=False)
    rl = RequestLog(None)
    rl.record_attempt("POST", "/inference/chat", 0, 100.0, status_code=200)
    rl.record("POST", "/inference/chat", duration_ms=100.0, status_code=200)
    # No assertions on file content — call must just not raise.
