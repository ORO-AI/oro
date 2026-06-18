"""Integration tests for the proxy nonce mint + verify protocol (ORO-1372).

These tests stand up the sandbox proxy plus a dummy LLM endpoint and a dummy
search-server to exercise the full mint -> dispatch flow end-to-end. The
proxy is the canonical authority for "did the LLM emit this tool_call?"; we
verify that:

1.  mint:        /inference/chat/completions response is enriched with
                 oro_metadata.tool_nonces keyed by call_id, and
                 oro_metadata.parsed_tool_calls carries the tool_name +
                 args_hash for each call.
2.  valid:       a /search/<tool> dispatch with the minted nonce + matching
                 body returns 200 and X-Nonce-Status: valid.
3.  missing:     dispatch with no X-Tool-Nonce returns 200 (Phase 0-2
                 default = informational) with X-Nonce-Status: missing.
4.  mismatch:    dispatch with the right nonce but a different body returns
                 X-Nonce-Status: mismatch.
5.  replay:      a second dispatch with the same nonce returns
                 X-Nonce-Status: replayed.
6.  expired:     dispatch after the 60s mint TTL returns
                 X-Nonce-Status: expired.

These tests require `docker compose up -d proxy dummy-llm dummy-search` with
shared $ORO_PROXY_HMAC_KEY and $ORO_EVAL_RUN_ID env vars wired in. We mark
the whole module @pytest.mark.docker so the default pytest run skips them;
the implementer (Task 11) runs them manually on staging once the supporting
fixtures are wired into docker-compose.yml.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import pytest

pytestmark = [
    pytest.mark.docker,
    pytest.mark.skip(
        reason=(
            "Requires docker compose orchestration (proxy + dummy-llm + "
            "dummy-search). Run manually with `docker compose up -d` and "
            "`pytest -m docker tests/proxy/`."
        )
    ),
]


PROXY_URL = os.environ.get("ORO_PROXY_URL", "http://localhost:8080")
EVAL_RUN_ID = os.environ.get("ORO_EVAL_RUN_ID", "test-run-1")


@pytest.fixture
def http():
    import httpx

    with httpx.Client(base_url=PROXY_URL, timeout=10.0) as c:
        yield c


def _mint_tool_call(
    http,
    *,
    tool_name: str = "find_product",
    arguments: str = '{"query":"red shoes"}',
    call_id: str = "call_abc",
) -> dict[str, Any]:
    """POST a fixture chat-completions request that the dummy LLM echoes as
    a single tool_call. The proxy enriches the response with oro_metadata.
    Returns the parsed response body."""

    fixture_response = {
        "id": "chatcmpl-test",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": arguments,
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }
    # Dummy LLM is configured to echo X-Test-Response back as the body.
    resp = http.post(
        "/inference/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        headers={
            "Authorization": "Bearer cak_test",
            "X-Test-Response": json.dumps(fixture_response),
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_mint_injects_nonces_into_response(http) -> None:
    body = _mint_tool_call(http, call_id="call_abc")
    meta = body.get("oro_metadata")
    assert meta is not None, body
    nonces = meta.get("tool_nonces")
    parsed = meta.get("parsed_tool_calls")
    # Native tool_calls are namespaced with `native_` prefix (Fix 3, ORO-1372
    # review) so that they cannot collide with `xml_<i>_<j>` synthetic ids.
    assert isinstance(nonces, dict) and "native_call_abc" in nonces
    assert isinstance(parsed, list) and len(parsed) == 1
    assert parsed[0]["call_id"] == "native_call_abc"
    assert parsed[0]["tool_name"] == "find_product"
    assert "args_hash" in parsed[0]


def test_valid_dispatch_returns_valid_status(http) -> None:
    body = _mint_tool_call(http)
    nonce = body["oro_metadata"]["tool_nonces"]["native_call_abc"]

    resp = http.post(
        "/search/find_product",
        content='{"query":"red shoes"}',
        headers={
            "Content-Type": "application/json",
            "X-Tool-Nonce": nonce,
            "X-Tool-Call-Id": "native_call_abc",
        },
    )
    assert resp.headers.get("X-Nonce-Status") == "valid"
    assert resp.status_code == 200


def test_missing_nonce_returns_missing_status(http) -> None:
    resp = http.post(
        "/search/find_product",
        content='{"query":"red shoes"}',
        headers={"Content-Type": "application/json"},
    )
    # Phase 0-2 default: still forwards (informational).
    assert resp.headers.get("X-Nonce-Status") == "missing"


def test_mismatch_body_returns_mismatch_status(http) -> None:
    body = _mint_tool_call(http)
    nonce = body["oro_metadata"]["tool_nonces"]["native_call_abc"]

    resp = http.post(
        "/search/find_product",
        content='{"query":"different body"}',
        headers={
            "Content-Type": "application/json",
            "X-Tool-Nonce": nonce,
            "X-Tool-Call-Id": "native_call_abc",
        },
    )
    assert resp.headers.get("X-Nonce-Status") == "mismatch"


def test_replay_returns_replayed_status(http) -> None:
    body = _mint_tool_call(http, call_id="call_replay")
    nonce = body["oro_metadata"]["tool_nonces"]["native_call_replay"]
    payload = '{"query":"red shoes"}'

    r1 = http.post(
        "/search/find_product",
        content=payload,
        headers={
            "Content-Type": "application/json",
            "X-Tool-Nonce": nonce,
            "X-Tool-Call-Id": "native_call_replay",
        },
    )
    assert r1.headers.get("X-Nonce-Status") == "valid"

    r2 = http.post(
        "/search/find_product",
        content=payload,
        headers={
            "Content-Type": "application/json",
            "X-Tool-Nonce": nonce,
            "X-Tool-Call-Id": "native_call_replay",
        },
    )
    assert r2.headers.get("X-Nonce-Status") == "replayed"


def test_expired_nonce_returns_expired_status(http) -> None:
    body = _mint_tool_call(http, call_id="call_expire")
    nonce = body["oro_metadata"]["tool_nonces"]["native_call_expire"]

    # Requires ORO_PROXY_NONCE_TTL_MS=200 set on the proxy container so the
    # TTL is 200ms instead of 60s. Sleep 0.3s to ensure expiry without the
    # test taking over a minute.
    time.sleep(0.3)

    resp = http.post(
        "/search/find_product",
        content='{"query":"red shoes"}',
        headers={
            "Content-Type": "application/json",
            "X-Tool-Nonce": nonce,
            "X-Tool-Call-Id": "call_expire",
        },
    )
    assert resp.headers.get("X-Nonce-Status") == "expired"


def test_strict_mode_returns_403_on_missing(http) -> None:
    """When ORO_PROXY_NONCE_STRICT=true is set on the proxy container, any
    non-valid status returns 403 instead of forwarding. This test variant
    requires the proxy be restarted with strict mode on."""
    resp = http.post(
        "/search/find_product",
        content='{"query":"red shoes"}',
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 403
    assert resp.headers.get("X-Nonce-Status") == "missing"


def test_xml_tool_call_in_content_minted(http) -> None:
    """The legacy default agent emits tool_calls as <tool_call>...</tool_call>
    blocks inside message.content (not as native tool_calls[]). The proxy
    must mint nonces for those too, keyed by synthesized xml_N call_ids."""
    xml = (
        '<tool_call>{"name":"find_product","arguments":'
        '{"query":"red shoes"}}</tool_call>'
    )
    fixture_response = {
        "id": "chatcmpl-xml",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": xml},
                "finish_reason": "stop",
            }
        ],
    }
    resp = http.post(
        "/inference/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        headers={
            "Authorization": "Bearer cak_test",
            "X-Test-Response": json.dumps(fixture_response),
        },
    )
    body = resp.json()
    nonces = body.get("oro_metadata", {}).get("tool_nonces", {})
    # XML blocks synthesize call_id = "xml_<choiceIdx>_<blockIdx>".
    assert "xml_0_0" in nonces
