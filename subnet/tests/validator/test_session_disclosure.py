"""Hostile disclosure checks for the miner-visible environment contract."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from subnet.tests.world_pack_fixture import ANSWERS, StubModelClient, StubSearchClient, load_world_pack
from validator.session_registry import SessionRegistry
from validator.session_service import SessionRuntime, create_session_app


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for nested in value.values() for key in _all_keys(nested)}
    if isinstance(value, list):
        return {key for nested in value for key in _all_keys(nested)}
    return set()


@pytest.fixture
def served(tmp_path):
    pack = load_world_pack(tmp_path)
    registry = SessionRegistry(pack, episode_nonces={"t1": "private-nonce", "t2": "n2"},
                               model_client=StubModelClient(), search_client=StubSearchClient())
    runtime = SessionRuntime()
    runtime.install(registry)
    yield registry, TestClient(create_session_app(runtime)), pack
    runtime.clear()


def test_bootstrap_and_step_results_are_explicit_public_projections(served) -> None:
    registry, client, pack = served
    bootstrap = registry.start("t1", session_id="public-session-token")

    assert set(bootstrap) == {"session_id", "binding", "policy_view"}
    assert set(bootstrap["policy_view"]) == {"query", "opening", "tools", "max_steps", "finished"}
    assert "handler" not in _all_keys(bootstrap)
    text = json.dumps(bootstrap)
    assert "private-nonce" not in text and pack.pack_id not in text

    responses = [
        client.post("/v1/session/call", json={**bootstrap["binding"], "step": 1,
                                              "action": {"name": "ask", "args": {"text": "hi"}}}),
        client.post("/v1/session/call", json={**bootstrap["binding"], "step": 2,
                                              "action": {"name": "buy", "args": {"id": "b"}}}),
    ]
    for response in responses:
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"observation", "finished", "step", "replayed"}
        assert not {"ledger", "verdict", "answer", "reading", "episode"} & _all_keys(body)
        assert "private-nonce" not in json.dumps(body)
    assert responses[1].json()["finished"] is True
    assert json.dumps(ANSWERS["t1"]) not in json.dumps([r.json() for r in responses])


def test_transport_errors_map_to_statuses_without_private_detail(served, monkeypatch) -> None:
    registry, client, _ = served
    binding = registry.start("t1")["binding"]

    def call(step, action):
        return client.post("/v1/session/call", json={**binding, "step": step, "action": action})

    assert client.post("/v1/session/call", json={"session_id": "forged", "step": 1,
                                                  "action": {"name": "find", "args": {}}}).status_code == 409
    assert call(0, {"name": "find", "args": {}}).status_code == 422
    assert call(5, {"name": "find", "args": {"query": "x"}}).status_code == 409

    state = registry._sessions[binding["session_id"]]
    monkeypatch.setattr(state.session, "step", lambda *_a: (_ for _ in ()).throw(RuntimeError("ORO_PRIVATE_CANARY")))
    failed = call(1, {"name": "find", "args": {"query": "x"}})
    assert failed.status_code == 500
    assert failed.json()["detail"]["environment_error"] is True
    assert "ORO_PRIVATE_CANARY" not in failed.text

    registry.key_exhausted.set()
    other = registry.start("t2")["binding"]
    exhausted = client.post("/v1/session/call", json={**other, "step": 1, "action": {"name": "find", "args": {"query": "x"}}})
    assert exhausted.status_code == 402
