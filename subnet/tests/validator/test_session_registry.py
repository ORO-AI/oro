"""World sessions in the validator: start, steps, idempotent replay, finish, grading and outside clients."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
import requests
from oro_env_runtime import models

from subnet.tests.world_pack_fixture import StubModelClient, StubSearchClient, load_world_pack
from validator import session_registry
from validator.session_registry import (
    VALIDATOR_CALLER_HEADER,
    VALIDATOR_CALLER_SECRET,
    AgentInferenceBudgetError,
    HarnessExecutionError,
    InferenceProviderError,
    InvalidSessionError,
    ProxyModelClient,
    SearchServerClient,
    SessionRegistry,
)

NONCES = {"t1": "nonce-1", "t2": "nonce-2"}


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    return load_world_pack(tmp_path_factory.mktemp("world"))


@pytest.fixture
def clients():
    return StubModelClient(), StubSearchClient()


@pytest.fixture
def registry(pack, clients):
    with SessionRegistry(pack, episode_nonces=NONCES, model_client=clients[0], search_client=clients[1]) as reg:
        yield reg


def _call(session_id: str, step: int, name: str, **args) -> dict:
    return {"session_id": session_id, "step": step, "action": {"name": name, "args": args}}


def test_start_returns_binding_and_an_openai_tool_policy_view(registry, pack) -> None:
    started = registry.start("t1", session_id="s1")

    assert started["session_id"] == "s1" and started["binding"] == {"session_id": "s1"}
    view = started["policy_view"]
    assert set(view) == {"query", "opening", "tools", "max_steps", "finished"}
    assert view["query"] == "Buy the item for t1."
    assert view["opening"] == {"greeting": "hello"}
    assert view["max_steps"] == pack.limits["max_steps"] and view["finished"] is False
    assert [t["type"] for t in view["tools"]] == ["function"] * 3
    assert [t["function"]["name"] for t in view["tools"]] == ["find", "ask", "buy"]
    assert set(view["tools"][0]["function"]) == {"name", "description", "parameters"}  # no handler ids


def test_start_rejects_unknown_tasks_missing_nonces_and_duplicate_sessions(pack, clients) -> None:
    with SessionRegistry(pack, episode_nonces={"t1": "n"}, model_client=clients[0], search_client=clients[1]) as reg:
        with pytest.raises(KeyError):
            reg.start("nope")
        with pytest.raises(ValueError, match="no episode nonce"):
            reg.start("t2")
        reg.start("t1", session_id="dup")
        with pytest.raises(InvalidSessionError):
            reg.start("t1", session_id="dup")


def test_indexes_are_uploaded_and_search_reaches_the_search_client(registry, pack, clients) -> None:
    index_id = pack.indexes["title"]["id"]
    assert index_id in clients[1].titles

    sid = registry.start("t1")["session_id"]
    result = registry.call(_call(sid, 1, "find", query="mug"))

    assert result == {"observation": {"ok": True, "ids": ["a", "b"]}, "finished": False, "step": 1, "replayed": False}
    assert clients[1].calls == [(index_id, "mug", {"k": 5})]


def test_a_correct_episode_is_graded_with_the_claim_nonce(registry, clients) -> None:
    sid = registry.start("t1")["session_id"]
    ask = registry.call(_call(sid, 1, "ask", text="which colour?"))
    buy = registry.call(_call(sid, 2, "buy", id="a"))

    assert ask["observation"] == {"reply": "I want the red one."}
    assert clients[0].calls[0][:2] == ("complete", "shopper-model")
    assert buy["finished"] is True and buy["step"] == 2
    with pytest.raises(InvalidSessionError, match="finished"):
        registry.call(_call(sid, 3, "find", query="mug"))

    [record] = registry.terminal_results()
    assert set(record) >= {"task_id", "session_id", "episode_nonce", "ledger", "verdict", "call_trace", "steps"}
    assert (record["task_id"], record["session_id"], record["episode_nonce"], record["steps"]) == ("t1", sid, "nonce-1", 2)
    assert record["verdict"]["score"] == 1.0 and record["verdict"]["category"] is None
    assert record["error"] is None
    episode = record["ledger"]["episode"]
    assert episode[0]["episode_nonce"] == "nonce-1" and episode[-1]["reason"] == "order"
    assert [(t["tool"], t["step"]) for t in record["call_trace"]] == [("ask", 1), ("buy", 2)]
    assert all(t["elapsed_ms"] >= 0 for t in record["call_trace"])
    assert registry.terminal_results() == [record]  # graded once, then cached


def test_replay_by_step_is_idempotent_and_bound_to_the_action(registry) -> None:
    sid = registry.start("t1")["session_id"]
    first = registry.call(_call(sid, 1, "find", query="red"))
    again = registry.call(_call(sid, 1, "find", query="red"))

    assert again == {**first, "replayed": True}
    with pytest.raises(InvalidSessionError, match="different action"):
        registry.call(_call(sid, 1, "find", query="blue"))
    with pytest.raises(InvalidSessionError, match="expected 2, got 3"):
        registry.call(_call(sid, 3, "find", query="blue"))
    assert registry.call(_call(sid, 2, "find", query="blue"))["step"] == 2


def test_grouped_actions_run_as_consecutive_steps_and_stop_at_finish(registry) -> None:
    sid = registry.start("t2")["session_id"]
    envelope = {"session_id": sid, "step": 1, "actions": [
        {"name": "find", "args": {"query": "blue"}},
        {"name": "buy", "args": {"id": "b"}},
        {"name": "find", "args": {"query": "red"}},  # never runs: the episode finished
    ]}
    result = registry.call(envelope)

    assert result["step"] == 2 and result["finished"] is True
    assert result["observations"] == [{"ok": True, "ids": ["b"]}, {"submitted": "b"}]
    assert result["observation"] == {"submitted": "b"}
    assert registry.call(envelope)["replayed"] is True
    [record] = registry.terminal_results()
    assert record["verdict"]["score"] == 1.0


def test_finish_all_stops_unfinished_episodes_and_seals_the_registry(registry) -> None:
    sid = registry.start("t1")["session_id"]
    idle = registry.start("t2")["session_id"]
    registry.call(_call(sid, 1, "find", query="mug"))
    assert registry.terminal_results() == []

    registry.finish_all()

    records = {r["session_id"]: r for r in registry.terminal_results()}
    assert set(records) == {sid, idle}
    for record in records.values():
        assert record["ledger"]["episode"][-1]["reason"] == "stopped"
        assert record["verdict"]["category"] == "did not finish"
    with pytest.raises(InvalidSessionError, match="finalized"):
        registry.call(_call(sid, 2, "find", query="mug"))
    with pytest.raises(RuntimeError):
        registry.start("t1")


def test_envelope_validation(registry) -> None:
    sid = registry.start("t1")["session_id"]
    with pytest.raises(InvalidSessionError, match="unknown session_id"):
        registry.call(_call("forged", 1, "find", query="x"))
    with pytest.raises(InvalidSessionError, match="task_id"):
        registry.call({**_call(sid, 1, "find", query="x"), "task_id": "t2"})
    with pytest.raises(ValueError, match="step"):
        registry.call(_call(sid, 0, "find", query="x"))
    with pytest.raises(ValueError, match="exactly one"):
        registry.call({"session_id": sid, "step": 1})
    with pytest.raises(ValueError, match="actions must be"):
        registry.call({"session_id": sid, "step": 1, "actions": []})
    # Tool-level mistakes are observations, not transport errors.
    bad = registry.call(_call(sid, 1, "find", query=3))
    assert bad["observation"]["error"] == "invalid_arguments"
    assert registry.call(_call(sid, 2, "nope"))["observation"] == {"error": "unknown_tool"}


def test_a_runtime_failure_quarantines_the_session_without_a_verdict(registry, monkeypatch) -> None:
    sid = registry.start("t1")["session_id"]
    state = registry._sessions[sid]
    monkeypatch.setattr(state.session, "step", lambda *_a: (_ for _ in ()).throw(OSError("worker died")))

    with pytest.raises(HarnessExecutionError, match="quarantined"):
        registry.call(_call(sid, 1, "find", query="x"))
    with pytest.raises(InvalidSessionError, match="quarantined"):
        registry.call(_call(sid, 1, "find", query="x"))

    registry.finish_all()
    [record] = registry.terminal_results()
    assert record["verdict"] is None and record["error"] == "runtime step failed: OSError"
    assert "worker died" not in json.dumps(record)


def test_an_exhausted_inference_key_stops_calls(registry) -> None:
    sid = registry.start("t1")["session_id"]
    registry.key_exhausted.set()
    with pytest.raises(AgentInferenceBudgetError):
        registry.call(_call(sid, 1, "find", query="x"))


def test_the_default_model_client_is_the_module_proxy_client(pack, monkeypatch, clients) -> None:
    built = []

    class Recording(StubModelClient):
        def __init__(self, token, **kwargs):
            super().__init__()
            built.append((token, kwargs))

    monkeypatch.setattr(session_registry, "ProxyModelClient", Recording)
    with SessionRegistry(pack, episode_nonces=NONCES, search_client=clients[1], inference_access_token="sk-or-run",
                         proxy_url="http://proxy.test", timeout=12.0) as reg:
        assert built[0][0] == "sk-or-run"
        assert built[0][1]["proxy_url"] == "http://proxy.test" and built[0][1]["key_exhausted_event"] is reg.key_exhausted
    with pytest.raises(ValueError):
        SessionRegistry(pack, episode_nonces=NONCES, search_client=clients[1])


def test_close_stops_every_sandbox_worker(pack, clients) -> None:
    reg = SessionRegistry(pack, episode_nonces=NONCES, model_client=clients[0], search_client=clients[1])
    sid = reg.start("t1")["session_id"]
    proc = reg._sessions[sid].session.box.proc
    reg.close()
    assert proc.poll() is not None
    with pytest.raises(InvalidSessionError):
        reg.call(_call(sid, 1, "find", query="x"))


# ---------------------------------------------------------------- SearchServerClient


class _Response:
    text = ""

    def __init__(self, status: int, body: dict | None = None) -> None:
        self.status_code, self._body = status, body or {}

    def json(self) -> dict:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


@pytest.fixture
def search_http(monkeypatch):
    log: list[tuple] = []
    search_statuses: list[int] = []

    def put(url, data, headers, timeout):
        log.append(("PUT", url, data.read(), headers))
        return _Response(201)

    def post(url, json, timeout):
        log.append(("POST", url, json))
        status = search_statuses.pop(0) if search_statuses else 200
        return _Response(status, {"hits": [{"id": "a", "score": 2.5}]})

    monkeypatch.setattr(session_registry.requests, "put", put)
    monkeypatch.setattr(session_registry.requests, "post", post)
    return log, search_statuses


def _index(tmp_path: Path) -> dict:
    archive = tmp_path / "title.archive"
    archive.write_bytes(b"index-bytes")
    return {"id": "f" * 64, "archive": str(archive), "settings": {"bm25": {"k1": 1.2, "b": 0.5}}}


def test_search_client_uploads_with_settings_and_posts_queries(tmp_path, search_http) -> None:
    log, _ = search_http
    client = SearchServerClient("http://search.test/")
    client.ensure(_index(tmp_path))
    hits = client.search("f" * 64, "red mug", {"k": 3, "all_terms": True}, 5.0)

    assert hits == [("a", 2.5)]
    method, url, body, headers = log[0]
    assert (method, url, body) == ("PUT", f"http://search.test/internal/worlds/{'f' * 64}", b"index-bytes")
    assert json.loads(headers["X-ORO-World-Settings"]) == {"bm25": {"k1": 1.2, "b": 0.5}}
    assert log[1] == ("POST", f"http://search.test/internal/worlds/{'f' * 64}/search",
                      {"query": "red mug", "params": {"k": 3, "all_terms": True}})


def test_search_client_reuploads_once_after_a_404(tmp_path, search_http) -> None:
    log, statuses = search_http
    client = SearchServerClient("http://search.test")
    client.ensure(_index(tmp_path))
    statuses.extend([404])
    assert client.search("f" * 64, "q", {}, 5.0) == [("a", 2.5)]
    assert [entry[0] for entry in log] == ["PUT", "POST", "PUT", "POST"]

    statuses.extend([404, 404])
    with pytest.raises(requests.HTTPError):
        client.search("f" * 64, "q", {}, 5.0)
    assert [entry[0] for entry in log[4:]] == ["POST", "PUT", "POST"]


# ---------------------------------------------------------------- ProxyModelClient


@pytest.fixture
def proxy_http(monkeypatch):
    sent: list[tuple] = []
    replies: list[_Response] = []

    def post(url, json, headers, timeout):
        sent.append((url, json, headers))
        return replies.pop(0)

    import src.agent.proxy_client as proxy_client

    monkeypatch.setattr(proxy_client.requests, "post", post)
    return sent, replies


def _ok(body: dict) -> _Response:
    return _Response(200, body)


def _fail(status: int, text: str) -> _Response:
    response = _Response(status)
    response.text = text
    return response


def _model_client(tmp_path, **kwargs) -> ProxyModelClient:
    return ProxyModelClient("sk-or-secret", proxy_url="http://proxy.test", inference_stats_file=str(tmp_path / "s.jsonl"), **kwargs)


def test_complete_uses_the_chat_route_with_the_validator_header(tmp_path, proxy_http) -> None:
    sent, replies = proxy_http
    replies.append(_ok({"choices": [{"message": {"content": "yes sk-or-secret"}}]}))

    text = _model_client(tmp_path).complete("m", "Hello", {"temperature": 0}, 5.0)

    assert text == "yes [REDACTED]"
    url, body, headers = sent[0]
    assert url == "http://proxy.test/inference/chat/completions"
    assert body == {"temperature": 0, "model": "m", "messages": [{"role": "user", "content": "Hello"}], "stream": False}
    assert headers[VALIDATOR_CALLER_HEADER] == VALIDATOR_CALLER_SECRET
    assert headers["Authorization"] == "Bearer sk-or-secret"


def test_decide_forwards_dict_questions_unchanged(tmp_path, proxy_http) -> None:
    sent, replies = proxy_http
    questions = {"q1": {"type": "noul", "text": "Did they confirm?", "lo": 0.3, "hi": 0.7}}
    replies.append(_ok({"answers": {"q1": 0.9}}))

    answers = _model_client(tmp_path).decide("typesafe/jev-1.13", {"messages": ["hi"]}, questions, {}, 5.0)

    assert answers == {"answers": {"q1": 0.9}}
    url, body, headers = sent[0]
    assert url == "http://proxy.test/inference/alpha/decisions"
    assert body == {"model": "typesafe/jev-1.13", "state": {"messages": ["hi"]}, "questions": questions}
    assert body["questions"] is questions
    assert headers[VALIDATOR_CALLER_HEADER] == VALIDATOR_CALLER_SECRET


def test_failures_raise_redacted_and_become_error_readings(tmp_path, proxy_http) -> None:
    _, replies = proxy_http
    exhausted = threading.Event()
    client = _model_client(tmp_path, key_exhausted_event=exhausted)

    replies.append(_fail(500, "boom sk-or-secret"))
    with pytest.raises(InferenceProviderError) as caught:
        client.complete("m", "p", {}, 5.0)
    assert "sk-or-secret" not in str(caught.value) and not exhausted.is_set()

    replies.append(_fail(402, "Insufficient credits"))
    with pytest.raises(InferenceProviderError):
        client.decide("m", {}, {}, {}, 5.0)
    assert exhausted.is_set()

    replies.append(_ok({"choices": []}))
    read = models.live_reader(client, StubSearchClient(), 5.0, 0)
    assert read("complete", {"model": "m", "prompt": "p", "params": {}}) == {"ok": False, "error": "error"}
