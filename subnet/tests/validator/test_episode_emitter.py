"""Unit coverage for the episode-result emitter."""

from __future__ import annotations

import asyncio
import copy
import json
import os
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from oro_env_runtime import canonical, models, session
from validator import episode_emitter
from validator.env_backend import environment_client
from validator.environment_preflight import (
    POLICY_ACTIONS,
    _StubModels,
    _StubSearch,
    build_preflight_pack,
)
from validator.episode_emitter import (
    EpisodeEmitError,
    ReplayMismatch,
    build_episode_artifact,
    emit_finalized_results,
    episode_artifact,
    episode_artifact_sha256,
    episode_payload,
    load_episode_artifact,
    load_inference_transcripts,
    regrade,
    replay_matches,
    serialize_episode_artifact,
    submit_episode_batch,
)
from validator.generated_progress_reporter import GeneratedProgressReporter


def _run_async(test: Callable[..., Any]) -> Callable[..., None]:
    """Run an async test without a pytest-asyncio dependency."""

    @wraps(test)
    def wrapper(*args: object, **kwargs: object) -> None:
        asyncio.run(test(*args, **kwargs))

    return wrapper


# ---------------------------------------------------------------------------
# Fixtures: real terminal records from the bundled preflight world
# ---------------------------------------------------------------------------


_PACK_SHA = "a" * 64
_EVAL_RUN_ID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
_DELIVERY = "validator-preflight"
_NONCE = "c" * 64
_FAMILIES = {task_id: "preflight" for task_id in POLICY_ACTIONS}


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory):
    pack = build_preflight_pack(tmp_path_factory.mktemp("world"))
    read = models.live_reader(_StubModels(), _StubSearch(), 5.0, 0)
    records = {}
    for task_id, actions in POLICY_ACTIONS.items():
        episode = session.Session(pack, task_id, _NONCE, read)
        for action in actions:
            episode.step(action["name"], action["args"])
        ledger = episode.ledger()
        verdict = session.grade_episode(pack, ledger, read)
        records[task_id] = {
            "task_id": task_id,
            "session_id": f"sess-{task_id}",
            "episode_nonce": _NONCE,
            "ledger": ledger,
            "verdict": verdict,
            "call_trace": [{"step": 1, "latency_ms": 1.25}],
            "steps": episode.steps,
        }
    return pack, records


def _record(world, task_id: str = "positive") -> dict[str, Any]:
    return copy.deepcopy(world[1][task_id])


def test_episode_artifact_is_content_addressed_and_round_trips(world):
    record = _record(world)
    artifact = build_episode_artifact(
        record, eval_run_id=_EVAL_RUN_ID, family="preflight", env_pack_sha256=_PACK_SHA
    )
    body = serialize_episode_artifact(artifact)
    digest = episode_artifact_sha256(body)

    assert load_episode_artifact(body, expected_sha256=digest) == artifact
    assert artifact["episode"]["ledger"] == record["ledger"]
    assert artifact["episode"]["verdict"] == record["verdict"]
    assert artifact["episode"]["call_trace"] == record["call_trace"]
    assert artifact["episode"]["evaluation_run_id"] == _EVAL_RUN_ID
    assert "inference_transcript" not in artifact
    with pytest.raises(EpisodeEmitError, match="sha256 mismatch"):
        load_episode_artifact(body, expected_sha256="0" * 64)


def test_episode_artifact_requires_ledger_verdict_and_trace(world):
    record = _record(world)
    assert set(episode_artifact(record)) == {"ledger", "verdict", "call_trace"}
    del record["call_trace"]
    with pytest.raises(EpisodeEmitError, match="call_trace"):
        episode_artifact(record)


def test_inference_transcript_preserves_reasoning_and_excludes_secrets(tmp_path):
    scoped_token = "scoped-secret-token"
    call = {
        "kind": "summary",
        "path": "/inference/chat",
        "timestamp": 2,
        "json_data": {
            "model": "requested/model",
            "messages": [{"role": "user", "content": "full prompt"}],
            "Authorization": "Bearer secret",
            "apiKey": "also-secret",
            "nested": {
                "openrouter_api_key": "also-secret",
                "cookie": "session=secret",
            },
            "note": f"do not retain {scoped_token}",
        },
        "response": {
            "model": "served/model",
            "reasoning": "full reasoning",
            "usage": {"prompt_tokens": 4, "completion_tokens": 7},
            "url": "https://example.test/search?q=boots&signature=secret",
        },
    }
    output = tmp_path / "output.jsonl"
    output.write_text(
        json.dumps(
            {
                "problem_id": "sess-1",
                "status": "SUCCESS",
                "dialogue": [
                    {
                        "role": "assistant",
                        "content": "final answer",
                        "extra_info": {"proxy_calls": [call]},
                    }
                ],
                "_shadow_proxy_calls": [call],
                "_shadow_inference_usage": {"cached_tokens": 3},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    transcript = load_inference_transcripts(output, secret_values=(scoped_token,))[
        "sess-1"
    ]

    assert transcript["calls"][0]["response"]["reasoning"] == "full reasoning"
    assert (
        transcript["calls"][0]["response"]["url"]
        == "https://example.test/search?q=boots&signature=[REDACTED]"
    )
    assert "Authorization" not in transcript["calls"][0]["json_data"]
    assert "apiKey" not in transcript["calls"][0]["json_data"]
    assert transcript["calls"][0]["json_data"]["nested"] == {}
    assert scoped_token not in transcript["calls"][0]["json_data"]["note"]
    assert "proxy_calls" not in transcript["agent_output"]["dialogue"][0]["extra_info"]
    assert "_shadow_inference_usage" not in transcript["agent_output"]


@pytest.mark.parametrize(
    "key",
    [
        "apikey",
        "APIKey",
        "IDToken",
        "id_token",
        "private_key",
        "privateKey",
        "secret_key",
        "aws_secret_access_key",
    ],
)
def test_inference_transcript_excludes_more_secret_key_spellings(tmp_path, key):
    call = {
        "kind": "summary",
        "path": "/inference/chat",
        "timestamp": 1,
        "json_data": {"model": "m", key: "secret-value"},
        "response": {"url": f"https://example.test/v1?q=boots&{key}=secret-value"},
    }
    output = tmp_path / "output.jsonl"
    output.write_text(
        json.dumps(
            {"problem_id": "sess-1", "status": "SUCCESS", "_shadow_proxy_calls": [call]}
        )
        + "\n",
        encoding="utf-8",
    )

    recorded = load_inference_transcripts(output)["sess-1"]["calls"][0]

    assert recorded["json_data"] == {"model": "m"}
    assert recorded["response"]["url"] == (
        f"https://example.test/v1?q=boots&{key}=[REDACTED]"
    )


@pytest.mark.parametrize(
    "key", ["max_tokens", "prompt_tokens", "eos_token", "page_token", "access_key_id"]
)
def test_inference_transcript_keeps_non_secret_key_names(tmp_path, key):
    call = {
        "kind": "summary",
        "path": "/inference/chat",
        "timestamp": 1,
        "json_data": {key: "kept"},
    }
    output = tmp_path / "output.jsonl"
    output.write_text(
        json.dumps(
            {"problem_id": "sess-1", "status": "SUCCESS", "_shadow_proxy_calls": [call]}
        )
        + "\n",
        encoding="utf-8",
    )

    recorded = load_inference_transcripts(output)["sess-1"]["calls"][0]

    assert recorded["json_data"] == {key: "kept"}


def test_transcript_loader_ignores_malformed_and_duplicate_output(tmp_path, caplog):
    output = tmp_path / "output.jsonl"
    output.write_text(
        "not json\n"
        + json.dumps({"status": "missing id"})
        + "\n"
        + json.dumps({"problem_id": "sess-1", "status": "FIRST"})
        + "\n"
        + json.dumps({"problem_id": "sess-1", "status": "FORGED"})
        + "\n",
        encoding="utf-8",
    )

    transcripts = load_inference_transcripts(output)

    assert transcripts["sess-1"]["final_status"] == "FIRST"
    assert "malformed sandbox output" in caplog.text
    assert "duplicate sandbox output" in caplog.text


def test_transcript_loader_drops_only_the_lines_it_cannot_serialize(tmp_path):
    lines = [
        '{"problem_id": "non-finite", "dialogue": [{"was": Infinity, "gap": NaN}]}',
        '{"problem_id": "long-int", "n": %s}' % ("9" * 5000),
        '{"problem_id": "deep", "dialogue": %s}' % ("[" * 600 + "]" * 600),
        '{"problem_id": "deeper", "dialogue": %s}' % ("[" * 5000 + "]" * 5000),
        json.dumps(
            {
                "problem_id": "mixed-timestamps",
                "_shadow_proxy_calls": [
                    {"path": "/inference/a", "timestamp": "late"},
                    {"path": "/inference/b", "timestamp": 1},
                ],
            }
        ),
        '{"problem_id": "ok"}',
    ]
    output = tmp_path / "output.jsonl"
    output.write_bytes("\n".join(lines).encode() + b"\n\xff\n")

    transcripts = load_inference_transcripts(output)

    assert set(transcripts) == {"non-finite", "ok"}
    dialogue = transcripts["non-finite"]["agent_output"]["dialogue"]
    assert dialogue == [{"was": None, "gap": None}]
    serialize_episode_artifact({"inference_transcript": transcripts["non-finite"]})




# ---------------------------------------------------------------------------
# episode_payload
# ---------------------------------------------------------------------------


def test_payload_carries_verdict_digests_and_nonce(world):
    record = _record(world)
    payload = episode_payload(record, delivery_id=_DELIVERY)

    assert payload == {
        "task_id": "positive",
        "outcome": "completed",
        "score": "1.000000",
        "category": None,
        "flags": [],
        "ledger_sha256": canonical.digest(record["ledger"]["episode"]),
        "verdict_sha256": canonical.digest(record["verdict"]),
        "episode_nonce": _NONCE,
    }
    assert payload["ledger_sha256"] == record["verdict"]["ledger_sha256"]


def test_payload_passes_failure_category_through(world):
    payload = episode_payload(_record(world, "negative"), delivery_id=_DELIVERY)
    assert (payload["score"], payload["category"]) == ("0.000000", "request not met")


def test_payload_score_has_at_most_six_decimals(world):
    record = _record(world)
    record["verdict"]["score"] = 1 / 3
    assert episode_payload(record, delivery_id=_DELIVERY)["score"] == "0.333333"


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda r: r.update(episode_nonce="d" * 64), "another episode nonce"),
        (lambda r: r.update(task_id="negative"), "does not open task"),
        (lambda r: r["ledger"]["episode"].append({"kind": "event"}), "does not grade its ledger"),
    ],
)
def test_payload_rejects_unbound_records(world, mutate, match):
    record = _record(world)
    mutate(record)
    with pytest.raises(EpisodeEmitError, match=match):
        episode_payload(record, delivery_id=_DELIVERY)


def test_payload_rejects_another_delivery(world):
    with pytest.raises(EpisodeEmitError, match="outside delivery"):
        episode_payload(_record(world), delivery_id="race-other")


# ---------------------------------------------------------------------------
# submit_episode_batch / emit_finalized_results
# ---------------------------------------------------------------------------


class _MockTransport(httpx.MockTransport):
    """Records requests + returns pre-canned responses."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []
        self._responses = list(responses)

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return self._responses.pop(0)

        super().__init__(handler)


def _keypair():
    from bittensor_wallet import Keypair

    return Keypair.create_from_uri("//TestValidator")


def _entry(task_id="t1"):
    return {
        "eval_run_id": _EVAL_RUN_ID,
        "env_pack_sha256": _PACK_SHA,
        "task_id": task_id,
        "family": "preflight",
        "outcome": "completed",
        "score": "1.000000",
        "category": None,
        "flags": [],
        "ledger_sha256": "e" * 64,
        "verdict_sha256": "f" * 64,
        "episode_nonce": _NONCE,
    }


@_run_async
async def test_submit_single_batch_forwards_body_and_returns_merged():
    resp_body = {
        "results": [
            {"eval_run_id": _EVAL_RUN_ID, "task_id": "t1", "status": 201, "episode_result_id": None},
        ],
        "counts": {"201": 1},
    }
    transport = _MockTransport([httpx.Response(200, json=resp_body)])
    async with environment_client("https://api.example", _keypair(), transport=transport) as client:
        out = await submit_episode_batch(
            client,
            entries=[_entry()],
        )
    assert out["counts"] == {"201": 1}
    assert len(transport.requests) == 1
    body = json.loads(transport.requests[0].content)
    assert body == {"results": [_entry()]}


@_run_async
async def test_submit_batch_splits_over_backend_cap():
    """Backend caps at 500; over-cap input must chunk without silent drop."""
    def receipt(start, stop):
        return {"results": [
            {"eval_run_id": _EVAL_RUN_ID, "task_id": f"t{i}", "status": 201}
            for i in range(start, stop)
        ], "counts": {"201": stop - start}}
    transport = _MockTransport(
        [httpx.Response(200, json=receipt(0, 500)), httpx.Response(200, json=receipt(500, 501))]
    )
    async with environment_client("https://api.example", _keypair(), transport=transport) as client:
        entries = [_entry(f"t{i}") for i in range(501)]
        out = await submit_episode_batch(
            client,
            entries=entries,
        )
    assert len(transport.requests) == 2
    assert out["counts"] == {"201": 501}


@_run_async
async def test_submit_batch_raises_on_4xx_permanent():
    """A 422/400 is not a retry — surface as a hard EpisodeEmitError with body."""
    transport = _MockTransport([httpx.Response(422, text="bad schema field")])
    async with environment_client("https://api.example", _keypair(), transport=transport) as client:
        with pytest.raises(EpisodeEmitError, match="rejected status=422"):
            await submit_episode_batch(
                client,
                entries=[_entry()],
            )


@_run_async
async def test_submit_empty_batch_rejected():
    async with environment_client("https://api.example", _keypair()) as client:
        with pytest.raises(ValueError, match="empty batch"):
            await submit_episode_batch(
                client,
                entries=[],
            )


@_run_async
async def test_emit_retries_then_uploads_artifact_and_submits_summary(
    monkeypatch: pytest.MonkeyPatch, world
):
    record = _record(world)
    uploaded: bytes | None = None
    submitted: dict[str, Any] | None = None
    presign_request: dict[str, Any] | None = None
    presign_attempts = 0
    presign_nonces: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal uploaded, submitted, presign_request, presign_attempts
        if request.url.path.endswith("/episode-artifacts/presign"):
            presign_attempts += 1
            presign_nonces.append(request.headers["X-Nonce"])
            if presign_attempts == 1:
                return httpx.Response(503)
            presign_request = json.loads(request.content)
            digest = presign_request["artifact_sha256"]
            return httpx.Response(
                200,
                json={
                    "upload_url": "https://objects.test/upload",
                    "artifact_uri": f"s3://episodes/{digest}.json",
                    "artifact_sha256": digest,
                },
            )
        if request.url.host == "objects.test":
            assert not any(header.lower().startswith("x-") for header in request.headers)
            uploaded = request.content
            return httpx.Response(200)
        submitted = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "results": [{"eval_run_id": _EVAL_RUN_ID, "task_id": "positive", "status": 201}],
                "counts": {"201": 1},
            },
        )

    sleep = AsyncMock()
    monkeypatch.setattr(episode_emitter.asyncio, "sleep", sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transcript = {
            "schema_version": "oro.internal_inference_transcript.v1",
            "session_id": "sess-positive",
            "calls": [{"response": {"reasoning": "retained"}}],
        }
        result = await emit_finalized_results(
            backend_url="https://api.example",
            validator_keypair=_keypair(),
            env_pack_sha256=_PACK_SHA,
            eval_run_id=_EVAL_RUN_ID,
            delivery_id=_DELIVERY,
            task_families=_FAMILIES,
            results=[record],
            inference_transcripts={"sess-positive": transcript},
            http_client=client,
            backend_transport=client._transport,
        )

    assert result["counts"] == {"201": 1}
    assert presign_attempts == 2
    assert len(set(presign_nonces)) == 2
    sleep.assert_awaited_once_with(1)
    assert uploaded is not None and presign_request is not None
    artifact = load_episode_artifact(
        uploaded,
        expected_sha256=presign_request["artifact_sha256"],
    )
    assert artifact["episode"]["ledger"] == record["ledger"]
    assert artifact["inference_transcript"] == transcript
    assert presign_request["eval_run_id"] == _EVAL_RUN_ID
    assert submitted is not None
    entry = submitted["results"][0]
    assert entry["ledger_uri"].startswith("s3://episodes/")
    assert entry == {
        "eval_run_id": _EVAL_RUN_ID,
        "env_pack_sha256": _PACK_SHA,
        "family": "preflight",
        "outcome": "completed",
        "ledger_uri": entry["ledger_uri"],
        "step_count": 2,
        **episode_payload(record, delivery_id=_DELIVERY),
    }


@_run_async
async def test_unscored_episodes_are_infrastructure_rows_that_never_block_the_batch(world):
    scored = _record(world)
    environment = {**_record(world, "negative"), "verdict": None, "unscored": "environment",
                   "error": "environment fault persisted after 2 re-runs"}
    quarantined = {**_record(world, "negative"), "task_id": "negative", "verdict": None,
                   "error": "runtime step failed: OSError"}
    submitted = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/episode-artifacts/presign"):
            digest = json.loads(request.content)["artifact_sha256"]
            return httpx.Response(200, json={"upload_url": "https://objects.test/upload",
                                             "artifact_uri": f"s3://episodes/{digest}.json", "artifact_sha256": digest})
        if request.url.host == "objects.test":
            return httpx.Response(200)
        body = json.loads(request.content)
        submitted.extend(body["results"])
        return httpx.Response(200, json={"results": [{"eval_run_id": _EVAL_RUN_ID, "task_id": e["task_id"], "status": 201}
                                                     for e in body["results"]], "counts": {"201": len(body["results"])}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        for unscored in (environment, quarantined):
            await emit_finalized_results(
                backend_url="https://api.example", validator_keypair=_keypair(), env_pack_sha256=_PACK_SHA,
                eval_run_id=_EVAL_RUN_ID, delivery_id=_DELIVERY, task_families=_FAMILIES,
                results=[scored, unscored], http_client=client, backend_transport=client._transport,
            )

    rows = {(e["task_id"], e["outcome"], e["category"]): e for e in submitted}
    assert set(rows) == {("positive", "completed", None), ("negative", "environment_error", "environment fault"),
                         ("negative", "environment_error", None)}
    for key in [("negative", "environment_error", "environment fault"), ("negative", "environment_error", None)]:
        row = rows[key]
        assert row["score"] is None and "ledger_sha256" not in row and "verdict_sha256" not in row
        assert row["ledger_uri"].startswith("s3://episodes/") and row["episode_nonce"] == _NONCE
    assert rows[("negative", "environment_error", "environment fault")]["flags"] == ["environment_fault"]
    assert rows[("negative", "environment_error", None)]["flags"] == []


@_run_async
async def test_emit_refuses_a_task_off_the_roster(world):
    with pytest.raises(EpisodeEmitError, match="not on the run roster"):
        await emit_finalized_results(
            backend_url="https://api.example",
            validator_keypair=_keypair(),
            env_pack_sha256=_PACK_SHA,
            eval_run_id=_EVAL_RUN_ID,
            delivery_id=_DELIVERY,
            task_families={},
            results=[_record(world)],
            backend_transport=httpx.MockTransport(lambda request: httpx.Response(500)),
        )


@pytest.mark.parametrize("bad_receipt", ["missing", "duplicate", "wrong_run", "unexpected"])
@_run_async
async def test_incomplete_or_unrelated_receipts_are_not_acknowledgements(bad_receipt):
    item = {"eval_run_id": _EVAL_RUN_ID, "task_id": "t1", "status": 201}
    receipts = {
        "missing": [],
        "duplicate": [item, item],
        "wrong_run": [{**item, "eval_run_id": "00000000-0000-0000-0000-000000000001"}],
        "unexpected": [{**item, "task_id": "other"}],
    }[bad_receipt]
    transport = _MockTransport([httpx.Response(200, json={"results": receipts, "counts": {}})])
    async with environment_client("https://api.example", _keypair(), transport=transport) as client:
        with pytest.raises(EpisodeEmitError, match="roster mismatch"):
            await submit_episode_batch(client, entries=[_entry()])


def test_partial_ack_remains_pending_and_replay_receipts_allow_recovery(world):
    batches = []
    artifacts = []
    results = [_record(world, task) for task in POLICY_ACTIONS]

    def handler(request):
        if request.url.path.endswith("/episode-artifacts/presign"):
            digest = json.loads(request.content)["artifact_sha256"]
            return httpx.Response(200, json={
                "upload_url": "https://objects.test/" + digest,
                "artifact_uri": "s3://episodes/" + digest,
                "artifact_sha256": digest,
            })
        if request.url.host == "objects.test":
            assert "X-Signature" not in request.headers
            artifacts.append(request.content)
            return httpx.Response(200)
        batches.append(json.loads(request.content))
        statuses = [201, 422] if len(batches) == 1 else [409, 201]
        return httpx.Response(200, json={
            "results": [{"eval_run_id": _EVAL_RUN_ID, "task_id": item["task_id"], "status": status}
                        for item, status in zip(results, statuses)],
            "counts": {str(status): 1 for status in statuses},
        })

    def emit(batch):
        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as objects:
                await emit_finalized_results(
                    backend_url="https://api.example", validator_keypair=_keypair(),
                    env_pack_sha256=_PACK_SHA, eval_run_id=_EVAL_RUN_ID,
                    delivery_id=_DELIVERY, task_families=_FAMILIES, results=batch,
                    http_client=objects, backend_transport=objects._transport,
                )
        asyncio.run(run())

    reporter = GeneratedProgressReporter(None, emit)
    with pytest.raises(EpisodeEmitError, match="rejected"):
        reporter.flush(results)
    assert reporter._acknowledged == set()
    reporter.flush(results)
    reporter.flush(results)
    assert reporter._acknowledged == set(POLICY_ACTIONS)
    assert len(batches) == 2
    assert batches[0] == batches[1]
    assert artifacts[:2] == artifacts[2:]


# ---------------------------------------------------------------------------
# regrade / replay_matches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_id", list(POLICY_ACTIONS))
def test_regrade_reproduces_the_verdict_from_recorded_readings(world, task_id):
    pack = world[0]
    record = _record(world, task_id)

    assert regrade(pack, record["ledger"]) == record["verdict"]
    assert replay_matches(pack, record) is True


def test_positive_grading_records_a_model_reading(world):
    grading = _record(world)["ledger"]["grading"]
    assert [reading["service"] for reading in grading] == ["complete"]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda grading: grading.clear(), "missing"),
        (lambda grading: grading.append(copy.deepcopy(grading[0])), "not all used"),
        (lambda grading: grading[0]["request"].update(prompt="other"), "differs"),
        (lambda grading: grading[0].update(service="search"), "differs"),
    ],
)
def test_regrade_fails_on_missing_extra_or_different_readings(world, mutate, match):
    pack = world[0]
    record = _record(world)
    mutate(record["ledger"]["grading"])

    with pytest.raises(ReplayMismatch, match=match):
        regrade(pack, record["ledger"])
    assert replay_matches(pack, record) is False


def test_replay_detects_a_changed_verdict_or_recorded_result(world):
    pack = world[0]
    record = _record(world)
    record["verdict"]["score"] = 0.5
    assert replay_matches(pack, record) is False

    record = _record(world)
    record["ledger"]["grading"][0]["result"] = {"ok": True, "text": "FAIL"}
    assert replay_matches(pack, record) is False


def test_regrade_refuses_a_ledger_from_another_pack(world, tmp_path: Path):
    other = build_preflight_pack(tmp_path)  # freshly signed: another manifest
    with pytest.raises(ReplayMismatch, match="another pack"):
        regrade(other, _record(world)["ledger"])


def test_load_inference_transcripts_rejects_a_fifo_without_blocking(tmp_path):
    output = tmp_path / "output.jsonl"
    os.mkfifo(output)

    with pytest.raises(ValueError):
        load_inference_transcripts(output)

