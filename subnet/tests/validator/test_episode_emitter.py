"""Unit coverage for the episode-result emitter."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from decimal import Decimal
from functools import wraps
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from validator import episode_emitter
from validator.env_backend import environment_client
from validator.generated_progress_reporter import GeneratedProgressReporter
from validator.episode_emitter import (
    EpisodeEmitError,
    build_episode_artifact,
    build_episode_payload,
    emit_finalized_results,
    episode_artifact_sha256,
    load_episode_artifact,
    replay_ledger,
    serialize_episode_artifact,
    submit_episode_batch,
)


def _run_async(test: Callable[..., Any]) -> Callable[..., None]:
    """Run an async test without pytest-asyncio dependency (matches
    ``test_env_pack_loader``'s pattern)."""

    @wraps(test)
    def wrapper(*args: object, **kwargs: object) -> None:
        asyncio.run(test(*args, **kwargs))

    return wrapper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_PACK_SHA = "a" * 64
_TERM_HASH = "b" * 64
_EVAL_RUN_ID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"


def _ledger_entry(seq: int, state_hash: str = _TERM_HASH) -> dict[str, Any]:
    """A minimal LedgerEntry dict as SessionRegistry.verdict() emits."""
    return {
        "seq": seq,
        "turn": seq,
        "kind": "model_action",
        "actor": "agent",
        "payload": {"tool": "noop"},
        "state_hash": state_hash,
    }


def _base_verdict(
    *,
    correct: bool = True,
    paid_reward: float | None = 0.75,
    environment_error: bool = False,
    verifier_error: bool = False,
) -> dict[str, Any]:
    """The dict shape SessionRegistry.verdict() returns."""
    return {
        "evaluation_run_id": _EVAL_RUN_ID,
        "agent_version_id": "agentv-1",
        "task_id": "TF2-retrieval_recall-1",
        "session_id": "sess-1",
        "pack_sha256": _PACK_SHA,
        "family": "retrieval_recall",
        "terminal_reason": "environment_done",
        "verdict": {
            "correct": correct,
            "paid_reward": paid_reward,
            "efficiency": 0.9,
            "checks": {"final_in_gold": True, "within_budget": True},
            "reward_record": {"retrieval": 0.9, "efficiency": 0.6},
        },
        "terminal_state_hash": _TERM_HASH,
        "step_count": 12,
        "wall_seconds": 123.456789,
        "render_budget": None,
        "bootstrap": {
            "session_id": "sess-1",
            "state_hash": "0" * 64,
            "policy_view": {"query": "find a valid option"},
        },
        "call_trace": [
            {
                "request": {"call_id": "call-1", "turn": 1},
                "response": {"state_hash": _TERM_HASH},
                "state_hash_before": "0" * 64,
                "state_hash_after": _TERM_HASH,
                "latency_ms": 1.25,
                "error": None,
            }
        ],
        "ledger": [_ledger_entry(1), _ledger_entry(2)],
        "provenance": {"pack_sha256": _PACK_SHA, "runtime_version": "1"},
        "environment_error": environment_error,
        "verifier_error": verifier_error,
    }


def _payload_kwargs(**overrides: Any) -> dict[str, Any]:
    """Standard kwargs for build_episode_payload."""
    base = {
        "eval_run_id": _EVAL_RUN_ID,
        "env_pack_sha256": _PACK_SHA,
        "ledger_uri": "s3://b/k",
    }
    base.update(overrides)
    return base


def test_episode_artifact_is_content_addressed_and_round_trips():
    artifact = build_episode_artifact(_base_verdict())
    body = serialize_episode_artifact(artifact)
    digest = episode_artifact_sha256(body)

    assert load_episode_artifact(body, expected_sha256=digest) == artifact
    with pytest.raises(EpisodeEmitError, match="sha256 mismatch"):
        load_episode_artifact(body, expected_sha256="0" * 64)


# ---------------------------------------------------------------------------
# build_episode_payload — outcome classification + Backend invariants
# ---------------------------------------------------------------------------


def test_payload_completed_happy_path():
    verdict = _base_verdict(correct=True, paid_reward=0.75)
    p = build_episode_payload(verdict, **_payload_kwargs())
    assert p["outcome"] == "completed"
    assert p["verdict_correct"] is True
    assert p["aggregate_reward"] == "0.75"
    assert p["terminal_state_hash"] == _TERM_HASH
    assert p["ledger_uri"] == "s3://b/k"
    assert p["step_count"] == 12
    assert p["wall_seconds"] == 123.456789
    assert p["verdict_checks"]["final_in_gold"] is True
    assert p["reward_components"]["retrieval"] == 0.9


def test_payload_environment_error_dominates_terminal_reason():
    """Even if the session claims environment_done, an environment_error
    flag from the harness (quarantined session) wins."""
    verdict = _base_verdict(environment_error=True)
    p = build_episode_payload(verdict, **_payload_kwargs(ledger_uri=None))
    assert p["outcome"] == "environment_error"
    # State outcomes get the hash; error outcomes must not (Backend rejects).
    assert p["terminal_state_hash"] is None


def test_payload_verifier_error_dominates_verdict():
    verdict = _base_verdict(verifier_error=True)
    p = build_episode_payload(verdict, **_payload_kwargs(ledger_uri=None))
    assert p["outcome"] == "verifier_error"
    assert p["terminal_state_hash"] is None


def test_payload_reward_null_when_verdict_incorrect():
    """Backend rejects aggregate_reward != null when verdict_correct=false;
    the emitter enforces that here so failures surface with a specific
    error instead of a per-item 422. TF4 shadow reward stays visible via
    reward_components — it's telemetry, not on-chain payment."""
    verdict = _base_verdict(correct=False, paid_reward=0.5)
    p = build_episode_payload(verdict, **_payload_kwargs())
    assert p["verdict_correct"] is False
    assert p["aggregate_reward"] is None
    # TF4-shadow-style rewards keep the telemetry in reward_components.
    assert p["reward_components"]["retrieval"] == 0.9


def test_payload_step_limit_still_completes():
    """Step-limit termination isn't an error — the verifier ran and
    produced a verdict. Outcome=completed even if correct=False."""
    verdict = _base_verdict(correct=False, paid_reward=None)
    verdict["terminal_reason"] = "step_limit"
    p = build_episode_payload(verdict, **_payload_kwargs())
    assert p["outcome"] == "completed"
    assert p["verdict_correct"] is False
    assert p["aggregate_reward"] is None


def test_payload_raises_when_state_outcome_missing_hash():
    """completed / partial / leakage / exploit all require a hash;
    Backend's model_validator would reject the row otherwise."""
    verdict = _base_verdict()
    verdict["terminal_state_hash"] = None
    with pytest.raises(EpisodeEmitError, match="requires terminal_state_hash"):
        build_episode_payload(verdict, **_payload_kwargs())


def test_payload_raises_when_missing_task_id():
    """Missing required verdict field surfaces as EpisodeEmitError with
    a descriptive message, not a bare KeyError leaking through."""
    verdict = _base_verdict()
    del verdict["task_id"]
    with pytest.raises(EpisodeEmitError, match="missing required field 'task_id'"):
        build_episode_payload(verdict, **_payload_kwargs())


def test_payload_raises_when_missing_family():
    verdict = _base_verdict()
    del verdict["family"]
    with pytest.raises(EpisodeEmitError, match="missing required field 'family'"):
        build_episode_payload(verdict, **_payload_kwargs())


def test_payload_uses_pack_sha_from_kwargs_not_verdict():
    """The pack SHA bound to the work item in the claim response is
    authoritative — the verdict's pack_sha256 is validated separately by
    the loader and must match, but the payload must carry the frozen
    binding regardless."""
    verdict = _base_verdict()
    verdict["pack_sha256"] = "c" * 64  # deliberately different
    p = build_episode_payload(verdict, **_payload_kwargs())
    assert p["env_pack_sha256"] == _PACK_SHA


def test_payload_reward_is_exact_decimal_string():
    """Backend accepts Decimal or str with ≤6 decimal places; keep the
    format stable so audits can compare across languages / JSON parsers."""
    verdict = _base_verdict(correct=True, paid_reward=0.123456)
    p = build_episode_payload(verdict, **_payload_kwargs())
    assert p["aggregate_reward"] == format(Decimal("0.123456"), "f")


# ---------------------------------------------------------------------------
# submit_episode_batch
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
    return build_episode_payload({**_base_verdict(), "task_id": task_id}, **_payload_kwargs())


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
    monkeypatch: pytest.MonkeyPatch,
):
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
                "results": [{"eval_run_id": _EVAL_RUN_ID, "task_id": "TF2-retrieval_recall-1", "status": 201}],
                "counts": {"201": 1},
            },
        )

    sleep = AsyncMock()
    monkeypatch.setattr(episode_emitter.asyncio, "sleep", sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await emit_finalized_results(
            backend_url="https://api.example",
            validator_keypair=_keypair(),
            env_pack_sha256=_PACK_SHA,
            results=[_base_verdict()],
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
    assert artifact["episode"]["call_trace"][0]["request"]["call_id"] == "call-1"
    assert submitted is not None
    assert submitted["results"][0]["ledger_uri"].startswith("s3://episodes/")


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


def test_partial_ack_remains_pending_and_replay_receipts_allow_recovery():
    batches = []
    artifacts = []
    results = [{**_base_verdict(), "task_id": task} for task in ("one", "two")]

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
                    env_pack_sha256=_PACK_SHA, results=batch,
                    http_client=objects, backend_transport=objects._transport,
                )
        asyncio.run(run())

    reporter = GeneratedProgressReporter(None, emit)
    with pytest.raises(EpisodeEmitError, match="rejected"):
        reporter.flush(results)
    assert reporter._acknowledged == set()
    reporter.flush(results)
    reporter.flush(results)
    assert reporter._acknowledged == {"one", "two"}
    assert len(batches) == 2
    assert batches[0] == batches[1]
    assert artifacts[:2] == artifacts[2:]

# ---------------------------------------------------------------------------
# replay_ledger
# ---------------------------------------------------------------------------


class _FakeVerifierResult:
    """Stand-in for VerifierResult so the replay test doesn't need a full
    pack + catalog. The real verifier is exercised end-to-end in
    the shared runtime compatibility tests."""

    def __init__(self, correct: bool, paid_reward: float, efficiency: float, checks: dict[str, Any]):
        self.correct = correct
        self.paid_reward = paid_reward
        self.efficiency = efficiency
        self.checks = checks

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return {
            "correct": self.correct,
            "paid_reward": self.paid_reward,
            "efficiency": self.efficiency,
            "checks": self.checks,
        }


def test_replay_matches_on_deterministic_verdict_and_hash(monkeypatch):
    from oro_env_runtime.schema import LedgerEntry

    ledger = [
        LedgerEntry(seq=1, turn=1, kind="model_action", actor="agent", payload={"tool": "noop"}, state_hash="00" * 32),
        LedgerEntry(seq=2, turn=2, kind="observation", actor="harness", payload={}, state_hash=_TERM_HASH),
    ]
    expected_verdict = {
        "correct": True,
        "paid_reward": 0.75,
        "efficiency": 0.9,
        "checks": {"final_in_gold": True},
    }

    monkeypatch.setattr(
        episode_emitter,
        "verify",
        lambda task, ledger, catalog, *, reference_count, render_budget: _FakeVerifierResult(
            correct=True, paid_reward=0.75, efficiency=0.9, checks={"final_in_gold": True}
        ),
    )

    task = MagicMock()
    task.reference_action_count = 3
    result = replay_ledger(
        ledger=ledger,
        task=task,
        catalog=MagicMock(),
        expected_terminal_hash=_TERM_HASH,
        expected_verdict=expected_verdict,
        render_budget=50,
    )
    assert result.terminal_hash_matches is True
    assert result.verdict_matches is True
    assert result.replayed_terminal_hash == _TERM_HASH


def test_replay_flags_terminal_hash_mismatch(monkeypatch):
    from oro_env_runtime.schema import LedgerEntry

    ledger = [
        LedgerEntry(seq=1, turn=1, kind="model_action", actor="agent", payload={}, state_hash="ff" * 32),
    ]
    monkeypatch.setattr(
        episode_emitter,
        "verify",
        lambda task, ledger, catalog, *, reference_count, render_budget: _FakeVerifierResult(
            correct=True, paid_reward=0.0, efficiency=0.0, checks={}
        ),
    )
    task = MagicMock()
    task.reference_action_count = 1
    result = replay_ledger(
        ledger=ledger,
        task=task,
        catalog=MagicMock(),
        expected_terminal_hash=_TERM_HASH,  # differs from ledger's last state_hash
        expected_verdict={"correct": True, "paid_reward": 0.0, "efficiency": 0.0, "checks": {}},
        render_budget=50,
    )
    assert result.terminal_hash_matches is False
    assert result.replayed_terminal_hash == "ff" * 32


def test_replay_forwards_render_budget_to_verifier(monkeypatch):
    """Regression guard: `render_budget`
    MUST reach `verify()` — a None default silently changes efficiency
    scoring and would always report ``verdict_matches=False`` for a
    faithfully-recorded ledger."""
    from oro_env_runtime.schema import LedgerEntry

    captured: dict[str, Any] = {}

    def _fake_verify(task, ledger, catalog, *, reference_count, render_budget):
        captured["render_budget"] = render_budget
        captured["reference_count"] = reference_count
        return _FakeVerifierResult(correct=True, paid_reward=0.0, efficiency=0.0, checks={})

    monkeypatch.setattr(episode_emitter, "verify", _fake_verify)

    task = MagicMock()
    task.reference_action_count = 4
    replay_ledger(
        ledger=[
            LedgerEntry(
                seq=1, turn=1, kind="model_action", actor="agent",
                payload={}, state_hash=_TERM_HASH,
            )
        ],
        task=task,
        catalog=MagicMock(),
        expected_terminal_hash=_TERM_HASH,
        expected_verdict={"correct": True, "paid_reward": 0.0, "efficiency": 0.0, "checks": {}},
        render_budget=17,
    )
    assert captured["render_budget"] == 17
    assert captured["reference_count"] == 4
