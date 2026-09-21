"""Emit and replay complete episode results.

Ships a terminated ``TaskSession`` through the Backend's episode-result
receiver (``POST /v1/validator/episode-results``). The
validator holds the truth end-to-end: session termination produces a
verdict + trace + ledger, orchestration uploads one content-addressed
artifact through a Backend-issued URL, then posts the existing searchable
``EpisodeResultEntry`` summary. Replay verifies the stored artifact's hash,
terminal state, and deterministic verdict offline.

The emitter is intentionally **standalone**:

- Takes the dict output of ``SessionRegistry.finalized_results()``
  rather than importing the registry — session termination wiring
  belongs to the orchestration layer, not the transport layer.
- Uses the published SDK for Backend request/response models and signing.
  Content-addressed object-store uploads use a separate unsigned client.

Payload shape must match ``EpisodeResultEntry`` in
``ORO-AI/Backend:app/models/schemas/env_pack.py``. See constants below
for the outcome taxonomy — Backend's Literal ``_OUTCOME_LITERAL`` is
the source of truth.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
from oro_sdk import Client
from oro_sdk.api.validator import presign_episode_artifact, submit_episode_results
from oro_sdk.models.episode_artifact_presign_request import EpisodeArtifactPresignRequest
from oro_sdk.models.episode_artifact_presign_response import EpisodeArtifactPresignResponse
from oro_sdk.models.submit_episode_results_request import SubmitEpisodeResultsRequest
from oro_sdk.models.submit_episode_results_response import SubmitEpisodeResultsResponse
from oro_sdk.retry import compute_delay, parse_retry_after
from oro_env_runtime.schema import LedgerEntry, TaskSpec, VerifierResult
from oro_env_runtime.verify import verify

from .backend_client import BackendError
from .env_backend import call_environment_api, environment_client

EPISODE_ARTIFACT_SCHEMA_VERSION = "oro.environment_episode.v1"
INFERENCE_TRANSCRIPT_SCHEMA_VERSION = "oro.internal_inference_transcript.v1"
_REQUEST_ATTEMPTS = 3

_SECRET_KEY = re.compile(
    r"^(?:authorization|proxy[_-]authorization|token|api[_-]key|access[_-]token|"
    r"management[_-]token|credential|credentials|password|secret)$|"
    r"(?:^|_)(?:api_key|access_token|management_token|credential|credentials|password|secret)$",
    re.IGNORECASE,
)
_URL = re.compile(r"https?://[^\s\"'<>]+")


# Backend requires terminal_state_hash for these outcomes (state exists).
# `_OUTCOMES_WITH_STATE` at Backend/app/models/schemas/env_pack.py:273.
# MVP emits only `completed | environment_error | verifier_error` —
# widening to `partial | leakage | exploit` is additive when the
# verifier + admission gates start surfacing those signals.
_OUTCOMES_WITH_STATE = frozenset({"completed", "partial", "leakage", "exploit"})


class EpisodeEmitError(RuntimeError):
    """Episode artifact or summary emission failed."""


async def _request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    operation: str,
    **kwargs: Any,
) -> httpx.Response:
    """Retry an unsigned object-store upload; never attach Backend credentials."""

    for attempt in range(_REQUEST_ATTEMPTS):
        retry_after = None
        try:
            response = await client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            if attempt == _REQUEST_ATTEMPTS - 1:
                raise EpisodeEmitError(
                    f"{operation} failed: {type(exc).__name__}"
                ) from exc
        else:
            if response.status_code < 400:
                return response
            if response.status_code != 429 and response.status_code < 500:
                raise EpisodeEmitError(
                    f"{operation} rejected status={response.status_code}"
                )
            if attempt == _REQUEST_ATTEMPTS - 1:
                raise EpisodeEmitError(
                    f"{operation} transient failure status={response.status_code}"
                )
            retry_after = parse_retry_after(response)
        await asyncio.sleep(compute_delay(attempt, 1.0, 60.0, False, retry_after))

    raise AssertionError("request retry loop exhausted unexpectedly")


def _sanitize_transcript_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize_transcript_value(item)
            for key, item in value.items()
            if _SECRET_KEY.search(str(key).replace("-", "_")) is None
        }
    if isinstance(value, list):
        return [_sanitize_transcript_value(item) for item in value]
    if isinstance(value, str):
        return _URL.sub(lambda match: match.group(0).split("?", 1)[0], value)
    return value


def _inference_calls(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    raw_calls = envelope.get("_shadow_proxy_calls")
    if not isinstance(raw_calls, list):
        return []
    calls = [
        call
        for call in raw_calls
        if isinstance(call, dict)
        and "/inference/" in str(call.get("path") or "")
    ]
    return sorted(calls, key=lambda call: call.get("timestamp", 0))


def _transcript_agent_output(envelope: dict[str, Any]) -> dict[str, Any]:
    agent_output = copy.deepcopy(envelope)
    agent_output.pop("_shadow_inference_usage", None)
    agent_output.pop("_shadow_proxy_calls", None)
    for step in agent_output.get("dialogue") or []:
        extra = step.get("extra_info") if isinstance(step, dict) else None
        if isinstance(extra, dict):
            extra.pop("proxy_calls", None)
    return agent_output


def load_inference_transcripts(output_file: str | Path) -> dict[str, dict[str, Any]]:
    """Join each sandbox output envelope to its ordered inference calls."""

    path = Path(output_file)
    transcripts: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            envelope = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EpisodeEmitError("sandbox output contains invalid JSON") from exc
        session_id = envelope.get("problem_id") if isinstance(envelope, dict) else None
        if not isinstance(session_id, str) or not session_id:
            raise EpisodeEmitError("sandbox output is missing episode identifier")

        transcripts[session_id] = _sanitize_transcript_value(
            {
                "schema_version": INFERENCE_TRANSCRIPT_SCHEMA_VERSION,
                "session_id": session_id,
                "final_status": envelope.get("status"),
                "agent_output": _transcript_agent_output(envelope),
                "calls": _inference_calls(envelope),
            }
        )
    return transcripts


def build_episode_artifact(
    result: dict[str, Any], *, inference_transcript: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Wrap existing runtime snapshots without inventing a parallel trace model."""

    if not isinstance(result.get("bootstrap"), dict):
        raise EpisodeEmitError("finalized result is missing session bootstrap")
    if not isinstance(result.get("call_trace"), list):
        raise EpisodeEmitError("finalized result is missing ordered call trace")
    if not isinstance(result.get("ledger"), list):
        raise EpisodeEmitError("finalized result is missing environment ledger")
    artifact = {
        "schema_version": EPISODE_ARTIFACT_SCHEMA_VERSION,
        "episode": result,
    }
    if inference_transcript is not None:
        artifact["inference_transcript"] = inference_transcript
    return artifact


def serialize_episode_artifact(artifact: dict[str, Any]) -> bytes:
    """Return deterministic bytes suitable for content-addressed storage."""

    return json.dumps(
        artifact,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def episode_artifact_sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def load_episode_artifact(
    body: bytes, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    """Load a standalone artifact and verify its content address when supplied."""

    actual_sha256 = episode_artifact_sha256(body)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise EpisodeEmitError(
            f"episode artifact sha256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        artifact = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EpisodeEmitError("episode artifact is not valid JSON") from exc
    if not isinstance(artifact, dict):
        raise EpisodeEmitError("episode artifact must be a JSON object")
    if artifact.get("schema_version") != EPISODE_ARTIFACT_SCHEMA_VERSION:
        raise EpisodeEmitError("unsupported episode artifact schema_version")
    if not isinstance(artifact.get("episode"), dict):
        raise EpisodeEmitError("episode artifact is missing episode result")
    return artifact


# ---------------------------------------------------------------------------
# verdict() -> EpisodeResultEntry payload
# ---------------------------------------------------------------------------


def _classify_outcome(verdict: dict[str, Any]) -> str:
    """Prefer the finalized outcome, retaining legacy verdict compatibility."""
    explicit = verdict.get("outcome")
    if explicit in {
        "completed",
        "partial",
        "agent_error",
        "environment_error",
        "verifier_error",
        "leakage",
        "exploit",
    }:
        return str(explicit)
    if verdict.get("environment_error"):
        return "environment_error"
    if verdict.get("verifier_error"):
        return "verifier_error"
    return "completed"


def build_episode_payload(
    verdict: dict[str, Any],
    *,
    eval_run_id: str,
    env_pack_sha256: str,
    ledger_uri: str | None,
) -> dict[str, Any]:
    """Map ``SessionRegistry.verdict()`` output to an ``EpisodeResultEntry``.

    ``env_pack_sha256`` comes from the claim response (frozen
    at work-item creation), NOT from ``verdict['pack_sha256']``. The two
    should agree, but the frozen claim binding is authoritative — the
    pack the session loaded is verified by the pack loader
    against the claim binding separately.

    Reward vs telemetry split — Backend rejects a paid reward attached
    to a failed hard gate as a cheating attempt:

    - ``aggregate_reward`` is the on-chain payment. Populated ONLY when
      ``verdict_correct=True``.
    - ``reward_components`` is the full ``reward_record`` telemetry.
      TF4 shadow rewards (paid_value > 0 with correct=False) still
      appear here for training signal, but do NOT influence the
      on-chain payment.

    Enforces two Backend invariants at the emitter boundary so a bad
    verdict fails fast with a specific error instead of coming back as
    a per-item 422:

    - ``terminal_state_hash`` is set iff the outcome is in
      ``_OUTCOMES_WITH_STATE`` (else Backend rejects).
    - Required verdict fields (``task_id``, ``family``) must be present;
      absence bubbles as ``EpisodeEmitError`` rather than a bare KeyError.
    """
    try:
        task_id = str(verdict["task_id"])
        family = str(verdict["family"])
    except KeyError as exc:
        raise EpisodeEmitError(
            f"verdict missing required field {exc.args[0]!r} (run={eval_run_id})"
        ) from exc

    outcome = _classify_outcome(verdict)
    verifier_result = verdict.get("verdict") or {}
    verdict_correct = bool(verifier_result.get("correct"))

    paid_reward = verifier_result.get("paid_reward")
    if verdict_correct and paid_reward is not None:
        # Serialize through Decimal to keep the string exact (Backend
        # accepts str/Decimal, 6 decimal places max).
        aggregate_reward: str | None = format(Decimal(str(paid_reward)), "f")
    else:
        aggregate_reward = None

    terminal_state_hash = verdict.get("terminal_state_hash")
    if outcome in _OUTCOMES_WITH_STATE:
        if not terminal_state_hash:
            raise EpisodeEmitError(
                f"outcome={outcome} requires terminal_state_hash but verdict "
                f"provided none (run={eval_run_id}, task={task_id})"
            )
    else:
        terminal_state_hash = None

    return {
        "eval_run_id": eval_run_id,
        "env_pack_sha256": env_pack_sha256,
        "task_id": task_id,
        "family": family,
        "outcome": outcome,
        "verdict_correct": verdict_correct,
        "verdict_checks": dict(verifier_result.get("checks") or {}),
        "reward_components": dict(verifier_result.get("reward_record") or {}),
        "aggregate_reward": aggregate_reward,
        "wall_seconds": verdict.get("wall_seconds"),
        "terminal_state_hash": terminal_state_hash,
        "ledger_uri": ledger_uri,
        "step_count": int(verdict.get("step_count") or 0),
    }


async def upload_episode_artifact(
    client: httpx.AsyncClient,
    *,
    backend: Client,
    result: dict[str, Any],
    inference_transcript: dict[str, Any] | None = None,
    download_url_rewriter: Callable[[str], str] | None = None,
) -> str:
    """Presign and upload one content-addressed episode artifact."""

    artifact = build_episode_artifact(
        result,
        inference_transcript=inference_transcript,
    )
    body = serialize_episode_artifact(artifact)
    artifact_sha256 = episode_artifact_sha256(body)
    request = EpisodeArtifactPresignRequest.from_dict(
        {
            "eval_run_id": result.get("evaluation_run_id"),
            "env_pack_sha256": result.get("pack_sha256"),
            "artifact_sha256": artifact_sha256,
            "content_length": len(body),
        }
    )
    try:
        response = await call_environment_api(
            presign_episode_artifact.asyncio_detailed,
            EpisodeArtifactPresignResponse,
            operation="episode artifact presign",
            client=backend,
            body=request,
        )
    except BackendError as exc:
        raise EpisodeEmitError(str(exc)) from None
    presign = response.parsed
    if presign.artifact_sha256 != artifact_sha256:
        raise EpisodeEmitError("episode artifact presign hash mismatch")
    upload_url = presign.upload_url
    artifact_uri = presign.artifact_uri
    if not upload_url or not artifact_uri.startswith("s3://"):
        raise EpisodeEmitError("episode artifact presign response is incomplete")
    if download_url_rewriter is not None:
        upload_url = download_url_rewriter(upload_url)
    await _request(
        client,
        "PUT",
        upload_url,
        operation="episode artifact upload",
        content=body,
        headers={"Content-Type": "application/json"},
    )
    return artifact_uri


def _transcript_for_result(
    transcripts: dict[str, dict[str, Any]] | None,
    result: dict[str, Any],
) -> dict[str, Any] | None:
    if transcripts is None:
        return None
    transcript = transcripts.get(str(result.get("session_id")))
    if transcript is None and _classify_outcome(result) == "completed":
        raise EpisodeEmitError("completed episode is missing its inference transcript")
    return transcript


# ---------------------------------------------------------------------------
# Backend POST
# ---------------------------------------------------------------------------


_MAX_BATCH_SIZE = 500  # Backend cap on results per request (env_pack.py:391).


async def submit_episode_batch(
    client: Client,
    *,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """POST a batch to ``/v1/validator/episode-results``.

    Backend returns 200 with a per-item ``results`` array + ``counts``
    breakdown; a single bad row does NOT tank the batch (Backend
    handles per-item 422/409/404 individually). Transient failures are
    retried in place with fresh validator-auth nonces.

    Splits batches over ``_MAX_BATCH_SIZE`` and merges responses so the
    caller can hand any-size list.
    """
    if not entries:
        raise ValueError("cannot submit an empty batch")

    merged_results: list[dict[str, Any]] = []
    merged_counts: Counter[str] = Counter()

    for chunk_start in range(0, len(entries), _MAX_BATCH_SIZE):
        chunk = entries[chunk_start : chunk_start + _MAX_BATCH_SIZE]
        try:
            response = await call_environment_api(
                submit_episode_results.asyncio_detailed,
                SubmitEpisodeResultsResponse,
                operation=f"episode-results chunk {chunk_start}",
                client=client,
                body=SubmitEpisodeResultsRequest.from_dict({"results": chunk}),
            )
        except BackendError as exc:
            raise EpisodeEmitError(str(exc)) from None
        body = response.parsed.to_dict()
        expected = Counter((str(item["eval_run_id"]), item["task_id"]) for item in chunk)
        received = Counter((item["eval_run_id"], item["task_id"]) for item in body["results"])
        if received != expected:
            raise EpisodeEmitError("episode-results acknowledgement roster mismatch")
        merged_results.extend(body.get("results", []))
        merged_counts.update(
            {k: int(v) for k, v in (body.get("counts") or {}).items()}
        )

    return {"results": merged_results, "counts": dict(merged_counts)}


async def emit_finalized_results(
    *,
    backend_url: str,
    validator_keypair: Any,
    env_pack_sha256: str,
    results: list[dict[str, Any]],
    inference_transcripts: dict[str, dict[str, Any]] | None = None,
    download_url_rewriter: Callable[[str], str] | None = None,
    http_client: httpx.AsyncClient | None = None,
    backend_transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Upload complete artifacts and submit their searchable summaries."""

    if not results:
        return {"results": [], "counts": {}}
    owned_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(60.0))
    try:
        async with environment_client(
            backend_url, validator_keypair, transport=backend_transport
        ) as backend:
            payloads = []
            for result in results:
                transcript = _transcript_for_result(inference_transcripts, result)
                artifact_uri = await upload_episode_artifact(
                    client,
                    backend=backend,
                    result=result,
                    inference_transcript=transcript,
                    download_url_rewriter=download_url_rewriter,
                )
                payloads.append(
                    build_episode_payload(
                        result,
                        eval_run_id=str(result["evaluation_run_id"]),
                        env_pack_sha256=env_pack_sha256,
                        ledger_uri=artifact_uri,
                    )
                )
            submitted = await submit_episode_batch(
                backend,
                entries=payloads,
            )
        rejected = [
            item
            for item in submitted.get("results", [])
            if item.get("status") not in {201, 409}
        ]
        if rejected:
            raise EpisodeEmitError(
                "episode-results rejected one or more summaries: "
                + json.dumps(rejected[:3], sort_keys=True)
            )
        return submitted
    except BackendError as exc:
        raise EpisodeEmitError(str(exc)) from None
    finally:
        if owned_client:
            await client.aclose()


# ---------------------------------------------------------------------------
# Replay — cross-validator parity check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayResult:
    """Outcome of a ledger replay against a locally-loaded pack.

    ``terminal_hash_matches`` is the parity anchor Backend + audits
    compare across validators. ``verdict_matches`` is the deeper
    check: independently re-running the verifier on the same ledger
    should give the same VerifierResult (correct + reward + checks).
    """

    terminal_hash_matches: bool
    verdict_matches: bool
    expected_terminal_hash: str | None
    replayed_terminal_hash: str | None
    expected_verdict: dict[str, Any] | None
    replayed_verdict: dict[str, Any] | None


def replay_ledger(
    *,
    ledger: list[LedgerEntry],
    task: TaskSpec,
    catalog: Any,
    expected_terminal_hash: str | None,
    expected_verdict: dict[str, Any] | None,
    render_budget: int | None,
) -> ReplayResult:
    """Re-run the verifier on a fetched ledger and compare.

    ``render_budget`` is REQUIRED (no default) — pass the same value
    the original session used. ``verify.py`` uses it to reconstruct the
    solver-visible surface for observed-candidate metrics; a wrong
    value silently changes efficiency scoring and the replay reports
    ``verdict_matches=False`` for a faithfully-recorded ledger,
    defeating the cross-validator parity check. When SessionRegistry starts
    persisting the render budget in the verdict / ledger metadata,
    callers should thread it through from there.

    ``catalog`` is the pack's ``env.catalog.Catalog`` (same instance
    the session used). Caller owns pack loading — this function stays
    pure so it's cheap to run in a batch parity job.

    ``verdict_matches`` compares the semantically-important fields
    (``correct``, ``paid_reward``, ``efficiency``, ``checks``) rather
    than the full dict — TF4 judge nondeterminism (async LLM call)
    means full equality would flap on preference_reasoning tasks; the
    deterministic core still matches.
    """
    replayed_terminal_hash = ledger[-1].state_hash if ledger else None
    terminal_hash_matches = (
        expected_terminal_hash is not None
        and expected_terminal_hash == replayed_terminal_hash
    )

    replayed_verifier: VerifierResult = verify(
        task,
        ledger,
        catalog,
        reference_count=task.reference_action_count,
        render_budget=render_budget,
    )
    replayed_verdict = replayed_verifier.model_dump(mode="json")

    verdict_matches = expected_verdict is not None and all(
        expected_verdict.get(k) == replayed_verdict.get(k)
        for k in ("correct", "paid_reward", "efficiency")
    ) and (expected_verdict.get("checks") or {}) == (replayed_verdict.get("checks") or {})

    return ReplayResult(
        terminal_hash_matches=terminal_hash_matches,
        verdict_matches=verdict_matches,
        expected_terminal_hash=expected_terminal_hash,
        replayed_terminal_hash=replayed_terminal_hash,
        expected_verdict=expected_verdict,
        replayed_verdict=replayed_verdict,
    )


__all__ = [
    "EPISODE_ARTIFACT_SCHEMA_VERSION",
    "EpisodeEmitError",
    "ReplayResult",
    "build_episode_artifact",
    "build_episode_payload",
    "emit_finalized_results",
    "episode_artifact_sha256",
    "load_episode_artifact",
    "load_inference_transcripts",
    "replay_ledger",
    "serialize_episode_artifact",
    "submit_episode_batch",
    "upload_episode_artifact",
]
