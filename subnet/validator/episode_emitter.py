"""Emit and replay complete world-pack episode results.

Ships the terminal records of ``SessionRegistry.terminal_results()`` through
the Backend's episode-result receiver (``POST /v1/validator/episode-results``):
orchestration uploads one content-addressed artifact (the runtime ledger,
verdict and call trace) through a Backend-issued URL, then posts the
searchable ``EpisodeResultEntry`` summary carrying the verdict's score,
category, flags, canonical digests and the claim's episode nonce.

Replay re-grades a recorded ledger with the runtime's public
``grade_episode`` and a reader that serves the ledger's recorded grading
readings in order, so it never queries a model or the search server.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import re
from collections import Counter
from collections.abc import Callable, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import unquote_plus, urlsplit, urlunsplit

import httpx
from oro_env_runtime import canonical
from oro_env_runtime.pack import Pack
from oro_env_runtime.session import grade_episode
from oro_sdk import Client
from oro_sdk.api.validator import presign_episode_artifact, submit_episode_results
from oro_sdk.models.episode_artifact_presign_request import (
    EpisodeArtifactPresignRequest,
)
from oro_sdk.models.episode_artifact_presign_response import (
    EpisodeArtifactPresignResponse,
)
from oro_sdk.models.submit_episode_results_request import SubmitEpisodeResultsRequest
from oro_sdk.models.submit_episode_results_response import SubmitEpisodeResultsResponse
from oro_sdk.retry import compute_delay, parse_retry_after
from src.agent.sandbox_executor import MAX_SANDBOX_OUTPUT_BYTES, read_regular_file

from .backend_client import BackendError
from .env_backend import call_environment_api, environment_client

EPISODE_ARTIFACT_SCHEMA_VERSION = "oro.environment_episode.v1"
INFERENCE_TRANSCRIPT_SCHEMA_VERSION = "oro.internal_inference_transcript.v1"
_REQUEST_ATTEMPTS = 3

_SECRET_KEY = re.compile(
    r"^(?:authorization|proxy_authorization|token|api_key|access_token|"
    r"management_token|session_token|security_token|auth_token|refresh_token|"
    r"credential|credentials|password|secret|"
    r"signature|sig|cookie|set_cookie)$|"
    r"(?:^|_)(?:api_key|access_token|management_token|session_token|security_token|"
    r"auth_token|refresh_token|credential|credentials|password|secret|signature|"
    r"sig|cookie)$"
)
_URL = re.compile(r"https?://[^\s\"'<>]+")
logger = logging.getLogger(__name__)


class EpisodeEmitError(RuntimeError):
    """Episode artifact or summary emission failed."""


class ReplayMismatch(RuntimeError):
    """A recorded reading is missing, unused or different from the replayed request."""


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


def _normalized_key(value: Any) -> str:
    key = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(value))
    return re.sub(r"[^a-zA-Z0-9]+", "_", key).strip("_").lower()


def _sanitize_url(value: str) -> str:
    parts = urlsplit(value)
    query = []
    for item in parts.query.split("&"):
        key, separator, _ = item.partition("=")
        if separator and _SECRET_KEY.search(_normalized_key(unquote_plus(key))):
            query.append(f"{key}=[REDACTED]")
        else:
            query.append(item)
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, "&".join(query), parts.fragment)
    )


def _sanitize_transcript_value(
    value: Any, *, secret_values: tuple[str, ...] = ()
) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize_transcript_value(item, secret_values=secret_values)
            for key, item in value.items()
            if _SECRET_KEY.search(_normalized_key(key)) is None
        }
    if isinstance(value, list):
        return [
            _sanitize_transcript_value(item, secret_values=secret_values)
            for item in value
        ]
    if isinstance(value, str):
        sanitized = _URL.sub(lambda match: _sanitize_url(match.group(0)), value)
        for secret in secret_values:
            if secret:
                sanitized = sanitized.replace(secret, "[REDACTED]")
        return sanitized
    return value


def _inference_calls(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    raw_calls = envelope.get("_shadow_proxy_calls")
    if not isinstance(raw_calls, list):
        return []
    calls = [
        call
        for call in raw_calls
        if isinstance(call, dict) and "/inference/" in str(call.get("path") or "")
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


def load_inference_transcripts(
    output_file: str | Path, *, secret_values: tuple[str, ...] = ()
) -> dict[str, dict[str, Any]]:
    """Join each sandbox output envelope to its ordered inference calls."""

    path = Path(output_file)
    transcripts: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        read_regular_file(path, MAX_SANDBOX_OUTPUT_BYTES).decode("utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            envelope = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Ignoring malformed sandbox output line %d", line_number)
            continue
        session_id = envelope.get("problem_id") if isinstance(envelope, dict) else None
        if not isinstance(session_id, str) or not session_id:
            logger.warning(
                "Ignoring sandbox output line %d without an episode ID", line_number
            )
            continue
        if session_id in transcripts:
            logger.warning("Ignoring duplicate sandbox output line %d", line_number)
            continue

        transcripts[session_id] = _sanitize_transcript_value(
            {
                "schema_version": INFERENCE_TRANSCRIPT_SCHEMA_VERSION,
                "session_id": session_id,
                "final_status": envelope.get("status"),
                "agent_output": _transcript_agent_output(envelope),
                "calls": _inference_calls(envelope),
            },
            secret_values=secret_values,
        )
    return transcripts


def episode_artifact(record: dict[str, Any]) -> dict[str, Any]:
    """The replayable core of a terminal record: ledger, verdict and call trace."""

    for key, kind in (("ledger", dict), ("verdict", dict), ("call_trace", list)):
        if not isinstance(record.get(key), kind):
            raise EpisodeEmitError(f"terminal result is missing its {key}")
    return {key: record[key] for key in ("ledger", "verdict", "call_trace")}


def build_episode_artifact(
    record: dict[str, Any],
    *,
    eval_run_id: str,
    family: str,
    env_pack_sha256: str,
    inference_transcript: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap a terminal record with the identity the Backend checks on read."""

    episode = {
        "evaluation_run_id": eval_run_id,
        "task_id": record["task_id"],
        "family": family,
        "pack_sha256": env_pack_sha256,
        "session_id": record.get("session_id"),
        "episode_nonce": record.get("episode_nonce"),
        "steps": record.get("steps"),
        **episode_artifact(record),
    }
    artifact = {"schema_version": EPISODE_ARTIFACT_SCHEMA_VERSION, "episode": episode}
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
# Terminal record -> EpisodeResultEntry payload
# ---------------------------------------------------------------------------


def episode_payload(record: dict[str, Any], *, delivery_id: str) -> dict[str, Any]:
    """The verdict fields of an ``EpisodeResultEntry`` for one terminal record.

    Fails fast when the record is not bound to the claimed delivery, its
    nonce differs from the ledger's, or the verdict does not hash the ledger.
    """

    core = episode_artifact(record)
    ledger, verdict = core["ledger"], core["verdict"]
    episode = ledger.get("episode") or []
    header = episode[0] if episode and isinstance(episode[0], dict) else {}
    task_id = str(record.get("task_id"))
    if header.get("kind") != "open" or header.get("task_id") != task_id:
        raise EpisodeEmitError(f"ledger does not open task {task_id}")
    if (header.get("delivery") or {}).get("id") != delivery_id:
        raise EpisodeEmitError(f"task {task_id} ran outside delivery {delivery_id}")
    if header.get("episode_nonce") != record.get("episode_nonce"):
        raise EpisodeEmitError(f"task {task_id} ran with another episode nonce")
    ledger_sha256 = canonical.digest(episode)
    if verdict.get("ledger_sha256") != ledger_sha256 or verdict.get("task_id") != task_id:
        raise EpisodeEmitError(f"verdict for task {task_id} does not grade its ledger")
    return {
        "task_id": task_id,
        # Backend accepts at most six decimal places.
        "score": format(Decimal(repr(float(verdict["score"]))).quantize(Decimal("0.000001")), "f"),
        "category": verdict["category"],
        "flags": list(verdict["flags"]),
        "ledger_sha256": ledger_sha256,
        "verdict_sha256": canonical.digest(verdict),
        "episode_nonce": record["episode_nonce"],
    }


async def upload_episode_artifact(
    client: httpx.AsyncClient,
    *,
    backend: Client,
    artifact: dict[str, Any],
    download_url_rewriter: Callable[[str], str] | None = None,
) -> str:
    """Presign and upload one content-addressed episode artifact."""

    body = serialize_episode_artifact(artifact)
    artifact_sha256 = episode_artifact_sha256(body)
    request = EpisodeArtifactPresignRequest.from_dict(
        {
            "eval_run_id": artifact["episode"]["evaluation_run_id"],
            "env_pack_sha256": artifact["episode"]["pack_sha256"],
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
    eval_run_id: str,
    delivery_id: str,
    task_families: Mapping[str, str],
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
            for record in results:
                payload = episode_payload(record, delivery_id=delivery_id)
                family = task_families.get(payload["task_id"])
                if family is None:
                    raise EpisodeEmitError(f"task {payload['task_id']} is not on the run roster")
                transcript = (
                    inference_transcripts.get(str(record.get("session_id")))
                    if inference_transcripts is not None
                    else None
                )
                artifact_uri = await upload_episode_artifact(
                    client,
                    backend=backend,
                    artifact=build_episode_artifact(
                        record,
                        eval_run_id=eval_run_id,
                        family=family,
                        env_pack_sha256=env_pack_sha256,
                        inference_transcript=transcript,
                    ),
                    download_url_rewriter=download_url_rewriter,
                )
                payloads.append(
                    {
                        "eval_run_id": eval_run_id,
                        "env_pack_sha256": env_pack_sha256,
                        "family": family,
                        # A world-pack verdict always grades a closed episode;
                        # Backend reads failure from the score and category.
                        "outcome": "completed",
                        "ledger_uri": artifact_uri,
                        "step_count": int(record.get("steps") or 0),
                        **payload,
                    }
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
# Replay from recorded readings
# ---------------------------------------------------------------------------


def _recorded_reader(readings: list[dict[str, Any]]) -> tuple[Callable[[str, dict], dict], list]:
    """Serve recorded readings in order; the returned queue holds the unused ones."""

    queue = list(readings)

    def read(service: str, request: dict) -> dict:
        if not queue:
            raise ReplayMismatch("a needed reading is missing")
        reading = queue.pop(0)
        if reading.get("service") != service or canonical.dumps(reading.get("request")) != canonical.dumps(request):
            raise ReplayMismatch("the replayed request differs from the recorded one")
        return reading["result"]

    return read, queue


def regrade(pack: Pack, ledger: dict[str, Any]) -> dict[str, Any]:
    """Re-grade a recorded ledger offline; raise ReplayMismatch on a reading divergence."""

    if ledger["episode"][0].get("manifest_sha256") != pack.manifest_sha256:
        raise ReplayMismatch("the ledger was recorded on another pack")
    read, unused = _recorded_reader(ledger.get("grading") or [])
    verdict = grade_episode(pack, {**ledger, "grading": []}, read)
    if unused:
        raise ReplayMismatch("recorded readings were not all used")
    return verdict


def replay_matches(pack: Pack, record: dict[str, Any]) -> bool:
    """True iff re-grading the record's ledger reproduces its verdict exactly."""

    try:
        verdict = regrade(pack, record["ledger"])
    except (ReplayMismatch, KeyError, IndexError, TypeError):
        return False
    return canonical.dumps(verdict) == canonical.dumps(record["verdict"])


__all__ = [
    "EPISODE_ARTIFACT_SCHEMA_VERSION",
    "EpisodeEmitError",
    "ReplayMismatch",
    "build_episode_artifact",
    "emit_finalized_results",
    "episode_artifact",
    "episode_artifact_sha256",
    "episode_payload",
    "load_episode_artifact",
    "load_inference_transcripts",
    "regrade",
    "replay_matches",
    "serialize_episode_artifact",
    "submit_episode_batch",
    "upload_episode_artifact",
]
