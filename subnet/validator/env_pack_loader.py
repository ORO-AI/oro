"""Fetch and pre-flight sealed environment packs for the validator.

The pack and all private task truth remain in validator-owned storage. This
module deliberately stops at returning a validated handle; assignment/session
registry integration belongs to the validator orchestration layer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import tarfile
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import httpx
from bittensor_auth import generate_auth_headers

from oro_env_runtime.contracts import (
    ENV_CONTRACT_VERSION,
    RESULT_SCHEMA_VERSION,
    RUNTIME_VERSION,
    TOOL_CONTRACT_VERSION,
    VERIFIER_VERSION,
)
from oro_env_runtime.delivery import DeliverySubsetError, validate_delivery_binding
from oro_env_runtime.pack import sha256_file
from oro_env_runtime.runtime import (
    TaskSession,
    cache_validated_epoch,
    evict_epoch_resources,
)
from oro_env_runtime.schema import TaskSpec
from oro_env_runtime.validation import validate_epoch

logger = logging.getLogger(__name__)

PACK_VERSION_IDENTITIES = {
    "contract_version": ENV_CONTRACT_VERSION,
    "runtime_version": RUNTIME_VERSION,
    "tool_contract_version": TOOL_CONTRACT_VERSION,
    "verifier_version": VERIFIER_VERSION,
    "result_schema_version": RESULT_SCHEMA_VERSION,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMPACT_TIMEZONE_RE = re.compile(r"([+-]\d{2})(\d{2})$")
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
MAX_ARTIFACT_SIZE_BYTES = 4 * 1024 * 1024 * 1024
MAX_EXTRACTED_SIZE_BYTES = 4 * MAX_ARTIFACT_SIZE_BYTES
# Bumped from v1 → v2 when ``artifact_size_bytes`` was renamed to
# ``download_url_size_bytes``. Downstream metrics consumers keyed on the old
# field name should follow the schema-version pin.
PACK_LOAD_METRICS_SCHEMA_VERSION = "oro.validator.pack_load.v2"
# The dedup key is the delivered-bytes SHA, not the parent pack SHA.
# Qualifying + race sub-archives share a parent pack_sha256 but ship distinct
# byte streams; keying by parent would let the second (distinct) archive skip
# ``validate_epoch`` on a stale cache hit.
_VALIDATED_PACKS: set[str] = set()
_VALIDATED_PACKS_LOCK = threading.Lock()


@dataclass
class LoadedPack:
    """A validated pack whose private contents are owned by the validator."""

    pack_dir: Path
    manifest: dict[str, Any] = field(repr=False)
    task_specs: list[TaskSpec] = field(repr=False)
    task_ids: list[str]
    pack_sha256: str
    metadata: dict[str, Any] = field(repr=False)
    _scratch_dir: Path = field(repr=False)

    def open_session(self, task_id: str, *, state_blind: bool = False) -> TaskSession:
        """Open a runtime session only after pack pre-flight has succeeded."""

        if task_id not in self.task_ids:
            raise KeyError(f"unknown task_id {task_id!r}")
        return TaskSession(self.pack_dir, task_id, state_blind=state_blind)

    def close(self) -> None:
        """Remove the validator-local extracted pack and downloaded archive."""

        evict_epoch_resources(self.pack_dir)
        shutil.rmtree(self._scratch_dir, ignore_errors=True)

    def __enter__(self) -> LoadedPack:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class PackValidationError(ValueError):
    """A pack failed download, compatibility, or sealed-content validation."""


class PackCompatibilityError(PackValidationError):
    """A pack advertises a public contract version this validator cannot use."""


@contextmanager
def _record_timing(timings: dict[str, float], stage: str) -> Iterator[None]:
    """Record a monotonic stage duration, including time spent before failure."""

    started_at = time.perf_counter()
    try:
        yield
    finally:
        timings[stage] = round(time.perf_counter() - started_at, 6)


def _pack_load_metrics(
    pack_sha256: str,
    outcome: str,
    stage: str,
    timings: dict[str, float],
    metadata: dict[str, Any] | None,
    *,
    task_count: int | None = None,
    portable_validation_cache_hit: bool | None = None,
) -> dict[str, Any]:
    """Build a safe structured metric without URLs or private task contents."""

    metrics: dict[str, Any] = {
        "schema_version": PACK_LOAD_METRICS_SCHEMA_VERSION,
        "outcome": outcome,
        "stage": stage,
        "pack_sha256": pack_sha256,
        "timings_seconds": dict(timings),
    }
    if metadata is not None:
        metrics["download_url_size_bytes"] = metadata.get("download_url_size_bytes")
        metrics["declared_task_count"] = metadata.get("task_count")
    if task_count is not None:
        metrics["loaded_task_count"] = task_count
    if portable_validation_cache_hit is not None:
        metrics["portable_validation_cache_hit"] = portable_validation_cache_hit
    return metrics


def _validate_portable_once(
    archive_sha256: str, pack_dir: Path
) -> tuple[dict[str, Any], bool]:
    """Validate identical archive bytes once per validator process.

    Keyed by the delivered-bytes ``archive_sha256`` (the response's
    ``download_url_sha256``), not the parent ``pack_sha256``. The qualifying
    sub-archive and every race sub-archive derived from the same pack share
    the parent sha but are distinct byte streams — each must be independently
    validated at the loader stage, not silently skipped on a stale cache hit.
    """

    # The caller verifies the downloaded archive SHA before reaching this point.
    # Holding the lock through validation prevents duplicate work when the same
    # archive is claimed concurrently; distinct archives are rare and validation
    # is CPU-bound.
    with _VALIDATED_PACKS_LOCK:
        if archive_sha256 in _VALIDATED_PACKS:
            return {"status": "pass"}, True
        validation = validate_epoch(pack_dir)
        if validation.get("status") == "pass":
            _VALIDATED_PACKS.add(archive_sha256)
        return validation, False


def _parse_expiry(value: Any) -> datetime:
    if not isinstance(value, str):
        raise PackValidationError("download_url_expires_at must be an ISO-8601 string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    normalized = _COMPACT_TIMEZONE_RE.sub(r"\1:\2", normalized)
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise PackValidationError(
            "download_url_expires_at is not valid ISO-8601"
        ) from exc
    if parsed.tzinfo is None:
        raise PackValidationError("download_url_expires_at must include a timezone")
    return parsed.astimezone(timezone.utc)


def _require_metadata(metadata: Any, requested_sha256: str) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise PackValidationError("pack fetch response must be a JSON object")
    if metadata.get("pack_sha256") != requested_sha256:
        raise PackValidationError(
            "pack fetch response hash does not match the requested hash"
        )

    for field_name, expected in PACK_VERSION_IDENTITIES.items():
        actual = metadata.get(field_name)
        if actual != expected:
            raise PackCompatibilityError(
                f"incompatible {field_name}: expected {expected!r}, got {actual!r}"
            )

    download_url = metadata.get("download_url")
    if not isinstance(download_url, str) or not download_url:
        raise PackValidationError("download_url must be a non-empty string")
    if _parse_expiry(metadata.get("download_url_expires_at")) <= datetime.now(
        timezone.utc
    ):
        raise PackValidationError("pack download URL is expired")

    # The Backend now presigns a qualifying-only sub-archive rather
    # than the sealed pack tarball, so integrity is verified against
    # ``download_url_sha256`` (bytes-hash of what the URL delivers) instead of
    # ``pack_sha256`` (parent pack identity, still returned for join keys).
    download_url_sha256 = metadata.get("download_url_sha256")
    if not isinstance(download_url_sha256, str) or not _SHA256_RE.fullmatch(
        download_url_sha256
    ):
        raise PackValidationError(
            "download_url_sha256 must be a 64-character hex sha256"
        )
    artifact_size = metadata.get("download_url_size_bytes")
    if (
        not isinstance(artifact_size, int)
        or isinstance(artifact_size, bool)
        or artifact_size < 1
        or artifact_size > MAX_ARTIFACT_SIZE_BYTES
    ):
        raise PackValidationError(
            "download_url_size_bytes must be a positive integer no greater than "
            f"{MAX_ARTIFACT_SIZE_BYTES}"
        )
    if metadata.get("artifact_signature") is not None:
        raise PackValidationError(
            "qualifying delivery must not reuse the parent artifact_signature"
        )
    _require_delivery_roster(metadata, "qualifying")
    return metadata


def _require_delivery_roster(metadata: dict[str, Any], scope: str) -> None:
    if metadata.get("delivery_scope") != scope:
        raise PackValidationError(f"delivery_scope must be {scope!r}")
    task_ids = metadata.get("delivery_task_ids")
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or any(not isinstance(task_id, str) or not task_id for task_id in task_ids)
        or len(task_ids) != len(set(task_ids))
    ):
        raise PackValidationError("delivery_task_ids must be unique nonempty strings")
    task_count = metadata.get("task_count")
    if (
        not isinstance(task_count, int)
        or isinstance(task_count, bool)
        or task_count < 1
    ):
        raise PackValidationError("task_count must be a positive integer")
    if task_count != len(task_ids):
        raise PackValidationError(
            "task_count must equal the authorized delivery roster size"
        )
    family_counts = metadata.get("family_counts")
    if not isinstance(family_counts, dict) or any(
        not isinstance(name, str)
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count < 0
        for name, count in family_counts.items()
    ):
        raise PackValidationError(
            "family_counts must map family names to non-negative integers"
        )
    if sum(family_counts.values()) != task_count:
        raise PackValidationError("family_counts must sum to task_count")


def _require_race_metadata(
    metadata: Any,
    requested_race_id: str,
    requested_pack_sha256: str,
) -> dict[str, Any]:
    """Schema-validate ``POST /v1/validator/race/{race_id}/pack`` response.

    Same download-URL fields + integrity hash as :func:`_require_metadata`.
    The response also echoes the parent pack's contract
    identity so ``SessionRegistry._provenance()`` finds real values (not
    ``None``) when it walks ``loaded_pack.metadata`` for race results. The
    race archive is compiled from the parent pack, so it inherits the
    parent's contract pins — fast-fail here on any drift.
    """
    if not isinstance(metadata, dict):
        raise PackValidationError("race pack fetch response must be a JSON object")
    if str(metadata.get("race_id")) != requested_race_id:
        raise PackValidationError(
            "race pack fetch response race_id does not match request"
        )
    if metadata.get("pack_sha256") != requested_pack_sha256:
        raise PackValidationError(
            "race pack fetch response pack_sha256 does not match qualifying pack"
        )
    for field_name, expected in PACK_VERSION_IDENTITIES.items():
        actual = metadata.get(field_name)
        if actual != expected:
            raise PackCompatibilityError(
                f"incompatible {field_name}: expected {expected!r}, got {actual!r}"
            )
    download_url = metadata.get("download_url")
    if not isinstance(download_url, str) or not download_url:
        raise PackValidationError("download_url must be a non-empty string")
    if _parse_expiry(metadata.get("download_url_expires_at")) <= datetime.now(
        timezone.utc
    ):
        raise PackValidationError("race pack download URL is expired")
    download_url_sha256 = metadata.get("download_url_sha256")
    if not isinstance(download_url_sha256, str) or not _SHA256_RE.fullmatch(
        download_url_sha256
    ):
        raise PackValidationError(
            "download_url_sha256 must be a 64-character hex sha256"
        )
    size = metadata.get("download_url_size_bytes")
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size < 1
        or size > MAX_ARTIFACT_SIZE_BYTES
    ):
        raise PackValidationError(
            "download_url_size_bytes must be a positive integer no greater than "
            f"{MAX_ARTIFACT_SIZE_BYTES}"
        )
    _require_delivery_roster(metadata, "race")
    return metadata


async def _download(
    client: httpx.AsyncClient,
    url: str,
    destination: Path,
    expected_size: int,
) -> None:
    downloaded = 0
    async with client.stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            async for chunk in response.aiter_raw(_DOWNLOAD_CHUNK_SIZE):
                downloaded += len(chunk)
                if downloaded > expected_size:
                    raise PackValidationError(
                        "downloaded artifact exceeds declared download_url_size_bytes"
                    )
                handle.write(chunk)
    if downloaded != expected_size:
        raise PackValidationError(
            f"artifact size mismatch: expected {expected_size}, got {downloaded}"
        )


def _enforce_extracted_size_limit(size: int) -> None:
    if size > MAX_EXTRACTED_SIZE_BYTES:
        raise PackValidationError(
            "pack exceeds maximum uncompressed size of "
            f"{MAX_EXTRACTED_SIZE_BYTES} bytes"
        )


def _safe_extract(archive_path: Path, scratch_dir: Path) -> Path:
    """Extract regular files/directories under exactly one ``epoch/`` root."""

    try:
        with tarfile.open(archive_path, mode="r|gz") as archive:
            member_names: set[str] = set()
            member_count = 0
            declared_size = 0
            extracted_size = 0
            for member_count, member in enumerate(archive, 1):
                relative = PurePosixPath(member.name)
                if (
                    relative.is_absolute()
                    or not relative.parts
                    or relative.parts[0] != "epoch"
                    or ".." in relative.parts
                    or not (member.isdir() or member.isfile())
                ):
                    raise PackValidationError(f"unsafe pack member: {member.name!r}")
                normalized_name = relative.as_posix().rstrip("/")
                if normalized_name in member_names:
                    raise PackValidationError(f"duplicate pack member: {member.name!r}")
                member_names.add(normalized_name)
                if member.isfile():
                    if member.size < 0:
                        raise PackValidationError(
                            f"invalid pack member size: {member.name!r}"
                        )
                    declared_size += member.size
                    _enforce_extracted_size_limit(declared_size)

                destination = scratch_dir.joinpath(*relative.parts)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise PackValidationError(
                        f"could not read pack member: {member.name!r}"
                    )
                with source, destination.open("wb") as output:
                    while chunk := source.read(_DOWNLOAD_CHUNK_SIZE):
                        extracted_size += len(chunk)
                        _enforce_extracted_size_limit(extracted_size)
                        output.write(chunk)
            if not member_count:
                raise PackValidationError("pack tarball is empty")
    except PackValidationError:
        raise
    except (EOFError, OSError, tarfile.TarError) as exc:
        raise PackValidationError(
            f"invalid pack tarball: {type(exc).__name__}"
        ) from exc

    pack_dir = scratch_dir / "epoch"
    if not pack_dir.is_dir():
        raise PackValidationError("pack tarball does not contain an epoch directory")
    return pack_dir


def _load_validated_contents(
    archive_path: Path,
    scratch_dir: Path,
    pack_sha256: str,
    metadata: dict[str, Any],
    timings: dict[str, float],
) -> tuple[Path, dict[str, Any], list[TaskSpec], list[str], bool]:
    with _record_timing(timings, "archive_extract"):
        pack_dir = _safe_extract(archive_path, scratch_dir)
    with _record_timing(timings, "portable_validation"):
        # Deduplicate on delivered-bytes SHA, not parent pack SHA, so
        # qualifying and race sub-archives (same parent, distinct bytes) are
        # each validated independently at the loader stage.
        validation, cache_hit = _validate_portable_once(
            metadata["download_url_sha256"], pack_dir
        )
    if validation.get("status") != "pass":
        detail = json.dumps(validation, sort_keys=True, separators=(",", ":"))
        raise PackValidationError(f"sealed epoch validation failed: {detail}")

    with _record_timing(timings, "pack_contents_load"):
        try:
            manifest = json.loads((pack_dir / "manifest.json").read_text())
            rows = [
                json.loads(line)
                for line in (pack_dir / "data" / "tasks" / "private_tasks.jsonl")
                .read_text()
                .splitlines()
                if line.strip()
            ]
            task_specs = [TaskSpec.model_validate(row["task"]) for row in rows]
            task_ids = [str(row["task_id"]) for row in rows]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PackValidationError(
                f"could not load validated pack contents: {exc}"
            ) from exc

        try:
            validate_delivery_binding(
                manifest,
                task_ids,
                scope=metadata["delivery_scope"],
                scope_id=(str(metadata["race_id"]) if "race_id" in metadata else None),
                parent_pack_sha256=pack_sha256,
                expected_task_ids=metadata["delivery_task_ids"],
            )
        except DeliverySubsetError as exc:
            raise PackValidationError(f"delivery subset binding failed: {exc}") from exc

        actual_family_counts = Counter(task.family for task in task_specs)
        if actual_family_counts != Counter(metadata["family_counts"]):
            raise PackValidationError(
                "pack family counts do not match Backend metadata: "
                f"expected {metadata['family_counts']!r}, "
                f"got {dict(actual_family_counts)!r}"
            )
    return pack_dir, manifest, task_specs, task_ids, cache_hit


def load_local_pack(
    archive_path: str | Path,
    expected_sha256: str | None = None,
    *,
    scratch_root: str | Path | None = None,
) -> LoadedPack:
    """Validate and load a sealed pack already present on the local host."""

    source = Path(archive_path)
    if not source.is_file():
        raise PackValidationError(f"local pack does not exist: {source}")
    if expected_sha256 is not None and not _SHA256_RE.fullmatch(expected_sha256):
        raise PackValidationError("expected_sha256 must be 64 lowercase hex characters")

    actual_sha256 = sha256_file(source)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise PackValidationError(
            f"artifact sha256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )

    scratch_parent = None if scratch_root is None else Path(scratch_root)
    if scratch_parent is not None:
        scratch_parent.mkdir(parents=True, exist_ok=True)
    scratch_dir = Path(
        tempfile.mkdtemp(prefix=f"oro-pack-{actual_sha256[:12]}-", dir=scratch_parent)
    )
    try:
        pack_dir = _safe_extract(source, scratch_dir)
        validation = validate_epoch(pack_dir)
        if validation.get("status") != "pass":
            detail = json.dumps(validation, sort_keys=True, separators=(",", ":"))
            raise PackValidationError(f"sealed epoch validation failed: {detail}")

        manifest = json.loads((pack_dir / "manifest.json").read_text())
        rows = [
            json.loads(line)
            for line in (pack_dir / "data" / "tasks" / "private_tasks.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
        task_specs = [TaskSpec.model_validate(row["task"]) for row in rows]
        task_ids = [str(row["task_id"]) for row in rows]
        family_counts = Counter(task.family for task in task_specs)
        catalog_epochs = {task.catalog_epoch for task in task_specs}
        catalog_epoch = next(iter(catalog_epochs)) if len(catalog_epochs) == 1 else None
        search = manifest.get("search")
        metadata = {
            "pack_sha256": actual_sha256,
            "artifact_size_bytes": source.stat().st_size,
            "task_count": len(task_specs),
            "family_counts": dict(family_counts),
            **PACK_VERSION_IDENTITIES,
            "catalog_epoch": catalog_epoch,
            "catalog_sha256": manifest.get("catalog_fingerprint"),
            "search_index_epoch": None,
            "search_index_sha256": (
                search.get("index_sha256") if isinstance(search, dict) else None
            ),
        }
        cache_validated_epoch(pack_dir)
        return LoadedPack(
            pack_dir=pack_dir,
            manifest=manifest,
            task_specs=task_specs,
            task_ids=task_ids,
            pack_sha256=actual_sha256,
            metadata=metadata,
            _scratch_dir=scratch_dir,
        )
    except BaseException:
        evict_epoch_resources(scratch_dir / "epoch")
        shutil.rmtree(scratch_dir, ignore_errors=True)
        raise


async def fetch_and_validate_pack(
    pack_sha256: str,
    backend_url: str,
    validator_keypair: Any,
    *,
    scratch_root: str | Path | None = None,
    timeout: float = 60.0,
    http_client: httpx.AsyncClient | None = None,
    download_url_rewriter: Callable[[str], str] | None = None,
) -> LoadedPack | None:
    """Fetch, validate, and load a sealed pack, returning ``None`` on rejection.

    A supplied ``http_client`` remains caller-owned. The default client carries
    no persistent auth headers: only the Backend metadata request receives a
    fresh SR25519 header set, so those credentials cannot leak to the presigned
    object-store URL.
    """

    if not isinstance(pack_sha256, str) or not _SHA256_RE.fullmatch(pack_sha256):
        logger.warning("Skipping invalid pack hash %r", pack_sha256)
        return None

    scratch_dir: Path | None = None
    metadata: dict[str, Any] | None = None
    timings: dict[str, float] = {}
    portable_validation_cache_hit: bool | None = None
    current_stage = "backend_metadata"
    total_started_at = time.perf_counter()
    owned_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(timeout))
    try:
        with _record_timing(timings, current_stage):
            auth_headers = generate_auth_headers(validator_keypair)
            auth_headers["Accept-Encoding"] = "identity"
            fetch_url = f"{backend_url.rstrip('/')}/v1/validator/pack/{pack_sha256}"
            response = await client.get(fetch_url, headers=auth_headers)
            response.raise_for_status()
            metadata = _require_metadata(response.json(), pack_sha256)

        scratch_parent = None if scratch_root is None else Path(scratch_root)
        if scratch_parent is not None:
            scratch_parent.mkdir(parents=True, exist_ok=True)
        scratch_dir = Path(
            tempfile.mkdtemp(prefix=f"oro-pack-{pack_sha256[:12]}-", dir=scratch_parent)
        )
        archive_path = scratch_dir / "pack.tar.gz"
        download_url = metadata["download_url"]
        if download_url_rewriter is not None:
            download_url = download_url_rewriter(download_url)
        current_stage = "archive_download"
        with _record_timing(timings, current_stage):
            await _download(
                client,
                download_url,
                archive_path,
                metadata["download_url_size_bytes"],
            )

        current_stage = "archive_sha256"
        with _record_timing(timings, current_stage):
            actual_sha256 = await asyncio.to_thread(sha256_file, archive_path)
            expected_sha256 = metadata["download_url_sha256"]
            if actual_sha256 != expected_sha256:
                raise PackValidationError(
                    "delivered archive sha256 mismatch: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )

        current_stage = "archive_processing"
        (
            pack_dir,
            manifest,
            task_specs,
            task_ids,
            portable_validation_cache_hit,
        ) = await asyncio.to_thread(
            _load_validated_contents,
            archive_path,
            scratch_dir,
            pack_sha256,
            metadata,
            timings,
        )
        current_stage = "cache_validated_epoch"
        with _record_timing(timings, current_stage):
            await asyncio.to_thread(cache_validated_epoch, pack_dir)
        timings["total"] = round(time.perf_counter() - total_started_at, 6)
        metrics = _pack_load_metrics(
            pack_sha256,
            "loaded",
            "complete",
            timings,
            metadata,
            task_count=len(task_specs),
            portable_validation_cache_hit=portable_validation_cache_hit,
        )
        logger.info(
            "Environment pack load metrics: %s",
            json.dumps(metrics, sort_keys=True, separators=(",", ":")),
        )
        return LoadedPack(
            pack_dir=pack_dir,
            manifest=manifest,
            task_specs=task_specs,
            task_ids=task_ids,
            pack_sha256=pack_sha256,
            metadata=metadata,
            _scratch_dir=scratch_dir,
        )
    except (httpx.HTTPError, OSError, ValueError, TypeError, KeyError) as exc:
        if isinstance(exc, httpx.HTTPStatusError):
            error = f"HTTPStatusError status={exc.response.status_code}"
        elif isinstance(exc, PackCompatibilityError):
            error = f"PackValidationError: {exc}"
        else:
            error = type(exc).__name__
        if current_stage == "archive_processing" and timings:
            current_stage = next(reversed(timings))
        timings["total"] = round(time.perf_counter() - total_started_at, 6)
        metrics = _pack_load_metrics(
            pack_sha256,
            "rejected",
            current_stage,
            timings,
            metadata,
            portable_validation_cache_hit=portable_validation_cache_hit,
        )
        logger.warning(
            "Skipping sealed pack %s: %s; metrics=%s",
            pack_sha256,
            error,
            json.dumps(metrics, sort_keys=True, separators=(",", ":")),
        )
        if scratch_dir is not None:
            evict_epoch_resources(scratch_dir / "epoch")
            shutil.rmtree(scratch_dir, ignore_errors=True)
        return None
    finally:
        if owned_client:
            await client.aclose()


async def fetch_and_validate_race_pack(
    race_id: str,
    pack_sha256: str,
    backend_url: str,
    validator_keypair: Any,
    *,
    scratch_root: str | Path | None = None,
    timeout: float = 60.0,
    http_client: httpx.AsyncClient | None = None,
    download_url_rewriter: Callable[[str], str] | None = None,
) -> LoadedPack | None:
    """Fetch, validate, and extract the race-scoped sub-archive.

    Same shape as :func:`fetch_and_validate_pack` but calls the race-scoped
    endpoint (``POST /v1/validator/race/{race_id}/pack``), which the Backend
    only presigns to validators holding an active ``EvaluationRun`` on the
    race. Returns ``None`` on any validation/fetch failure so the caller can
    fail the run cleanly instead of crashing the loop.

    The returned :class:`LoadedPack` carries the race's selected task specs — a
    strict subset of the parent pack the caller previously loaded via
    :func:`fetch_and_validate_pack`. Callers pointing sessions at race task
    ids should use this pack, not the qualifying one.
    """
    if not isinstance(race_id, str) or not race_id:
        logger.warning("Skipping empty race_id")
        return None
    if not isinstance(pack_sha256, str) or not _SHA256_RE.fullmatch(pack_sha256):
        logger.warning("Skipping invalid pack hash %r", pack_sha256)
        return None

    scratch_dir: Path | None = None
    metadata: dict[str, Any] | None = None
    timings: dict[str, float] = {}
    portable_validation_cache_hit: bool | None = None
    current_stage = "backend_metadata"
    total_started_at = time.perf_counter()
    owned_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(timeout))
    try:
        with _record_timing(timings, current_stage):
            auth_headers = generate_auth_headers(validator_keypair)
            auth_headers["Accept-Encoding"] = "identity"
            fetch_url = f"{backend_url.rstrip('/')}/v1/validator/race/{race_id}/pack"
            response = await client.post(fetch_url, headers=auth_headers)
            response.raise_for_status()
            metadata = _require_race_metadata(response.json(), race_id, pack_sha256)

        scratch_parent = None if scratch_root is None else Path(scratch_root)
        if scratch_parent is not None:
            scratch_parent.mkdir(parents=True, exist_ok=True)
        scratch_dir = Path(
            tempfile.mkdtemp(prefix=f"oro-race-{race_id[:12]}-", dir=scratch_parent)
        )
        archive_path = scratch_dir / "race.tar.gz"
        download_url = metadata["download_url"]
        if download_url_rewriter is not None:
            download_url = download_url_rewriter(download_url)
        current_stage = "archive_download"
        with _record_timing(timings, current_stage):
            await _download(
                client,
                download_url,
                archive_path,
                metadata["download_url_size_bytes"],
            )

        current_stage = "archive_sha256"
        with _record_timing(timings, current_stage):
            actual_sha256 = await asyncio.to_thread(sha256_file, archive_path)
            expected_sha256 = metadata["download_url_sha256"]
            if actual_sha256 != expected_sha256:
                raise PackValidationError(
                    "race archive sha256 mismatch: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )

        current_stage = "archive_processing"
        (
            pack_dir,
            manifest,
            task_specs,
            task_ids,
            portable_validation_cache_hit,
        ) = await asyncio.to_thread(
            _load_validated_contents,
            archive_path,
            scratch_dir,
            pack_sha256,
            metadata,
            timings,
        )
        # Cache the validated epoch so the first TaskSession opened on this
        # race pack does not re-run validate_epoch (matches qualifying path).
        current_stage = "cache_validated_epoch"
        with _record_timing(timings, current_stage):
            await asyncio.to_thread(cache_validated_epoch, pack_dir)
        timings["total"] = round(time.perf_counter() - total_started_at, 6)
        metrics = _pack_load_metrics(
            pack_sha256,
            "loaded",
            "complete",
            timings,
            metadata,
            task_count=len(task_specs),
            portable_validation_cache_hit=portable_validation_cache_hit,
        )
        metrics["race_id"] = race_id
        logger.info(
            "Race pack load metrics: %s",
            json.dumps(metrics, sort_keys=True, separators=(",", ":")),
        )
        return LoadedPack(
            pack_dir=pack_dir,
            manifest=manifest,
            task_specs=task_specs,
            task_ids=task_ids,
            pack_sha256=pack_sha256,
            metadata=metadata,
            _scratch_dir=scratch_dir,
        )
    except (httpx.HTTPError, OSError, ValueError, TypeError, KeyError) as exc:
        if isinstance(exc, httpx.HTTPStatusError):
            error = f"HTTPStatusError status={exc.response.status_code}"
        elif isinstance(exc, PackCompatibilityError):
            # Now reachable: _require_race_metadata validates PACK_VERSION_IDENTITIES.
            error = f"PackValidationError: {exc}"
        else:
            error = type(exc).__name__
        if current_stage == "archive_processing" and timings:
            current_stage = next(reversed(timings))
        timings["total"] = round(time.perf_counter() - total_started_at, 6)
        metrics = _pack_load_metrics(
            pack_sha256,
            "rejected",
            current_stage,
            timings,
            metadata,
            portable_validation_cache_hit=portable_validation_cache_hit,
        )
        metrics["race_id"] = race_id
        logger.warning(
            "Skipping race pack race=%s pack=%s: %s; metrics=%s",
            race_id,
            pack_sha256,
            error,
            json.dumps(metrics, sort_keys=True, separators=(",", ":")),
        )
        if scratch_dir is not None:
            evict_epoch_resources(scratch_dir / "epoch")
            shutil.rmtree(scratch_dir, ignore_errors=True)
        return None
    finally:
        if owned_client:
            await client.aclose()


__all__ = [
    "ENV_CONTRACT_VERSION",
    "PACK_VERSION_IDENTITIES",
    "PACK_LOAD_METRICS_SCHEMA_VERSION",
    "MAX_ARTIFACT_SIZE_BYTES",
    "MAX_EXTRACTED_SIZE_BYTES",
    "RESULT_SCHEMA_VERSION",
    "LoadedPack",
    "PackValidationError",
    "fetch_and_validate_pack",
    "fetch_and_validate_race_pack",
    "load_local_pack",
]
