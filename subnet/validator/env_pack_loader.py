"""Fetch, verify and load signed world-pack deliveries for the validator.

A delivery is a tar archive with ``manifest.json`` at its root. It is extracted
safely into a cache keyed by the archive's sha256, then verified and parsed by
the runtime's ``pack.load_pack`` against :data:`KEYRING`. The delivery id the
manifest declares must match the work item.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import shutil
import tarfile
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID

import httpx
from oro_env_runtime.pack import Keyring, Pack, PackRefused, load_pack, pinned_keyring
from oro_sdk.api.validator import get_pack, get_race_pack
from oro_sdk.models.pack_fetch_response import PackFetchResponse
from oro_sdk.models.race_pack_fetch_response import RacePackFetchResponse

from .backend_client import BackendError
from .env_backend import call_environment_api, environment_client

logger = logging.getLogger(__name__)

# Public keys a delivery's signature must verify against. The default is the
# runtime's pinned ``keys.json``: trusting a new generator key takes a runtime
# release that adds it there. Tests and local harnesses replace this global.
KEYRING: Keyring = pinned_keyring()

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMPACT_TIMEZONE_RE = re.compile(r"([+-]\d{2})(\d{2})$")
_CHUNK_SIZE = 1024 * 1024
MAX_ARTIFACT_SIZE_BYTES = 4 * 1024 * 1024 * 1024
MAX_EXTRACTED_SIZE_BYTES = 4 * MAX_ARTIFACT_SIZE_BYTES
DEFAULT_CACHE_ROOT = Path(tempfile.gettempdir()) / "oro-deliveries"


class PackLoadError(ValueError):
    """A delivery failed download, extraction, verification or binding."""


@dataclass(frozen=True)
class LoadedDelivery:
    """A verified pack and the cached directory it was loaded from."""

    pack: Pack
    root: Path
    archive_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive_path: Path, destination: Path) -> None:
    """Extract regular files and directories only; no absolute paths, ``..`` or links."""

    names: set[str] = set()
    size = 0
    try:
        with tarfile.open(archive_path, mode="r|*") as archive:
            for member in archive:
                relative = PurePosixPath(member.name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not (member.isdir() or member.isfile())
                ):
                    raise PackLoadError(f"unsafe archive member: {member.name!r}")
                name = relative.as_posix()
                if name in names:
                    raise PackLoadError(f"duplicate archive member: {member.name!r}")
                names.add(name)
                target = destination.joinpath(*relative.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                size += member.size
                if size > MAX_EXTRACTED_SIZE_BYTES:
                    raise PackLoadError(
                        f"archive exceeds {MAX_EXTRACTED_SIZE_BYTES} extracted bytes"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise PackLoadError(f"could not read archive member: {member.name!r}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, _CHUNK_SIZE)
    except (EOFError, OSError, tarfile.TarError) as exc:
        raise PackLoadError(f"invalid archive: {type(exc).__name__}") from exc
    if not names:
        raise PackLoadError("archive is empty")


def load_delivery(
    archive: Path,
    *,
    expected_delivery_id: str | None,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> LoadedDelivery:
    """Verify and load a delivery archive, reusing its extraction by sha256."""

    archive, cache_root = Path(archive), Path(cache_root)
    try:
        archive_sha256 = _sha256_file(archive)
    except OSError as exc:
        raise PackLoadError(f"cannot read delivery archive: {exc}") from exc
    root = cache_root / archive_sha256
    if not root.is_dir():
        cache_root.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix=f".{archive_sha256[:12]}-", dir=cache_root))
        try:
            _safe_extract(archive, scratch)
            scratch.rename(root)
        except OSError:
            if not root.is_dir():  # a concurrent load may have won the rename
                raise
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    if not (root / "manifest.json").is_file():
        raise PackLoadError("delivery has no manifest.json at the archive root")
    try:
        pack = load_pack(root, KEYRING)
    except (PackRefused, OSError, ValueError, KeyError, TypeError) as exc:
        shutil.rmtree(root, ignore_errors=True)  # never reuse a refused extraction
        raise PackLoadError(f"pack refused: {exc}") from exc
    delivery_id = (pack.manifest.get("delivery") or {}).get("id")
    if expected_delivery_id is not None and delivery_id != expected_delivery_id:
        raise PackLoadError(
            f"delivery id {delivery_id!r} does not match the work item's "
            f"{expected_delivery_id!r}"
        )
    return LoadedDelivery(pack=pack, root=root, archive_sha256=archive_sha256)


def _parse_expiry(value: Any) -> datetime:
    if not isinstance(value, str):
        raise PackLoadError("download_url_expires_at must be an ISO-8601 string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    normalized = _COMPACT_TIMEZONE_RE.sub(r"\1:\2", normalized)
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise PackLoadError("download_url_expires_at is not valid ISO-8601") from exc
    if parsed.tzinfo is None:
        raise PackLoadError("download_url_expires_at must include a timezone")
    return parsed.astimezone(timezone.utc)


def _require_download(metadata: Any, echoes: dict[str, str]) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise PackLoadError("pack fetch response must be a JSON object")
    for key, expected in echoes.items():
        if str(metadata.get(key)) != expected:
            raise PackLoadError(f"pack fetch response {key} does not match the request")
    if not isinstance(metadata.get("download_url"), str) or not metadata["download_url"]:
        raise PackLoadError("download_url must be a non-empty string")
    if _parse_expiry(metadata.get("download_url_expires_at")) <= datetime.now(timezone.utc):
        raise PackLoadError("pack download URL is expired")
    sha256 = metadata.get("download_url_sha256")
    if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256):
        raise PackLoadError("download_url_sha256 must be a 64-character hex sha256")
    size = metadata.get("download_url_size_bytes")
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or not 1 <= size <= MAX_ARTIFACT_SIZE_BYTES
    ):
        raise PackLoadError(
            "download_url_size_bytes must be a positive integer no greater than "
            f"{MAX_ARTIFACT_SIZE_BYTES}"
        )
    return metadata


async def _download(
    client: httpx.AsyncClient, url: str, destination: Path, expected_size: int
) -> None:
    downloaded = 0
    async with client.stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            async for chunk in response.aiter_raw(_CHUNK_SIZE):
                downloaded += len(chunk)
                if downloaded > expected_size:
                    raise PackLoadError(
                        "downloaded artifact exceeds declared download_url_size_bytes"
                    )
                handle.write(chunk)
    if downloaded != expected_size:
        raise PackLoadError(
            f"artifact size mismatch: expected {expected_size}, got {downloaded}"
        )


async def _fetch_delivery(
    label: str,
    echoes: dict[str, str],
    fetch_metadata: Callable[[Any], Any],
    backend_url: str,
    validator_keypair: Any,
    *,
    expected_delivery_id: str | None,
    cache_root: Path,
    timeout: float,
    http_client: httpx.AsyncClient | None,
    backend_transport: httpx.AsyncBaseTransport | None,
    download_url_rewriter: Callable[[str], str] | None,
) -> LoadedDelivery | None:
    started = time.perf_counter()
    owned_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(timeout))
    try:
        async with environment_client(
            backend_url, validator_keypair, timeout=timeout, transport=backend_transport
        ) as backend:
            response = await fetch_metadata(backend)
        # Validate the delivered wire fields, not SDK-filled defaults.
        metadata = _require_download(json.loads(response.content), echoes)
        url = metadata["download_url"]
        if download_url_rewriter is not None:
            url = download_url_rewriter(url)
        with tempfile.TemporaryDirectory(prefix="oro-download-") as scratch:
            archive = Path(scratch) / "delivery.tar"
            await _download(client, url, archive, metadata["download_url_size_bytes"])
            actual = await asyncio.to_thread(_sha256_file, archive)
            if actual != metadata["download_url_sha256"]:
                raise PackLoadError(
                    f"delivered archive sha256 mismatch: expected "
                    f"{metadata['download_url_sha256']}, got {actual}"
                )
            loaded = await asyncio.to_thread(
                load_delivery,
                archive,
                expected_delivery_id=expected_delivery_id,
                cache_root=cache_root,
            )
        logger.info(
            "Loaded %s delivery %s (%d tasks) in %.3fs",
            label, loaded.archive_sha256, len(loaded.pack.tasks), time.perf_counter() - started,
        )
        return loaded
    except (BackendError, httpx.HTTPError, OSError, ValueError, TypeError, KeyError) as exc:
        if isinstance(exc, httpx.HTTPStatusError):
            error = f"HTTPStatusError status={exc.response.status_code}"
        elif isinstance(exc, BackendError):
            error = f"BackendError status={exc.status_code}"
        elif isinstance(exc, PackLoadError):
            error = f"PackLoadError: {exc}"
        else:
            error = type(exc).__name__
        logger.warning("Skipping %s delivery: %s", label, error)
        return None
    finally:
        if owned_client:
            await client.aclose()


async def fetch_and_validate_pack(
    pack_sha256: str,
    backend_url: str,
    validator_keypair: Any,
    *,
    expected_delivery_id: str | None = None,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    timeout: float = 60.0,
    http_client: httpx.AsyncClient | None = None,
    backend_transport: httpx.AsyncBaseTransport | None = None,
    download_url_rewriter: Callable[[str], str] | None = None,
) -> LoadedDelivery | None:
    """Fetch and load the pack's qualifying delivery; ``None`` on any rejection."""

    if not isinstance(pack_sha256, str) or not _SHA256_RE.fullmatch(pack_sha256):
        logger.warning("Skipping invalid pack hash %r", pack_sha256)
        return None
    return await _fetch_delivery(
        f"pack {pack_sha256}",
        {"pack_sha256": pack_sha256},
        lambda backend: call_environment_api(
            get_pack.asyncio_detailed, PackFetchResponse,
            operation="pack fetch", client=backend, pack_sha256=pack_sha256,
        ),
        backend_url, validator_keypair,
        expected_delivery_id=expected_delivery_id, cache_root=cache_root, timeout=timeout,
        http_client=http_client, backend_transport=backend_transport,
        download_url_rewriter=download_url_rewriter,
    )


async def fetch_and_validate_race_pack(
    race_id: str,
    pack_sha256: str,
    backend_url: str,
    validator_keypair: Any,
    *,
    expected_delivery_id: str | None = None,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    timeout: float = 60.0,
    http_client: httpx.AsyncClient | None = None,
    backend_transport: httpx.AsyncBaseTransport | None = None,
    download_url_rewriter: Callable[[str], str] | None = None,
) -> LoadedDelivery | None:
    """Fetch and load a race delivery, presigned only to validators running the race."""

    if not isinstance(race_id, str) or not race_id:
        logger.warning("Skipping empty race_id")
        return None
    return await _fetch_delivery(
        f"race {race_id}",
        {"race_id": race_id, "pack_sha256": pack_sha256},
        lambda backend: call_environment_api(
            get_race_pack.asyncio_detailed, RacePackFetchResponse,
            operation="assigned pack fetch", client=backend, race_id=UUID(race_id),
        ),
        backend_url, validator_keypair,
        expected_delivery_id=expected_delivery_id, cache_root=cache_root, timeout=timeout,
        http_client=http_client, backend_transport=backend_transport,
        download_url_rewriter=download_url_rewriter,
    )


__all__ = [
    "KEYRING",
    "MAX_ARTIFACT_SIZE_BYTES",
    "MAX_EXTRACTED_SIZE_BYTES",
    "LoadedDelivery",
    "PackLoadError",
    "fetch_and_validate_pack",
    "fetch_and_validate_race_pack",
    "load_delivery",
]
