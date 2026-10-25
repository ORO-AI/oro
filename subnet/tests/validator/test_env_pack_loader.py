"""Signed world-pack delivery loading: safe extraction, verification, binding, fetch."""

from __future__ import annotations

import asyncio
import hashlib
import io
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from subnet.tests.world_pack_fixture import archive_pack, build_world_pack, world_keyring  # noqa: F401
from validator import env_pack_loader
from validator.env_pack_loader import (
    PackLoadError,
    fetch_and_validate_pack,
    fetch_and_validate_race_pack,
    load_delivery,
)

pytestmark = pytest.mark.usefixtures("world_keyring")


def _delivery(tmp_path: Path, **kwargs) -> Path:
    return archive_pack(build_world_pack(tmp_path / "build", **kwargs), tmp_path / "delivery.tar.gz")


def test_loads_a_delivery_with_the_manifest_at_the_archive_root(tmp_path: Path) -> None:
    archive = _delivery(tmp_path)
    loaded = load_delivery(archive, expected_delivery_id="practice-1", cache_root=tmp_path / "cache")

    assert loaded.archive_sha256 == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert loaded.root == tmp_path / "cache" / loaded.archive_sha256
    assert sorted(loaded.pack.tasks) == ["t1", "t2"]
    index = loaded.pack.indexes["title"]
    assert Path(index["archive"]).is_file() and index["settings"] == {"bm25": {"k1": 0.9, "b": 0.4}}
    assert hashlib.sha256(Path(index["archive"]).read_bytes()).hexdigest() == index["id"]

    again = load_delivery(archive, expected_delivery_id=None, cache_root=tmp_path / "cache")
    assert again.root == loaded.root
    assert [p.name for p in (tmp_path / "cache").iterdir()] == [loaded.archive_sha256]


def test_keeps_only_the_most_recently_used_extractions(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    keep = env_pack_loader.MAX_CACHED_DELIVERIES
    archives = [
        archive_pack(build_world_pack(tmp_path / f"b{i}", max_steps=6 + i), tmp_path / f"d{i}.tar.gz")
        for i in range(keep + 2)
    ]
    first = load_delivery(archives[0], expected_delivery_id=None, cache_root=cache)
    for archive in archives[1:]:
        load_delivery(archives[0], expected_delivery_id=None, cache_root=cache)  # stays most recent
        last = load_delivery(archive, expected_delivery_id=None, cache_root=cache)

    cached = {p.name for p in cache.iterdir()}
    assert len(cached) == keep and {first.archive_sha256, last.archive_sha256} <= cached
    assert sorted(last.pack.tasks) == ["t1", "t2"]


def test_refuses_a_pack_signed_by_an_untrusted_key(tmp_path: Path) -> None:
    archive = _delivery(tmp_path, private_key=Ed25519PrivateKey.generate())
    with pytest.raises(PackLoadError, match="pack refused"):
        load_delivery(archive, expected_delivery_id=None, cache_root=tmp_path / "cache")
    assert list((tmp_path / "cache").iterdir()) == []  # a refused extraction is not kept


def test_refuses_the_pinned_keyring_for_a_test_signed_pack(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(env_pack_loader, "KEYRING", env_pack_loader.pinned_keyring())
    with pytest.raises(PackLoadError, match="pack refused"):
        load_delivery(_delivery(tmp_path), expected_delivery_id=None, cache_root=tmp_path / "cache")


def test_refuses_a_delivery_for_another_work_item(tmp_path: Path) -> None:
    with pytest.raises(PackLoadError, match="delivery id 'practice-1'"):
        load_delivery(_delivery(tmp_path), expected_delivery_id="race-7", cache_root=tmp_path / "cache")


def test_refuses_a_tampered_part(tmp_path: Path) -> None:
    pack_dir = build_world_pack(tmp_path / "build")
    (pack_dir / "answers/t1.json").write_text('{"task_id": "t1", "answer": {"id": "b"}}')
    archive = archive_pack(pack_dir, tmp_path / "delivery.tar.gz")
    with pytest.raises(PackLoadError, match="checksum mismatch"):
        load_delivery(archive, expected_delivery_id=None, cache_root=tmp_path / "cache")


def test_refuses_a_manifest_below_the_archive_root(tmp_path: Path) -> None:
    pack_dir = build_world_pack(tmp_path / "build")
    archive = tmp_path / "nested.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(pack_dir, arcname="epoch")
    with pytest.raises(PackLoadError, match="no manifest.json at the archive root"):
        load_delivery(archive, expected_delivery_id=None, cache_root=tmp_path / "cache")


def _hostile(tmp_path: Path, member: tarfile.TarInfo) -> Path:
    archive = archive_pack(build_world_pack(tmp_path / "build"), tmp_path / "base.tar.gz")
    hostile = tmp_path / "hostile.tar.gz"
    with tarfile.open(archive) as src, tarfile.open(hostile, "w:gz") as tar:
        for info in src.getmembers():
            tar.addfile(info, src.extractfile(info))
        tar.addfile(member, io.BytesIO(b"x") if member.isfile() else None)
    return hostile


def _member(name: str, kind: bytes = tarfile.REGTYPE, linkname: str = "") -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type, info.linkname = kind, linkname
    info.size = 1 if kind == tarfile.REGTYPE else 0
    return info


@pytest.mark.parametrize(
    "member",
    [
        _member("../escape.txt"),
        _member("world/../../escape.txt"),
        _member("/etc/escape.txt"),
        _member("world/link", tarfile.SYMTYPE, "/etc/passwd"),
        _member("world/hard", tarfile.LNKTYPE, "manifest.json"),
        _member("world/fifo", tarfile.FIFOTYPE),
        _member("world/dev", tarfile.CHRTYPE),
        _member("manifest.json"),  # a duplicate member
    ],
    ids=["dotdot", "nested-dotdot", "absolute", "symlink", "hardlink", "fifo", "device", "duplicate"],
)
def test_refuses_unsafe_archive_members(tmp_path: Path, member: tarfile.TarInfo) -> None:
    with pytest.raises(PackLoadError, match="unsafe archive member|duplicate archive member"):
        load_delivery(_hostile(tmp_path, member), expected_delivery_id=None, cache_root=tmp_path / "cache")
    assert not (tmp_path / "escape.txt").exists()
    assert list((tmp_path / "cache").iterdir()) == []


def test_refuses_a_corrupt_archive(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar.gz"
    archive.write_bytes(b"not a tarball")
    with pytest.raises(PackLoadError, match="invalid archive"):
        load_delivery(archive, expected_delivery_id=None, cache_root=tmp_path / "cache")


# ---------------------------------------------------------------- fetch


@pytest.fixture(autouse=True)
def _stub_backend_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "oro_sdk.bittensor_auth._pkg_generate_auth_headers",
        lambda _keypair, **_kwargs: {"X-Hotkey": "test-hotkey", "X-Timestamp": "1", "X-Nonce": "n", "X-Signature": "0xs"},
    )


RACE_ID = "06aa0000-0000-7000-8000-000000000000"
PACK_SHA = "c" * 64


def _metadata(artifact: bytes, **updates: object) -> dict:
    # The Backend's contract-8 reply (race_id is the race fetch's).
    metadata = {
        "pack_sha256": PACK_SHA,
        "race_id": RACE_ID,
        "download_url": "https://objects.test/delivery.tar.gz?signature=secret",
        "download_url_expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        "download_url_sha256": hashlib.sha256(artifact).hexdigest(),
        "download_url_size_bytes": len(artifact),
        "artifact_signature": None,
        "pack_format": 1,
        "runtime_contract": 8,
        "world_id": "world-1",
        "result_schema_version": "1",
        "catalog_sha256": "1" * 64,
        "delivery_scope": "qualifying",
        "delivery_id": "practice-1",
        "delivery_task_ids": ["t1", "t2"],
        "task_count": 2,
        "family_counts": {"x": 2},
    }
    metadata.update(updates)
    return metadata


def _client(metadata: dict, artifact: bytes, requests: list[httpx.Request]) -> httpx.AsyncClient:
    class AsyncBytes(httpx.AsyncByteStream):
        async def __aiter__(self):  # noqa: ANN204
            yield artifact

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "backend.test":
            return httpx.Response(200, json=metadata)
        if request.url.host in ("objects.test", "host.docker.internal"):
            return httpx.Response(200, stream=AsyncBytes())
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _fetch(tmp_path: Path, metadata: dict, artifact: bytes, *, race: bool = False, **kwargs):
    if race:
        metadata = {**metadata, "delivery_scope": "race"}
    requests: list[httpx.Request] = []

    async def run():
        async with _client(metadata, artifact, requests) as client:
            common = dict(cache_root=tmp_path / "cache", http_client=client, backend_transport=client._transport, **kwargs)
            if race:
                return await fetch_and_validate_race_pack(RACE_ID, PACK_SHA, "https://backend.test", object(), **common)
            return await fetch_and_validate_pack(PACK_SHA, "https://backend.test", object(), **common)

    return asyncio.run(run()), requests


@pytest.mark.parametrize("race", [False, True])
def test_fetches_and_loads_a_delivery_without_leaking_auth(tmp_path: Path, race: bool) -> None:
    artifact = _delivery(tmp_path).read_bytes()
    loaded, requests = _fetch(
        tmp_path, _metadata(artifact, download_url="http://localhost:4566/d.tar.gz?signature=secret"), artifact,
        race=race, expected_delivery_id="practice-1",
        download_url_rewriter=lambda url: url.replace("http://localhost:", "http://host.docker.internal:", 1),
    )

    assert loaded is not None and sorted(loaded.pack.tasks) == ["t1", "t2"]
    assert requests[0].headers["X-Hotkey"] == "test-hotkey"
    assert requests[1].url.host == "host.docker.internal"
    assert "X-Hotkey" not in requests[1].headers


@pytest.mark.parametrize(
    ("updates", "race"),
    [
        ({"download_url_sha256": "f" * 64}, False),
        ({"download_url_expires_at": "2020-01-01T00:00:00+00:00"}, False),
        ({"pack_sha256": "a" * 64}, False),
        ({"race_id": "06aa0000-0000-7000-8000-000000000099"}, True),
        ({"download_url_size_bytes": 0}, False),
    ],
    ids=["sha-mismatch", "expired", "pack-mismatch", "race-mismatch", "bad-size"],
)
def test_fetch_rejects_bad_metadata(tmp_path: Path, caplog, updates: dict, race: bool) -> None:
    artifact = _delivery(tmp_path).read_bytes()
    loaded, _ = _fetch(tmp_path, _metadata(artifact, **updates), artifact, race=race)
    assert loaded is None
    assert "signature=secret" not in caplog.text


def test_fetch_rejects_the_wrong_delivery(tmp_path: Path) -> None:
    artifact = _delivery(tmp_path).read_bytes()
    loaded, _ = _fetch(tmp_path, _metadata(artifact), artifact, expected_delivery_id="race-1")
    assert loaded is None

