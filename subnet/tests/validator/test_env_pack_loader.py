"""Unit coverage for sealed environment-pack pre-flight loading."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import tarfile
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest
from oro_env_runtime.delivery import build_delivery_subset_archive

from validator import env_pack_loader
from validator.env_pack_loader import (
    PackValidationError,
    fetch_and_validate_pack,
    fetch_and_validate_race_pack,
    load_local_pack,
)

pytest_plugins = ("tests.compat_fixture",)

_REAL_VALIDATE_EPOCH = env_pack_loader.validate_epoch
_REAL_VALIDATE_DELIVERY_BINDING = env_pack_loader.validate_delivery_binding


def _run_async(test: Callable[..., Any]) -> Callable[..., None]:
    """Run an async test without adding a pytest-asyncio runtime dependency."""

    @wraps(test)
    def wrapper(*args: object, **kwargs: object) -> None:
        asyncio.run(test(*args, **kwargs))

    return wrapper


def _task_row() -> dict:
    return {
        "task_id": "TF2-retrieval_recall-1",
        "split": "private_eval",
        "runtime": {"max_steps": 30},
        "task": {
            "seed": 1,
            "catalog_epoch": "test-catalog",
            "family": "retrieval_recall",
            "family_payload": {},
            "goal_text": "Find the requested product.",
            "hard": {
                "budget": 100.0,
                "currency": "USD",
                "require_in_stock": True,
            },
            "latent_prefs": [],
            "gold_set": [{"product_id": "p1", "sku": "sku-1"}],
        },
    }


def _archive_bytes() -> bytes:
    files = {
        "epoch/manifest.json": json.dumps(
            {
                "pack_version": "test",
                "catalog_fingerprint": "1" * 64,
                "search": {"index_sha256": "2" * 64},
            }
        ).encode(),
        "epoch/data/tasks/private_tasks.jsonl": (
            json.dumps(_task_row(), sort_keys=True) + "\n"
        ).encode(),
    }
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def test_loads_local_pack_and_derives_digest_when_not_supplied(tmp_path: Path) -> None:
    artifact = _archive_bytes()
    archive_path = tmp_path / "pack.tar.gz"
    archive_path.write_bytes(artifact)

    loaded = load_local_pack(archive_path, scratch_root=tmp_path / "scratch")

    try:
        assert loaded.pack_sha256 == hashlib.sha256(artifact).hexdigest()
        assert loaded.task_ids == ["TF2-retrieval_recall-1"]
        assert loaded.metadata["task_count"] == 1
        assert loaded.metadata["family_counts"] == {"retrieval_recall": 1}
        assert loaded.metadata["catalog_epoch"] == "test-catalog"
        assert loaded.metadata["catalog_sha256"] == "1" * 64
        assert loaded.metadata["search_index_epoch"] is None
        assert loaded.metadata["search_index_sha256"] == "2" * 64
    finally:
        loaded.close()


def test_local_pack_rejects_wrong_optional_digest(tmp_path: Path) -> None:
    archive_path = tmp_path / "pack.tar.gz"
    archive_path.write_bytes(_archive_bytes())

    with pytest.raises(PackValidationError, match="artifact sha256 mismatch"):
        load_local_pack(
            archive_path,
            "0" * 64,
            scratch_root=tmp_path / "scratch",
        )


def test_local_pack_removes_scratch_data_when_validation_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "pack.tar.gz"
    archive_path.write_bytes(_archive_bytes())
    scratch_root = tmp_path / "scratch"
    monkeypatch.setattr(
        env_pack_loader,
        "validate_epoch",
        lambda _pack_dir: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        load_local_pack(archive_path, scratch_root=scratch_root)

    assert list(scratch_root.iterdir()) == []


def _metadata(pack_sha256: str, artifact: bytes, **updates: object) -> dict:
    result = {
        "pack_sha256": pack_sha256,
        "download_url": "https://objects.test/pack.tar.gz?signature=secret",
        "download_url_expires_at": (
            datetime.now(timezone.utc) + timedelta(hours=1)
        ).isoformat(),
        # The URL points at the qualifying sub-archive whose
        # bytes are what the validator hashes vs download_url_sha256. The test
        # feeds the same ``artifact`` bytes to both the URL and the sha, so
        # the round-trip is byte-exact.
        "download_url_sha256": hashlib.sha256(artifact).hexdigest(),
        "download_url_size_bytes": len(artifact),
        "artifact_signature": None,
        "delivery_scope": "qualifying",
        "delivery_task_ids": ["TF2-retrieval_recall-1"],
        "contract_version": env_pack_loader.ENV_CONTRACT_VERSION,
        "runtime_version": env_pack_loader.RUNTIME_VERSION,
        "tool_contract_version": env_pack_loader.TOOL_CONTRACT_VERSION,
        "verifier_version": env_pack_loader.VERIFIER_VERSION,
        "result_schema_version": env_pack_loader.RESULT_SCHEMA_VERSION,
        "catalog_epoch": "test-catalog",
        "catalog_sha256": "1" * 64,
        "search_index_epoch": None,
        "search_index_sha256": None,
        "task_count": 1,
        "family_counts": {"retrieval_recall": 1},
    }
    result.update(updates)
    return result


def _client(
    metadata: dict,
    artifact: bytes,
    requests: list[httpx.Request] | None = None,
    *,
    artifact_status: int = 200,
) -> httpx.AsyncClient:
    class AsyncBytes(httpx.AsyncByteStream):
        async def __aiter__(self):  # noqa: ANN201
            yield artifact

    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        if request.url.host == "backend.test":
            return httpx.Response(200, json=metadata)
        if request.url.host in ("objects.test", "host.docker.internal"):
            if artifact_status != 200:
                return httpx.Response(artifact_status)
            return httpx.Response(200, stream=AsyncBytes())
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@_run_async
async def test_rewrites_artifact_url_without_changing_backend_auth(
    tmp_path: Path,
) -> None:
    artifact = _archive_bytes()
    pack_sha256 = hashlib.sha256(artifact).hexdigest()
    requests: list[httpx.Request] = []
    metadata = _metadata(
        pack_sha256,
        artifact,
        download_url="http://localhost:4566/pack.tar.gz?signature=secret",
    )

    async with _client(metadata, artifact, requests) as client:
        loaded = await fetch_and_validate_pack(
            pack_sha256,
            "https://backend.test",
            object(),
            scratch_root=tmp_path,
            http_client=client,
            backend_transport=client._transport,
            download_url_rewriter=lambda url: url.replace(
                "http://localhost:",
                "http://host.docker.internal:",
                1,
            ),
        )

    assert loaded is not None
    loaded.close()
    assert requests[0].url.host == "backend.test"
    assert requests[1].url.host == "host.docker.internal"
    assert "X-Hotkey" not in requests[1].headers


async def _fetch_pack(
    metadata: dict,
    artifact: bytes,
    scratch_root: Path,
    requests: list[httpx.Request] | None = None,
) -> env_pack_loader.LoadedPack | None:
    async with _client(metadata, artifact, requests) as client:
        return await fetch_and_validate_pack(
            metadata["pack_sha256"],
            "https://backend.test",
            object(),
            scratch_root=scratch_root,
            http_client=client,
            backend_transport=client._transport,
        )


@pytest.fixture(autouse=True)
def _stub_auth_and_epoch_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    with env_pack_loader._VALIDATED_PACKS_LOCK:
        env_pack_loader._VALIDATED_PACKS.clear()
    monkeypatch.setattr(
        "oro_sdk.bittensor_auth._pkg_generate_auth_headers",
        lambda _keypair, **_kwargs: {
            "X-Hotkey": "test-hotkey",
            "X-Timestamp": "1",
            "X-Nonce": "nonce",
            "X-Signature": "0xsigned",
        },
    )
    monkeypatch.setattr(
        env_pack_loader, "validate_epoch", lambda _path: {"status": "pass"}
    )
    monkeypatch.setattr(
        env_pack_loader,
        "validate_delivery_binding",
        lambda *_args, **_kwargs: None,
    )


@_run_async
async def test_fetches_validates_and_loads_pack_without_leaking_auth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    artifact = _archive_bytes()
    pack_sha256 = hashlib.sha256(artifact).hexdigest()
    requests: list[httpx.Request] = []
    evicted: list[Path] = []
    monkeypatch.setattr(env_pack_loader, "evict_epoch_resources", evicted.append)
    with caplog.at_level("INFO", logger=env_pack_loader.__name__):
        async with _client(
            _metadata(pack_sha256, artifact), artifact, requests
        ) as client:
            loaded = await fetch_and_validate_pack(
                pack_sha256,
                "https://backend.test",
                object(),
                scratch_root=tmp_path,
                http_client=client,
                backend_transport=client._transport,
            )

    assert loaded is not None
    assert loaded.pack_sha256 == pack_sha256
    assert loaded.pack_dir.name == "epoch"
    assert loaded.manifest["pack_version"] == "test"
    assert loaded.task_ids == ["TF2-retrieval_recall-1"]
    assert loaded.task_specs[0].family == "retrieval_recall"
    assert requests[0].headers["X-Hotkey"] == "test-hotkey"
    assert requests[0].headers["Accept-Encoding"] == "identity"
    assert "X-Hotkey" not in requests[1].headers
    assert "signature=secret" not in repr(loaded)
    assert "metadata=" not in repr(loaded)
    assert "Find the requested product" not in repr(loaded)

    metrics_record = next(
        record
        for record in caplog.records
        if record.message.startswith("Environment pack load metrics: ")
    )
    metrics = json.loads(metrics_record.message.partition(": ")[2])
    assert metrics["schema_version"] == "oro.validator.pack_load.v2"
    assert metrics["outcome"] == "loaded"
    assert metrics["stage"] == "complete"
    assert metrics["download_url_size_bytes"] == len(artifact)
    assert metrics["declared_task_count"] == 1
    assert metrics["loaded_task_count"] == 1
    assert metrics["portable_validation_cache_hit"] is False
    assert set(metrics["timings_seconds"]) == {
        "archive_download",
        "archive_extract",
        "archive_sha256",
        "backend_metadata",
        "cache_validated_epoch",
        "pack_contents_load",
        "portable_validation",
        "total",
    }
    assert all(value >= 0 for value in metrics["timings_seconds"].values())
    assert "signature=secret" not in metrics_record.message

    scratch_dir = loaded.pack_dir.parent
    loaded.close()
    assert evicted == [loaded.pack_dir]
    assert not scratch_dir.exists()


def test_portable_validation_cache_reuses_only_successful_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = iter(({"status": "pass"}, {"status": "fail"}, {"status": "pass"}))
    validation_calls = 0

    def validate(_path: Path) -> dict[str, str]:
        nonlocal validation_calls
        validation_calls += 1
        return next(outcomes)

    monkeypatch.setattr(env_pack_loader, "validate_epoch", validate)
    first_pass = env_pack_loader._validate_portable_once("a" * 64, tmp_path)
    cached_pass = env_pack_loader._validate_portable_once("a" * 64, tmp_path)
    first_failure = env_pack_loader._validate_portable_once("b" * 64, tmp_path)
    later_pass = env_pack_loader._validate_portable_once("b" * 64, tmp_path)
    cached_later_pass = env_pack_loader._validate_portable_once("b" * 64, tmp_path)

    assert first_pass == ({"status": "pass"}, False)
    assert cached_pass == ({"status": "pass"}, True)
    assert first_failure == ({"status": "fail"}, False)
    assert later_pass == ({"status": "pass"}, False)
    assert cached_later_pass == ({"status": "pass"}, True)
    assert validation_calls == 3


@_run_async
async def test_loads_generator_compatibility_fixture_and_executes(
    compiled_epoch: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise extraction and the real sealed-epoch validator together."""

    monkeypatch.setattr(env_pack_loader, "validate_epoch", _REAL_VALIDATE_EPOCH)
    archive_path = compiled_epoch.parent / f"{compiled_epoch.name}.tar.gz"
    artifact = archive_path.read_bytes()
    pack_sha256 = hashlib.sha256(artifact).hexdigest()
    family_counts = {
        "intent_decomposition": 2,
        "retrieval_recall": 2,
        "constraint_satisfaction": 2,
        "preference_reasoning": 2,
        "ranking": 2,
        "recovery": 2,
        "justification": 2,
    }
    metadata = _metadata(
        pack_sha256,
        artifact,
        task_count=14,
        family_counts=family_counts,
        delivery_task_ids=[
            json.loads(line)["task_id"]
            for line in (compiled_epoch / "data/tasks/private_tasks.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ],
    )
    async with _client(metadata, artifact) as client:
        loaded = await fetch_and_validate_pack(
            pack_sha256,
            "https://backend.test",
            object(),
            scratch_root=tmp_path / "loaded",
            http_client=client,
            backend_transport=client._transport,
        )

    assert loaded is not None
    assert len(loaded.task_specs) == 14
    assert loaded.manifest["epoch"]["family_counts"] == family_counts
    monkeypatch.setattr(
        "oro_env_runtime.runtime.validate_epoch",
        lambda _path: pytest.fail("preflighted epoch was validated again"),
    )
    session = loaded.open_session(loaded.task_ids[0])
    assert session.task_id == loaded.task_ids[0]
    step = session.step({"name": "inspect_cart", "args": {}})
    assert isinstance(step["observation"], dict)
    assert step["error"] is None
    assert step["done"] is False
    loaded.close()


@_run_async
async def test_loads_scope_bound_delivery_and_rejects_wrong_roster(
    compiled_epoch: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(env_pack_loader, "validate_epoch", _REAL_VALIDATE_EPOCH)
    monkeypatch.setattr(
        env_pack_loader,
        "validate_delivery_binding",
        _REAL_VALIDATE_DELIVERY_BINDING,
    )
    source = (compiled_epoch.parent / f"{compiled_epoch.name}.tar.gz").read_bytes()
    parent_sha = hashlib.sha256(source).hexdigest()
    source_rows = [
        json.loads(line)
        for line in (compiled_epoch / "data/tasks/private_tasks.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    selected_by_family = {row["task"]["family"]: row["task_id"] for row in source_rows}
    selected_ids = list(selected_by_family.values())[:3]
    delivery = build_delivery_subset_archive(
        source,
        selected_ids,
        scope="qualifying",
        parent_pack_sha256=parent_sha,
    )
    metadata = _metadata(
        parent_sha,
        delivery.body,
        delivery_task_ids=selected_ids,
        task_count=3,
        family_counts=delivery.family_counts,
    )

    async with _client(metadata, delivery.body) as client:
        loaded = await fetch_and_validate_pack(
            parent_sha,
            "https://backend.test",
            object(),
            scratch_root=tmp_path / "accepted",
            http_client=client,
            backend_transport=client._transport,
        )
    assert loaded is not None
    loaded.close()

    unauthorized_id = next(
        row["task_id"] for row in source_rows if row["task_id"] not in selected_ids
    )
    metadata["delivery_task_ids"] = [*selected_ids[:-1], unauthorized_id]
    async with _client(metadata, delivery.body) as client:
        rejected = await fetch_and_validate_pack(
            parent_sha,
            "https://backend.test",
            object(),
            scratch_root=tmp_path / "rejected",
            http_client=client,
            backend_transport=client._transport,
        )
    assert rejected is None


@_run_async
async def test_rejects_content_hash_mismatch_and_removes_scratch(
    tmp_path: Path,
) -> None:
    # Integrity is verified against ``download_url_sha256`` from the
    # response, not the parent ``pack_sha256``. A response that claims the
    # bytes hash to X but actually delivers Y must still be rejected.
    artifact = _archive_bytes()
    pack_sha256 = hashlib.sha256(artifact).hexdigest()
    metadata = _metadata(pack_sha256, artifact, download_url_sha256="f" * 64)
    async with _client(metadata, artifact) as client:
        loaded = await fetch_and_validate_pack(
            pack_sha256,
            "https://backend.test",
            object(),
            scratch_root=tmp_path,
            http_client=client,
            backend_transport=client._transport,
        )

    assert loaded is None
    assert list(tmp_path.iterdir()) == []


@_run_async
async def test_rejects_bad_tarball(tmp_path: Path) -> None:
    artifact = b"not a gzip tarball"
    pack_sha256 = hashlib.sha256(artifact).hexdigest()
    async with _client(_metadata(pack_sha256, artifact), artifact) as client:
        loaded = await fetch_and_validate_pack(
            pack_sha256,
            "https://backend.test",
            object(),
            scratch_root=tmp_path,
            http_client=client,
            backend_transport=client._transport,
        )

    assert loaded is None
    assert list(tmp_path.iterdir()) == []


@_run_async
@pytest.mark.parametrize("read_error", [EOFError, tarfile.ReadError])
async def test_rejects_tar_read_errors_and_removes_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    read_error: type[Exception],
) -> None:
    artifact = _archive_bytes()
    pack_sha256 = hashlib.sha256(artifact).hexdigest()

    class CorruptArchive:
        def __enter__(self) -> CorruptArchive:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def __iter__(self):  # noqa: ANN204
            raise read_error()

    monkeypatch.setattr(
        env_pack_loader.tarfile,
        "open",
        lambda *_args, **_kwargs: CorruptArchive(),
    )
    loaded = await _fetch_pack(_metadata(pack_sha256, artifact), artifact, tmp_path)

    assert loaded is None
    assert list(tmp_path.iterdir()) == []


@_run_async
async def test_rejects_oversized_declared_artifact_before_fetch(tmp_path: Path) -> None:
    artifact = _archive_bytes()
    pack_sha256 = hashlib.sha256(artifact).hexdigest()
    requests: list[httpx.Request] = []
    metadata = _metadata(
        pack_sha256,
        artifact,
        download_url_size_bytes=env_pack_loader.MAX_ARTIFACT_SIZE_BYTES + 1,
    )
    loaded = await _fetch_pack(metadata, artifact, tmp_path, requests)

    assert loaded is None
    assert len(requests) == 1
    assert list(tmp_path.iterdir()) == []


@_run_async
@pytest.mark.parametrize(
    "updates",
    [
        {"delivery_scope": "race"},
        {"delivery_task_ids": []},
        {"delivery_task_ids": ["task-1", "task-1"], "task_count": 2},
        {"task_count": 2},
        {"family_counts": {"retrieval_recall": 2}},
    ],
)
async def test_rejects_invalid_delivery_metadata_before_fetch(
    tmp_path: Path,
    updates: dict[str, object],
) -> None:
    artifact = _archive_bytes()
    pack_sha256 = hashlib.sha256(artifact).hexdigest()
    requests: list[httpx.Request] = []

    loaded = await _fetch_pack(
        _metadata(pack_sha256, artifact, **updates), artifact, tmp_path, requests
    )

    assert loaded is None
    assert len(requests) == 1


@_run_async
async def test_rejects_oversized_uncompressed_pack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _archive_bytes()
    pack_sha256 = hashlib.sha256(artifact).hexdigest()
    monkeypatch.setattr(env_pack_loader, "MAX_EXTRACTED_SIZE_BYTES", 1)
    loaded = await _fetch_pack(_metadata(pack_sha256, artifact), artifact, tmp_path)

    assert loaded is None
    assert list(tmp_path.iterdir()) == []


@_run_async
async def test_accepts_zero_count_family_metadata(tmp_path: Path) -> None:
    artifact = _archive_bytes()
    pack_sha256 = hashlib.sha256(artifact).hexdigest()
    metadata = _metadata(
        pack_sha256,
        artifact,
        family_counts={"retrieval_recall": 1, "ranking": 0},
    )
    loaded = await _fetch_pack(metadata, artifact, tmp_path)

    assert loaded is not None
    loaded.close()


@_run_async
async def test_rejects_expired_download_url_before_fetch(tmp_path: Path) -> None:
    artifact = _archive_bytes()
    pack_sha256 = hashlib.sha256(artifact).hexdigest()
    metadata = _metadata(
        pack_sha256,
        artifact,
        download_url_expires_at=(
            datetime.now(timezone.utc) - timedelta(seconds=1)
        ).isoformat(),
    )
    requests: list[httpx.Request] = []
    async with _client(metadata, artifact, requests) as client:
        loaded = await fetch_and_validate_pack(
            pack_sha256,
            "https://backend.test",
            object(),
            scratch_root=tmp_path,
            http_client=client,
            backend_transport=client._transport,
        )

    assert loaded is None
    assert len(requests) == 1
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "expires_at",
    ["2030-01-02T03:04:05Z", "2030-01-02T03:04:05+0000"],
)
def test_parse_expiry_accepts_backend_timezone_formats(expires_at: str) -> None:
    assert env_pack_loader._parse_expiry(expires_at) == datetime(
        2030, 1, 2, 3, 4, 5, tzinfo=timezone.utc
    )


@_run_async
async def test_http_failure_does_not_log_presigned_url(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    artifact = _archive_bytes()
    pack_sha256 = hashlib.sha256(artifact).hexdigest()
    metadata = _metadata(pack_sha256, artifact)

    with caplog.at_level("WARNING", logger=env_pack_loader.__name__):
        async with _client(metadata, artifact, artifact_status=403) as client:
            loaded = await fetch_and_validate_pack(
                pack_sha256,
                "https://backend.test",
                object(),
                scratch_root=tmp_path,
                http_client=client,
                backend_transport=client._transport,
            )

    assert loaded is None
    assert "HTTPStatusError status=403" in caplog.text
    assert "signature=secret" not in caplog.text
    metrics_record = next(
        record for record in caplog.records if "; metrics=" in record.message
    )
    metrics = json.loads(metrics_record.message.partition("; metrics=")[2])
    assert metrics["schema_version"] == "oro.validator.pack_load.v2"
    assert metrics["outcome"] == "rejected"
    assert metrics["stage"] == "archive_download"
    assert set(metrics["timings_seconds"]) == {
        "archive_download",
        "backend_metadata",
        "total",
    }


@_run_async
async def test_backend_metadata_403_logs_status_and_stage(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pack_sha256 = "a" * 64
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(403, text="edge rejected request")
    )

    with caplog.at_level("WARNING", logger=env_pack_loader.__name__):
        async with httpx.AsyncClient(transport=transport) as client:
            loaded = await fetch_and_validate_pack(
                pack_sha256,
                "https://backend.test",
                object(),
                scratch_root=tmp_path,
                http_client=client,
                backend_transport=transport,
            )

    assert loaded is None
    metrics_record = next(
        record for record in caplog.records if "; metrics=" in record.message
    )
    assert "BackendError status=403" in metrics_record.message
    metrics = json.loads(metrics_record.message.partition("; metrics=")[2])
    assert metrics["outcome"] == "rejected"
    assert metrics["stage"] == "backend_metadata"


@_run_async
@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("contract_version", "oro.env.v999"),
        ("runtime_version", "99.0.0"),
        ("tool_contract_version", "other_tools"),
        ("verifier_version", "99.0.0"),
        ("result_schema_version", "v99"),
    ],
)
async def test_version_mismatch_skips_without_downloading(
    tmp_path: Path,
    field_name: str,
    bad_value: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    artifact = _archive_bytes()
    pack_sha256 = hashlib.sha256(artifact).hexdigest()
    requests: list[httpx.Request] = []
    metadata = _metadata(pack_sha256, artifact, **{field_name: bad_value})
    with caplog.at_level("WARNING", logger=env_pack_loader.__name__):
        async with _client(metadata, artifact, requests) as client:
            loaded = await fetch_and_validate_pack(
                pack_sha256,
                "https://backend.test",
                object(),
                scratch_root=tmp_path,
                http_client=client,
                backend_transport=client._transport,
            )

    assert loaded is None
    assert len(requests) == 1
    assert (
        f"PackValidationError: incompatible {field_name}: "
        f"expected {env_pack_loader.PACK_VERSION_IDENTITIES[field_name]!r}, "
        f"got {bad_value!r}"
    ) in caplog.text


@_run_async
async def test_qualifying_delivery_rejects_parent_signature(tmp_path: Path) -> None:
    artifact = _archive_bytes()
    pack_sha256 = hashlib.sha256(artifact).hexdigest()
    metadata = _metadata(pack_sha256, artifact, artifact_signature="base64-signature")

    async with _client(metadata, artifact) as client:
        loaded = await fetch_and_validate_pack(
            pack_sha256,
            "https://backend.test",
            object(),
            scratch_root=tmp_path,
            http_client=client,
            backend_transport=client._transport,
        )
    assert loaded is None


@_run_async
async def test_rejects_failed_sealed_epoch_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    artifact = _archive_bytes()
    pack_sha256 = hashlib.sha256(artifact).hexdigest()
    monkeypatch.setattr(
        env_pack_loader,
        "validate_epoch",
        lambda _path: {"status": "fail", "checksum_errors": ["bad checksum"]},
    )
    with caplog.at_level("WARNING", logger=env_pack_loader.__name__):
        async with _client(_metadata(pack_sha256, artifact), artifact) as client:
            loaded = await fetch_and_validate_pack(
                pack_sha256,
                "https://backend.test",
                object(),
                scratch_root=tmp_path,
                http_client=client,
                backend_transport=client._transport,
            )

    assert loaded is None
    assert list(tmp_path.iterdir()) == []
    metrics_record = next(
        record for record in caplog.records if "; metrics=" in record.message
    )
    metrics = json.loads(metrics_record.message.partition("; metrics=")[2])
    assert metrics["stage"] == "portable_validation"


def _race_metadata(
    race_id: str,
    pack_sha256: str,
    artifact: bytes,
    **updates: object,
) -> dict:
    """Race-scoped sub-archive fetch response. Includes the parent
    pack's contract identity so provenance flows through to race results."""
    result = {
        "race_id": race_id,
        "pack_sha256": pack_sha256,
        "download_url": "https://objects.test/race.tar.gz?signature=secret",
        "download_url_expires_at": (
            datetime.now(timezone.utc) + timedelta(hours=1)
        ).isoformat(),
        "download_url_sha256": hashlib.sha256(artifact).hexdigest(),
        "download_url_size_bytes": len(artifact),
        "contract_version": env_pack_loader.ENV_CONTRACT_VERSION,
        "runtime_version": env_pack_loader.RUNTIME_VERSION,
        "tool_contract_version": env_pack_loader.TOOL_CONTRACT_VERSION,
        "verifier_version": env_pack_loader.VERIFIER_VERSION,
        "result_schema_version": env_pack_loader.RESULT_SCHEMA_VERSION,
        "catalog_epoch": "test-catalog",
        "catalog_sha256": "1" * 64,
        "search_index_epoch": None,
        "search_index_sha256": None,
        "delivery_scope": "race",
        "delivery_task_ids": ["TF2-retrieval_recall-1"],
        "task_count": 1,
        "family_counts": {"retrieval_recall": 1},
    }
    result.update(updates)
    return result


def _race_client(
    metadata: dict,
    artifact: bytes,
    requests: list[httpx.Request] | None = None,
    *,
    artifact_status: int = 200,
) -> httpx.AsyncClient:
    class AsyncBytes(httpx.AsyncByteStream):
        async def __aiter__(self):  # noqa: ANN201
            yield artifact

    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        if request.url.host == "backend.test" and request.method == "POST":
            return httpx.Response(200, json=metadata)
        if request.url.host in ("objects.test", "host.docker.internal"):
            if artifact_status != 200:
                return httpx.Response(artifact_status)
            return httpx.Response(200, stream=AsyncBytes())
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@_run_async
async def test_race_pack_fetch_returns_loaded_pack(tmp_path: Path) -> None:
    artifact = _archive_bytes()
    pack_sha256 = hashlib.sha256(artifact).hexdigest()
    race_id = "06aa0000-0000-7000-8000-000000000000"
    metadata = _race_metadata(race_id, pack_sha256, artifact)
    async with _race_client(metadata, artifact) as client:
        loaded = await fetch_and_validate_race_pack(
            race_id,
            pack_sha256,
            "https://backend.test",
            object(),
            scratch_root=tmp_path,
            http_client=client,
            backend_transport=client._transport,
        )
    assert loaded is not None
    assert loaded.pack_sha256 == pack_sha256
    assert loaded.task_ids == ["TF2-retrieval_recall-1"]


@_run_async
async def test_race_pack_fetch_rejects_content_hash_mismatch(tmp_path: Path) -> None:
    artifact = _archive_bytes()
    pack_sha256 = hashlib.sha256(artifact).hexdigest()
    race_id = "06aa0000-0000-7000-8000-000000000001"
    metadata = _race_metadata(
        race_id, pack_sha256, artifact, download_url_sha256="f" * 64
    )
    async with _race_client(metadata, artifact) as client:
        loaded = await fetch_and_validate_race_pack(
            race_id,
            pack_sha256,
            "https://backend.test",
            object(),
            scratch_root=tmp_path,
            http_client=client,
            backend_transport=client._transport,
        )
    assert loaded is None
    assert list(tmp_path.iterdir()) == []


@_run_async
async def test_race_pack_fetch_rejects_pack_sha256_mismatch(tmp_path: Path) -> None:
    artifact = _archive_bytes()
    pack_sha256 = hashlib.sha256(artifact).hexdigest()
    race_id = "06aa0000-0000-7000-8000-000000000002"
    # Response claims a different parent pack than the caller requested — must reject.
    metadata = _race_metadata(race_id, "a" * 64, artifact)
    async with _race_client(metadata, artifact) as client:
        loaded = await fetch_and_validate_race_pack(
            race_id,
            pack_sha256,
            "https://backend.test",
            object(),
            scratch_root=tmp_path,
            http_client=client,
            backend_transport=client._transport,
        )
    assert loaded is None


@_run_async
async def test_race_pack_fetch_rejects_race_id_mismatch(tmp_path: Path) -> None:
    artifact = _archive_bytes()
    pack_sha256 = hashlib.sha256(artifact).hexdigest()
    requested_race_id = "06aa0000-0000-7000-8000-000000000003"
    metadata = _race_metadata(
        "06aa0000-0000-7000-8000-000000000099", pack_sha256, artifact
    )
    async with _race_client(metadata, artifact) as client:
        loaded = await fetch_and_validate_race_pack(
            requested_race_id,
            pack_sha256,
            "https://backend.test",
            object(),
            scratch_root=tmp_path,
            http_client=client,
            backend_transport=client._transport,
        )
    assert loaded is None
