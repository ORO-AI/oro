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

from validator import env_pack_loader
from validator.env_pack_loader import fetch_and_validate_pack

pytest_plugins = ("tests.compat_fixture",)

_REAL_VALIDATE_EPOCH = env_pack_loader.validate_epoch


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
        "epoch/manifest.json": json.dumps({"pack_version": "test"}).encode(),
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


def _metadata(pack_sha256: str, artifact: bytes, **updates: object) -> dict:
    result = {
        "pack_sha256": pack_sha256,
        "download_url": "https://objects.test/pack.tar.gz?signature=secret",
        "download_url_expires_at": (
            datetime.now(timezone.utc) + timedelta(hours=1)
        ).isoformat(),
        "artifact_size_bytes": len(artifact),
        "artifact_signature": None,
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
        )


@pytest.fixture(autouse=True)
def _stub_auth_and_epoch_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    with env_pack_loader._VALIDATED_PACKS_LOCK:
        env_pack_loader._VALIDATED_PACKS.clear()
    monkeypatch.setattr(
        env_pack_loader,
        "generate_auth_headers",
        lambda _keypair: {
            "X-Hotkey": "test-hotkey",
            "X-Timestamp": "1",
            "X-Nonce": "nonce",
            "X-Signature": "0xsigned",
        },
    )
    monkeypatch.setattr(env_pack_loader, "validate_epoch", lambda _path: {"status": "pass"})


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
        async with _client(_metadata(pack_sha256, artifact), artifact, requests) as client:
            loaded = await fetch_and_validate_pack(
                pack_sha256,
                "https://backend.test",
                object(),
                scratch_root=tmp_path,
                http_client=client,
            )

    assert loaded is not None
    assert loaded.pack_sha256 == pack_sha256
    assert loaded.pack_dir.name == "epoch"
    assert loaded.manifest == {"pack_version": "test"}
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
    assert metrics["schema_version"] == "oro.validator.pack_load.v1"
    assert metrics["outcome"] == "loaded"
    assert metrics["stage"] == "complete"
    assert metrics["artifact_size_bytes"] == len(artifact)
    assert metrics["declared_task_count"] == 1
    assert metrics["loaded_task_count"] == 1
    assert metrics["portable_validation_cache_hit"] is False
    assert set(metrics["timings_seconds"]) == {
        "archive_download",
        "archive_extract",
        "archive_sha256",
        "artifact_signature",
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
    outcomes = iter(
        ({"status": "pass"}, {"status": "fail"}, {"status": "pass"})
    )
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
    )
    async with _client(metadata, artifact) as client:
        loaded = await fetch_and_validate_pack(
            pack_sha256,
            "https://backend.test",
            object(),
            scratch_root=tmp_path / "loaded",
            http_client=client,
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
async def test_rejects_content_hash_mismatch_and_removes_scratch(tmp_path: Path) -> None:
    artifact = _archive_bytes()
    requested_sha256 = "0" * 64
    async with _client(_metadata(requested_sha256, artifact), artifact) as client:
        loaded = await fetch_and_validate_pack(
            requested_sha256,
            "https://backend.test",
            object(),
            scratch_root=tmp_path,
            http_client=client,
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
        artifact_size_bytes=env_pack_loader.MAX_ARTIFACT_SIZE_BYTES + 1,
    )
    loaded = await _fetch_pack(metadata, artifact, tmp_path, requests)

    assert loaded is None
    assert len(requests) == 1
    assert list(tmp_path.iterdir()) == []


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
            )

    assert loaded is None
    assert "HTTPStatusError status=403" in caplog.text
    assert "signature=secret" not in caplog.text
    metrics_record = next(
        record for record in caplog.records if "; metrics=" in record.message
    )
    metrics = json.loads(metrics_record.message.partition("; metrics=")[2])
    assert metrics["schema_version"] == "oro.validator.pack_load.v1"
    assert metrics["outcome"] == "rejected"
    assert metrics["stage"] == "archive_download"
    assert set(metrics["timings_seconds"]) == {
        "archive_download",
        "backend_metadata",
        "total",
    }


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
            )

    assert loaded is None
    assert len(requests) == 1
    assert (
        f"PackValidationError: incompatible {field_name}: "
        f"expected {env_pack_loader.PACK_VERSION_IDENTITIES[field_name]!r}, "
        f"got {bad_value!r}"
    ) in caplog.text


@_run_async
async def test_signed_pack_requires_and_uses_verifier(tmp_path: Path) -> None:
    artifact = _archive_bytes()
    pack_sha256 = hashlib.sha256(artifact).hexdigest()
    metadata = _metadata(pack_sha256, artifact, artifact_signature="base64-signature")

    async with _client(metadata, artifact) as client:
        missing_verifier = await fetch_and_validate_pack(
            pack_sha256,
            "https://backend.test",
            object(),
            scratch_root=tmp_path,
            http_client=client,
        )
    assert missing_verifier is None

    verified: list[tuple[Path, str]] = []

    def verifier(path: Path, signature: str) -> bool:
        verified.append((path, signature))
        return True

    async with _client(metadata, artifact) as client:
        loaded = await fetch_and_validate_pack(
            pack_sha256,
            "https://backend.test",
            object(),
            scratch_root=tmp_path,
            http_client=client,
            artifact_signature_verifier=verifier,
        )

    assert loaded is not None
    assert verified[0][1] == "base64-signature"
    loaded.close()


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
            )

    assert loaded is None
    assert list(tmp_path.iterdir()) == []
    metrics_record = next(
        record for record in caplog.records if "; metrics=" in record.message
    )
    metrics = json.loads(metrics_record.message.partition("; metrics=")[2])
    assert metrics["stage"] == "portable_validation"
