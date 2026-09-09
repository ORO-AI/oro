"""Live Backend + LocalStack coverage for sealed-pack loading."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bittensor_auth import generate_auth_headers
from bittensor_wallet import Keypair

from validator.env_pack_loader import (
    ENV_CONTRACT_VERSION,
    RESULT_SCHEMA_VERSION,
    RUNTIME_VERSION,
    TOOL_CONTRACT_VERSION,
    VERIFIER_VERSION,
    fetch_and_validate_pack,
)

pytest_plugins = ("tests.compat_fixture",)

_PACK_BUCKET = "oro-env-packs"


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


async def _redis_command(redis_url: str, *parts: str) -> bytes:
    """Issue one RESP command without adding Redis to validator dependencies."""

    parsed = urlparse(redis_url)
    reader, writer = await asyncio.open_connection(
        parsed.hostname or "localhost", parsed.port or 6379
    )
    try:
        commands: list[tuple[str, ...]] = []
        if parsed.password:
            commands.append(
                ("AUTH", parsed.username, parsed.password)
                if parsed.username
                else ("AUTH", parsed.password)
            )
        database = (parsed.path or "/0").lstrip("/") or "0"
        if database != "0":
            commands.append(("SELECT", database))
        commands.append(tuple(parts))

        response = b""
        for command in commands:
            encoded = [value.encode() for value in command]
            payload = [f"*{len(encoded)}\r\n".encode()]
            for value in encoded:
                payload.extend((f"${len(value)}\r\n".encode(), value, b"\r\n"))
            writer.writelines(payload)
            await writer.drain()
            response = await reader.readline()
            if response.startswith(b"-"):
                raise RuntimeError(f"Redis command failed: {response.decode().strip()}")
        return response
    finally:
        writer.close()
        await writer.wait_closed()


async def _warm_validator_permit(
    *, redis_url: str, netuid: int, validator_keypair: Keypair
) -> None:
    key = f"validator_permit:{netuid}:{validator_keypair.ss58_address}"
    response = await _redis_command(redis_url, "SET", key, "1", "EX", "3600")
    assert response == b"+OK\r\n", response


async def _provision_pack(
    *,
    archive_path: Path,
    manifest: dict,
    backend_url: str,
    localstack_url: str,
    admin_keypair: Keypair,
) -> str:
    """Upload and idempotently register one deterministic integration pack."""

    artifact = archive_path.read_bytes()
    pack_sha256 = hashlib.sha256(artifact).hexdigest()
    artifact_key = f"{pack_sha256}.tar.gz"
    family_counts = manifest["epoch"]["family_counts"]
    body = {
        "pack_sha256": pack_sha256,
        "contract_version": ENV_CONTRACT_VERSION,
        "catalog_epoch": "env-pack-integration-fixture-v1",
        "catalog_sha256": manifest["catalog_fingerprint"],
        "search_index_epoch": None,
        "search_index_sha256": None,
        "runtime_version": RUNTIME_VERSION,
        "tool_contract_version": TOOL_CONTRACT_VERSION,
        "verifier_version": VERIFIER_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "artifact_uri": f"s3://{_PACK_BUCKET}/{artifact_key}",
        "artifact_size_bytes": len(artifact),
        "artifact_signature": None,
        "family_counts": family_counts,
        "source_breakdown": {"integration_fixture": manifest["epoch"]["tasks"]},
        "task_count": manifest["epoch"]["tasks"],
    }

    async with httpx.AsyncClient(timeout=30) as client:
        # LocalStack's local endpoint accepts path-style S3 requests. Bucket
        # creation and object upload are idempotent for this deterministic key.
        bucket_response = await client.put(f"{localstack_url.rstrip('/')}/{_PACK_BUCKET}")
        assert bucket_response.status_code in (200, 409), bucket_response.text
        upload_response = await client.put(
            f"{localstack_url.rstrip('/')}/{_PACK_BUCKET}/{artifact_key}",
            content=artifact,
        )
        assert upload_response.status_code == 200, upload_response.text

        register_response = await client.post(
            f"{backend_url.rstrip('/')}/v1/admin/packs/register",
            headers=generate_auth_headers(admin_keypair),
            json=body,
        )
        assert register_response.status_code in (200, 201), (
            "pack registration failed; run the Backend env-pack smoke setup once "
            "to provision //AdminLocal and //ValidatorLocal: "
            f"{register_response.status_code} {register_response.text}"
        )
    return pack_sha256


def test_provisions_and_loads_pack_from_local_stack(compiled_epoch: Path) -> None:
    """Compile fixture -> S3 -> Backend registry -> validator loader."""

    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    localstack_url = os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    netuid_value = os.getenv("ORO_SUBNET_NETUID") or os.getenv("SUBNET_NETUID")
    assert netuid_value, "set ORO_SUBNET_NETUID to the local Backend's SUBNET_NETUID"
    netuid = int(netuid_value)
    admin_uri = os.getenv("ORO_ADMIN_KEY_URI", "//AdminLocal")
    validator_uri = os.getenv("ORO_VALIDATOR_KEY_URI", "//ValidatorLocal")
    archive_path = compiled_epoch.parent / f"{compiled_epoch.name}.tar.gz"
    manifest = _json(compiled_epoch / "manifest.json")

    async def run() -> None:
        validator_keypair = Keypair.create_from_uri(validator_uri)
        await _warm_validator_permit(
            redis_url=redis_url,
            netuid=netuid,
            validator_keypair=validator_keypair,
        )
        pack_sha256 = await _provision_pack(
            archive_path=archive_path,
            manifest=manifest,
            backend_url=backend_url,
            localstack_url=localstack_url,
            admin_keypair=Keypair.create_from_uri(admin_uri),
        )
        loaded = await fetch_and_validate_pack(
            pack_sha256,
            backend_url,
            validator_keypair,
        )
        assert loaded is not None
        assert len(loaded.task_specs) == manifest["epoch"]["tasks"]
        assert loaded.manifest == manifest

        session = loaded.open_session(loaded.task_ids[0])
        assert session.policy_view()["task_id"] == loaded.task_ids[0]
        loaded.close()

    asyncio.run(run())
