"""Fault injection against the published SDK, without network or inference."""

import asyncio
from unittest.mock import AsyncMock
from uuid import UUID

import httpx
import pytest
from bittensor_wallet import Keypair
from oro_sdk.api.validator import (
    get_pack,
    get_race_pack,
    presign_episode_artifact,
    submit_episode_results,
)
from oro_sdk.models.episode_artifact_presign_request import (
    EpisodeArtifactPresignRequest,
)
from oro_sdk.models.episode_artifact_presign_response import (
    EpisodeArtifactPresignResponse,
)
from oro_sdk.models.pack_fetch_response import PackFetchResponse
from oro_sdk.models.race_pack_fetch_response import RacePackFetchResponse
from oro_sdk.models.submit_episode_results_request import SubmitEpisodeResultsRequest
from oro_sdk.models.submit_episode_results_response import SubmitEpisodeResultsResponse

from validator import env_backend
from validator.env_backend import (
    BackendError,
    call_environment_api,
    environment_client,
)


RUN = UUID(int=1)
SHA = "a" * 64


def _operation(kind):
    pack = {
        "pack_sha256": SHA,
        "download_url": "https://objects.test/archive",
        "download_url_expires_at": "2099-01-01T00:00:00+00:00",
        "download_url_sha256": "b" * 64,
        "download_url_size_bytes": 123,
        "delivery_task_ids": ["task"],
        "task_count": 1,
        "family_counts": {"test": 1},
        "contract_version": "v1",
        "runtime_version": "v1",
        "tool_contract_version": "v1",
        "verifier_version": "v1",
        "result_schema_version": "v1",
        "catalog_epoch": "test",
        "catalog_sha256": SHA,
        "search_index_epoch": None,
        "search_index_sha256": None,
    }
    if kind == "pack":
        return get_pack.asyncio_detailed, PackFetchResponse, {"pack_sha256": SHA}, pack
    if kind == "race":
        return (
            get_race_pack.asyncio_detailed,
            RacePackFetchResponse,
            {"race_id": RUN},
            {
                **pack,
                "race_id": str(RUN),
                "delivery_scope": "race",
            },
        )
    if kind == "presign":
        return (
            presign_episode_artifact.asyncio_detailed,
            EpisodeArtifactPresignResponse,
            {
                "body": EpisodeArtifactPresignRequest(RUN, SHA, SHA, 123),
            },
            {
                "upload_url": "https://objects.test/upload",
                "artifact_uri": "s3://test/key",
                "artifact_sha256": SHA,
            },
        )
    return (
        submit_episode_results.asyncio_detailed,
        SubmitEpisodeResultsResponse,
        {
            "body": SubmitEpisodeResultsRequest.from_dict(
                {
                    "results": [
                        {
                            "eval_run_id": str(RUN),
                            "env_pack_sha256": SHA,
                            "task_id": "task",
                            "family": "test",
                            "outcome": "environment_error",
                        }
                    ]
                }
            ),
        },
        {
            "results": [{"eval_run_id": str(RUN), "task_id": "task", "status": 409}],
            "counts": {"409": 1},
        },
    )


@pytest.mark.parametrize("kind", ["pack", "race", "presign", "results"])
@pytest.mark.parametrize("failure", [429, 503, "connect", "read"])
def test_replay_safe_operations_retry_identically_with_fresh_auth(
    kind, failure, monkeypatch
):
    api, expected, kwargs, payload = _operation(kind)
    requests = []
    sleep = AsyncMock()
    monkeypatch.setattr(env_backend.asyncio, "sleep", sleep)

    def handler(request):
        requests.append(
            (request.method, str(request.url), request.content, dict(request.headers))
        )
        if len(requests) == 1:
            if failure == "connect":
                raise httpx.ConnectError("test", request=request)
            if failure == "read":
                raise httpx.ReadTimeout("test", request=request)
            return httpx.Response(
                failure, text="gateway error", headers={"Retry-After": "20"}
            )
        return httpx.Response(200, json=payload)

    async def run():
        async with environment_client(
            "https://backend.test",
            Keypair.create_from_uri("//TestValidator"),
            transport=httpx.MockTransport(handler),
        ) as client:
            response = await call_environment_api(
                api, expected, operation=kind, client=client, **kwargs
            )
            assert isinstance(response.parsed, expected)

    asyncio.run(run())
    assert len(requests) == 2
    assert requests[0][:3] == requests[1][:3]
    assert requests[0][3]["x-nonce"] != requests[1][3]["x-nonce"]
    assert requests[0][3]["x-signature"] != requests[1][3]["x-signature"]
    sleep.assert_awaited_once_with(20.0 if isinstance(failure, int) else 1.0)


@pytest.mark.parametrize("status", [401, 403, 404, 409, 422, 302, 503])
def test_bounded_retries_and_no_redirect_following(status, monkeypatch):
    calls = []
    sleep = AsyncMock()
    monkeypatch.setattr(env_backend.asyncio, "sleep", sleep)

    def handler(request):
        calls.append(request)
        return httpx.Response(
            status, text="not JSON", headers={"Location": "https://objects.test/stolen"}
        )

    async def run():
        async with environment_client(
            "https://backend.test",
            Keypair.create_from_uri("//TestValidator"),
            transport=httpx.MockTransport(handler),
        ) as client:
            with pytest.raises(BackendError):
                await call_environment_api(
                    get_pack.asyncio_detailed,
                    PackFetchResponse,
                    operation="pack",
                    client=client,
                    pack_sha256=SHA,
                )

    asyncio.run(run())
    assert len(calls) == (3 if status == 503 else 1)
    assert all(request.url.host == "backend.test" for request in calls)
    assert sleep.await_count == (2 if status == 503 else 0)


@pytest.mark.parametrize("cancelled", [False, True])
def test_owned_wire_transport_is_closed_even_on_cancellation(cancelled, monkeypatch):
    class Wire(httpx.MockTransport):
        closed = False

        async def aclose(self):
            self.closed = True

    wire = Wire(lambda request: httpx.Response(200, json={}))
    monkeypatch.setattr(env_backend.httpx, "AsyncHTTPTransport", lambda **kwargs: wire)

    async def run():
        try:
            async with environment_client("https://backend.test", object()):
                if cancelled:
                    raise asyncio.CancelledError()
        except asyncio.CancelledError:
            pass

    asyncio.run(run())
    assert wire.closed


def test_malformed_success_is_not_retried(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(env_backend.asyncio, "sleep", sleep)

    async def run():
        async with environment_client(
            "https://backend.test",
            Keypair.create_from_uri("//TestValidator"),
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
        ) as client:
            with pytest.raises(BackendError, match="invalid response"):
                await call_environment_api(
                    get_pack.asyncio_detailed,
                    PackFetchResponse,
                    operation="pack",
                    client=client,
                    pack_sha256=SHA,
                )

    asyncio.run(run())
    sleep.assert_not_awaited()
