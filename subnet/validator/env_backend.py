"""SDK transport policy for the four replay-safe environment operations."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, TypeVar, cast

import httpx
from oro_sdk import Client, is_transient_status
from oro_sdk.bittensor_auth import AsyncSigningTransport
from oro_sdk.retry import compute_delay, parse_retry_after
from oro_sdk.types import Response

from .backend_client import BackendError

T = TypeVar("T")


async def _check_status(response: httpx.Response) -> None:
    # Gate HTTP errors before generated JSON parsing: gateways can return HTML
    # or empty 429/5xx bodies. The status, not the error body, decides retries.
    response.raise_for_status()


@asynccontextmanager
async def environment_client(
    backend_url: str,
    keypair: Any,
    *,
    timeout: float = 60.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[Client]:
    """Keep SDK signing confined to Backend; object-store clients stay separate."""
    owned_transport = transport is None
    wire = transport if transport is not None else httpx.AsyncHTTPTransport(retries=0)
    # Explicitly close the wire transport: SDK signing wrappers do not forward
    # aclose. Supplied transports (e.g. test transports) remain caller-owned.
    try:
        async with Client(
            base_url=backend_url.rstrip("/"),
            timeout=httpx.Timeout(timeout),
            headers={"Accept-Encoding": "identity"},
            follow_redirects=False,
            httpx_args={
                "transport": AsyncSigningTransport(keypair, wrapped=wire),
                "event_hooks": {"response": [_check_status]},
            },
        ) as client:
            yield client
    finally:
        if owned_transport:
            await wire.aclose()


async def call_environment_api(
    api_call: Callable[..., Awaitable[Response[Any]]],
    expected: type[T],
    *,
    operation: str,
    **kwargs: Any,
) -> Response[T]:
    """Bound retries to these replay-safe calls, never arbitrary SDK POSTs.

    Fetch/presign operations mint URLs; result submission deduplicates by the
    bound run/pack/task. Retry the identical request, including ambiguous read
    failures, then let the caller inspect every receipt. No nested transport
    retries: the SDK owns signing/serialization and supplies backoff helpers.
    """
    for attempt in range(3):
        retry_after = None
        try:
            response = await api_call(**kwargs)
        except httpx.HTTPError as exc:
            status = (
                exc.response.status_code
                if isinstance(exc, httpx.HTTPStatusError)
                else None
            )
            if status is not None and not is_transient_status(status):
                raise BackendError(
                    f"{operation} rejected status={status}", status_code=status
                ) from None
            if attempt == 2:
                raise BackendError(
                    f"{operation} failed: {type(exc).__name__} status={status}",
                    status_code=status,
                ) from None
            if isinstance(exc, httpx.HTTPStatusError):
                retry_after = parse_retry_after(exc.response)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise BackendError(
                f"{operation} invalid response: {type(exc).__name__}", status_code=200
            ) from None
        else:
            if response.status_code != 200 or not isinstance(response.parsed, expected):
                raise BackendError(
                    f"{operation} invalid response status={response.status_code}",
                    status_code=response.status_code,
                )
            return cast(Response[T], response)
        await asyncio.sleep(compute_delay(attempt, 1.0, 60.0, False, retry_after))
    raise AssertionError("environment retry loop exhausted")
