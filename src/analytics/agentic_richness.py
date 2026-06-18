"""Public reference implementation of agentic_richness (ORO-1372).

The proxy stamps `nonce_status` on every catalogue dispatch in the
trajectory log. The metric is then a simple ratio:

    agentic_richness = (#dispatches with nonce_status == "valid")
                       / (#total catalogue dispatches in the trajectory)

Pre-enforcement trajectories (bundles missing nonce_status entirely on
every catalogue call) return None so callers can distinguish them from
zero-richness ones.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

CATALOGUE_PATHS_TO_TOOLS: dict[str, str] = {
    "/search/find_product": "find_product",
    "/search/view_product_information": "view_product_information",
    "/search/check_product_match": "check_product_match",
    "/search/find_products_in_same_shop": "find_products_in_same_shop",
    "/search/calculate_voucher": "calculate_voucher",
}


@dataclass(frozen=True)
class AgenticRichnessResult:
    """Per-trajectory result of `calc_agentic_richness`."""

    agentic_richness: float | None
    valid_count: int
    total_dispatch_count: int
    n_steps: int
    tool_breakdown: dict[str, int] = field(default_factory=dict)
    has_nonce_stamps: bool = False


def _normalize_path(raw: str) -> str:
    return urlparse(raw).path.rstrip("/")


def _walk_catalogue_calls(bundle: list[dict[str, Any]]):
    """Yield each catalogue-dispatch summary entry from the bundle."""
    for step in bundle:
        extra = step.get("extra_info") or {}
        if not isinstance(extra, dict):
            continue
        calls = extra.get("proxy_calls") or []
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict) or call.get("kind") != "summary":
                continue
            path = _normalize_path(call.get("path", ""))
            tool = CATALOGUE_PATHS_TO_TOOLS.get(path)
            if tool is None:
                continue
            yield call, tool


def calc_agentic_richness(bundle: list[dict[str, Any]]) -> AgenticRichnessResult:
    """Walk every step's catalogue dispatches, count valid nonces.

    Returns `agentic_richness = None` for pre-enforcement bundles (no
    catalogue dispatch carried a `nonce_status` field). Caller treats
    `None` as "no proof of compliance" rather than zero.
    """
    valid = 0
    total = 0
    has_stamps = False
    tools: Counter[str] = Counter()

    for call, tool in _walk_catalogue_calls(bundle):
        total += 1
        tools[tool] += 1
        status = call.get("nonce_status")
        if status is not None:
            has_stamps = True
        if status == "valid":
            valid += 1

    ratio = (valid / total) if (has_stamps and total) else (0.0 if has_stamps else None)
    return AgenticRichnessResult(
        agentic_richness=ratio,
        valid_count=valid,
        total_dispatch_count=total,
        n_steps=len(bundle),
        tool_breakdown=dict(tools),
        has_nonce_stamps=has_stamps,
    )


if __name__ == "__main__":
    bundle = json.loads(open(sys.argv[1]).read())
    result = calc_agentic_richness(bundle)
    print(json.dumps({
        "agentic_richness": result.agentic_richness,
        "valid_count": result.valid_count,
        "total_dispatch_count": result.total_dispatch_count,
        "n_steps": result.n_steps,
        "tool_breakdown": result.tool_breakdown,
    }, indent=2))
