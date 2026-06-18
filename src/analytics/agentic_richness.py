"""Deterministic agentic_richness calculator (ORO-1372).

Walks an eval trajectory bundle (list of step objects as captured by
the sandbox sandbox_output writer) and reports the share of catalogue
tool dispatches whose immediate-preceding LLM inference output
explicitly requested them. See the design spec for the algorithm
contract, anti-gaming rule, and worked examples.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

# Catalogue-tool proxy paths that count as "dispatches" for the metric.
# Everything else (inference, recommend_product local handler, terminate)
# is excluded from both numerator and denominator.
CATALOGUE_PATHS_TO_TOOLS: dict[str, str] = {
    "/search/find_product": "find_product",
    "/search/view_product_information": "view_product_information",
    "/search/check_product_match": "check_product_match",
    "/search/find_products_in_same_shop": "find_products_in_same_shop",
    "/search/calculate_voucher": "calculate_voucher",
}

INFERENCE_PATH = "/inference/chat/completions"

_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(\[.*?\]|\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


@dataclass(frozen=True)
class AgenticRichnessResult:
    """Per-trajectory result of `calc_agentic_richness`."""

    agentic_richness: float
    llm_emitted_count: int
    total_dispatch_count: int
    n_steps: int
    tool_breakdown: dict[str, int] = field(default_factory=dict)


def _normalize_path(raw: str) -> str:
    return urlparse(raw).path.rstrip("/")


def _extract_xml_tool_names(content: str) -> list[str]:
    out: list[str] = []
    if not content:
        return out
    for m in _TOOL_CALL_BLOCK_RE.finditer(content):
        try:
            payload = json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            continue
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("tool") or item.get("function")
            if isinstance(name, str):
                out.append(name)
    return out


def _extract_native_tool_names(message: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for tc in message.get("tool_calls", []) or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function", {}) or {}
        name = fn.get("name")
        if isinstance(name, str):
            out.append(name)
    return out


def _llm_emitted_targets(inference_call: dict[str, Any]) -> list[str]:
    """Return tool names the most-recent LLM response asked for, in order."""
    resp = inference_call.get("response") or {}
    if not isinstance(resp, dict):
        return []
    targets: list[str] = []
    for choice in resp.get("choices") or []:
        msg = choice.get("message") or {}
        targets.extend(_extract_native_tool_names(msg))
        targets.extend(_extract_xml_tool_names(msg.get("content") or ""))
    return targets


def calc_agentic_richness_for_step(step: dict[str, Any]) -> tuple[int, int]:
    """Walk one step's proxy_calls list. Return (llm_emitted_count, total_dispatch_count)."""
    proxy_calls = (step.get("extra_info") or {}).get("proxy_calls") or []
    summaries = [c for c in proxy_calls if c.get("kind") == "summary"]

    llm_emitted = 0
    total_dispatch = 0
    pending_targets: list[str] = []

    for call in summaries:
        path = _normalize_path(call.get("path", ""))
        if path == INFERENCE_PATH:
            pending_targets = _llm_emitted_targets(call)
            continue
        tool = CATALOGUE_PATHS_TO_TOOLS.get(path)
        if tool is None:
            continue  # not a catalogue dispatch — skip
        total_dispatch += 1
        if pending_targets and pending_targets[0] == tool:
            llm_emitted += 1
        if pending_targets:
            pending_targets.pop(0)
    return llm_emitted, total_dispatch


def calc_agentic_richness(bundle: list[dict[str, Any]]) -> AgenticRichnessResult:
    """Run the calculator across every step in a single trajectory bundle."""
    total_llm = 0
    total_disp = 0
    tools: Counter[str] = Counter()

    for step in bundle:
        l, t = calc_agentic_richness_for_step(step)
        total_llm += l
        total_disp += t
        for call in (step.get("extra_info") or {}).get("proxy_calls") or []:
            if call.get("kind") != "summary":
                continue
            tool = CATALOGUE_PATHS_TO_TOOLS.get(_normalize_path(call.get("path", "")))
            if tool is not None:
                tools[tool] += 1

    ratio = (total_llm / total_disp) if total_disp else 0.0
    return AgenticRichnessResult(
        agentic_richness=ratio,
        llm_emitted_count=total_llm,
        total_dispatch_count=total_disp,
        n_steps=len(bundle),
        tool_breakdown=dict(tools),
    )
