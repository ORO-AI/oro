"""Deterministic agentic_richness calculator (ORO-1372).

Walks an eval trajectory bundle (list of step objects as captured by
the sandbox_output writer) and reports the share of catalogue
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
        if not isinstance(fn, dict):
            continue
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
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message") or {}
        if not isinstance(msg, dict):
            continue
        targets.extend(_extract_native_tool_names(msg))
        content = msg.get("content") or ""
        if not isinstance(content, str):
            continue
        targets.extend(_extract_xml_tool_names(content))
    return targets


def calc_agentic_richness_for_step(
    step: dict[str, Any],
) -> tuple[int, int, Counter[str]]:
    """Walk one step's proxy_calls list.

    Returns ``(llm_emitted_count, total_dispatch_count, tool_breakdown)``
    where ``tool_breakdown`` counts catalogue dispatches by tool name.
    """
    extra_info = step.get("extra_info") or {}
    if not isinstance(extra_info, dict):
        extra_info = {}
    proxy_calls = extra_info.get("proxy_calls") or []
    if not isinstance(proxy_calls, list):
        proxy_calls = []
    summaries = [
        c for c in proxy_calls if isinstance(c, dict) and c.get("kind") == "summary"
    ]

    llm_emitted = 0
    total_dispatch = 0
    tool_breakdown: Counter[str] = Counter()
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
        tool_breakdown[tool] += 1
        if pending_targets and pending_targets[0] == tool:
            llm_emitted += 1
        if pending_targets:
            pending_targets.pop(0)
    return llm_emitted, total_dispatch, tool_breakdown


def calc_agentic_richness(bundle: list[dict[str, Any]]) -> AgenticRichnessResult:
    """Run the calculator across every step in a single trajectory bundle."""
    total_llm = 0
    total_disp = 0
    tools: Counter[str] = Counter()

    for step in bundle:
        step_llm, step_disp, step_tools = calc_agentic_richness_for_step(step)
        total_llm += step_llm
        total_disp += step_disp
        tools.update(step_tools)

    ratio = (total_llm / total_disp) if total_disp else 0.0
    return AgenticRichnessResult(
        agentic_richness=ratio,
        llm_emitted_count=total_llm,
        total_dispatch_count=total_disp,
        n_steps=len(bundle),
        tool_breakdown=dict(tools),
    )
