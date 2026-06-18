"""ORO-1372 reference miner agent (nonce-aware, native tool_calls).

Reads `oro_metadata.tool_nonces` from each inference response and
forwards the matching nonce on every `/search/*` dispatch so the proxy
can verify the call was LLM-emitted. Byte-identical `arguments` strings
are required — do NOT json.loads + re-serialise.
"""
from __future__ import annotations

import json
import os
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://proxy:80")
MODEL = os.getenv("SANDBOX_MODEL", "deepseek-ai/DeepSeek-V3.2-TEE")
MAX_STEPS = 25

TOOL_SCHEMA = [
    {"type": "function", "function": {"name": "find_product", "description": "Search for products (2-4 keywords).", "parameters": {"type": "object", "properties": {"q": {"type": "string"}, "page": {"type": "integer"}, "shop_id": {"type": "string"}, "price": {"type": "string"}, "sort": {"type": "string"}, "service": {"type": "string"}}, "required": ["q"]}}},
    {"type": "function", "function": {"name": "view_product_information", "description": "Get detailed product info for comma-separated product_ids.", "parameters": {"type": "object", "properties": {"product_ids": {"type": "string"}}, "required": ["product_ids"]}}},
    {"type": "function", "function": {"name": "check_product_match", "description": "Verify a product meets requirements JSON before recommending.", "parameters": {"type": "object", "properties": {"product_id": {"type": "string"}, "requirements": {"type": "string"}}, "required": ["product_id", "requirements"]}}},
    {"type": "function", "function": {"name": "find_products_in_same_shop", "description": "Find all listed products from one shop.", "parameters": {"type": "object", "properties": {"product_queries": {"type": "string"}}, "required": ["product_queries"]}}},
    {"type": "function", "function": {"name": "calculate_voucher", "description": "Calculate final price after voucher discount.", "parameters": {"type": "object", "properties": {"product_prices": {"type": "string"}, "voucher_type": {"type": "string"}, "discount_value": {"type": "number"}, "threshold": {"type": "number"}, "budget": {"type": "number"}, "cap": {"type": "number"}}, "required": ["product_prices", "voucher_type", "discount_value", "threshold", "budget"]}}},
    {"type": "function", "function": {"name": "recommend_product", "description": "Recommend comma-separated product_ids to the user. Call before terminate.", "parameters": {"type": "object", "properties": {"product_ids": {"type": "string"}}, "required": ["product_ids"]}}},
]

SYSTEM_PROMPT = """You are a shopping assistant for Lazada (Southeast Asia).

Tools: find_product, view_product_information, check_product_match, find_products_in_same_shop, calculate_voucher, recommend_product.

Rules:
- Use 2-3 keyword search queries — never paste the full user request into `q`.
- Call view_product_information on the top 3-5 candidates before recommending.
- Call check_product_match to verify attributes match exactly.
- You MUST call recommend_product before terminating. If no perfect match, recommend the best available.
- For same-shop or voucher tasks, use find_products_in_same_shop then calculate_voucher.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_nonces(resp: dict[str, Any]) -> dict[str, str]:
    """Extract {call_id: nonce} from oro_metadata.tool_nonces."""
    meta = (resp or {}).get("oro_metadata") or {}
    nonces = meta.get("tool_nonces") or {}
    return {str(k): str(v) for k, v in nonces.items() if isinstance(v, str)}


def dispatch_tool(name: str, arguments_raw: str, nonce: str, call_id: str) -> Any:
    """POST raw arguments bytes to the proxy with the nonce header."""
    url = f"{PROXY_URL.rstrip('/')}/search/{name}"
    headers = {
        "Content-Type": "application/json",
        "X-Tool-Nonce": nonce,
        "X-Tool-Call-Id": call_id,
    }
    try:
        r = requests.post(url, data=arguments_raw.encode(), headers=headers, timeout=120)
        return r.json() if r.status_code == 200 else {"error": r.status_code}
    except Exception as e:
        return {"error": str(e)}


def _infer(messages: list[dict]) -> dict[str, Any]:
    api_key = os.getenv("CHUTES_ACCESS_TOKEN", "")
    r = requests.post(
        f"{PROXY_URL.rstrip('/')}/inference/chat/completions",
        json={"model": MODEL, "messages": messages, "tools": TOOL_SCHEMA, "stream": False},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=120,
    )
    return r.json() if r.status_code == 200 else {}


# ---------------------------------------------------------------------------
# Main loop (≤50 lines)
# ---------------------------------------------------------------------------

def agent_main(problem_data: dict) -> str:
    query = problem_data.get("query", "")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    for _ in range(MAX_STEPS):
        resp = _infer(messages)
        if not resp:
            break
        nonces = parse_nonces(resp)
        choice = ((resp.get("choices") or [{}])[0]).get("message", {})
        tool_calls = choice.get("tool_calls") or []

        if not tool_calls:
            return choice.get("content", "")

        messages.append({"role": "assistant", "content": choice.get("content"), "tool_calls": tool_calls})

        for tc in tool_calls:
            name = tc.get("function", {}).get("name", "")
            args_raw = tc.get("function", {}).get("arguments", "{}")
            call_id = tc.get("id", "")
            nonce = nonces.get(call_id, "")

            if name == "recommend_product":
                result = f"Recommended: {json.loads(args_raw).get('product_ids', '')}"
            elif name == "terminate":
                return messages[-1].get("content") or query
            else:
                result = dispatch_tool(name, args_raw, nonce, call_id)

            messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)})

    return ""
