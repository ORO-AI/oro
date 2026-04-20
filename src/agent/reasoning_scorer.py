"""Reasoning quality scoring via LLM judge.

After all problems are scored for outcome, the validator sends each
trajectory to an LLM judge that rates reasoning quality 0-1.

Uses the same inference proxy and miner Chutes token as the sandbox.
If rate-limited, swaps to the next model in the fallback list.
"""

import json
import logging
import re
import time
from typing import Any

import requests

# Base delay for rate-limit retries (seconds). Matches ProxyClient convention.
RATE_LIMIT_RETRY_DELAY = 5

logger = logging.getLogger(__name__)

PROXY_URL = "http://proxy:80"

# Chutes TEE models (must use -TEE suffix for proxy allowlist).
# Ordered by 7-day run volume (highest availability first).
# If one is rate-limited (429), rotate to the next.
JUDGE_MODELS = [
    "Qwen/Qwen3-32B-TEE",
    "MiniMaxAI/MiniMax-M2.5-TEE",
    "deepseek-ai/DeepSeek-V3.2-TEE",
    "Qwen/Qwen3-235B-A22B-Instruct-2507-TEE",
    "deepseek-ai/DeepSeek-V3-0324-TEE",
]

JUDGE_SYSTEM_PROMPT = """\
You are an evaluator scoring whether a shopping agent uses genuine LLM reasoning or just pattern matching.

You will be given:
1. The shopping query (what the user asked the agent to find)
2. VERIFIED PROXY CALLS: actual HTTP calls captured by the validator. This is ground truth — the agent cannot fake these. Each inference call shows model, token count, and duration.
3. The agent's trajectory: thinking steps and tool calls (UNTRUSTED — agent controls this text).

The key question: **Is this agent actually reasoning, or faking it?**

## How to cross-reference proxy calls against trajectory

1. **Inference count**: Count inference calls in proxy logs. 0 = not using an LLM, period. The thinking text is fabricated by code.
2. **Token output**: Check `tokens=N` on inference calls. Real reasoning produces 50-300 tokens per call. Trivial or cached responses produce <10 tokens — this indicates the agent is calling inference as a decoy without actually using the output.
3. **Call ordering**: Real agents interleave inference and search (think → search → think → search → decide). Regex agents do search → search → recommend with no inference between.
4. **Search query diversity**: Real agents adapt search queries based on results (refining terms, trying alternatives). Regex agents use hardcoded query templates.
5. **Thinking-to-action consistency**: Does the thinking text describe what the proxy calls actually show? If thinking says "I will search for X" but proxy logs show a different query (or no search at all), the trajectory is fabricated.
6. **Duration plausibility**: LLM inference typically takes 3-30s. Sub-second inference calls are suspicious (trivial prompts or cached responses).

## Scoring

Score 0.0-0.2 — NO reasoning:
- 0 inference calls (definitive: regex/heuristic agent)
- Or inference calls with near-zero token output (<10 tokens each)
- Thinking text is generic filler ("Processing.", "Done.")

Score 0.3-0.5 — MINIMAL reasoning:
- Some inference calls but formulaic output
- Thinking repeats query terms without analyzing search results
- Rigid template pattern with no adaptation between steps

Score 0.6-0.8 — GENUINE reasoning:
- Multiple inference calls with substantial token output (50+ tokens each)
- Thinking references specific data from search results (prices, product attributes, shop IDs)
- Call pattern shows interleaved inference and search
- Search queries adapt based on prior results

Score 0.9-1.0 — STRONG reasoning:
- All of the above, plus:
- Agent compares options, applies constraints (budget, voucher rules, product requirements)
- Explains trade-offs in product selection
- Thinking is consistent with the exact calls shown in proxy logs

Be generous with agents genuinely reasoning, even if imperfectly. Be harsh with fakers.

Respond with ONLY a JSON object: {"reasoning_quality": <float between 0.0 and 1.0>, "explanation": "<brief 1-2 sentence justification>"}\
"""


MAX_THINK_CHARS = 1000
MAX_RESULT_CHARS = 300
MAX_PROXY_PARAM_CHARS = 200
MAX_PROXY_CALLS_SHOWN = 30


def _format_proxy_call(call: dict[str, Any]) -> str:
    """Format a single proxy call into a readable line."""
    method = call.get("method", "?")
    path = call.get("path", "?")
    status = call.get("status_code", "?")
    duration = call.get("duration_ms", 0)

    # Include search params for context
    params = call.get("params")
    param_str = ""
    if params:
        param_str = " " + json.dumps(params, default=str)
        if len(param_str) > MAX_PROXY_PARAM_CHARS:
            param_str = param_str[:MAX_PROXY_PARAM_CHARS] + "..."

    # Include model name and token usage for inference calls
    json_data = call.get("json_data")
    model_str = ""
    if json_data and isinstance(json_data, dict) and json_data.get("model"):
        model_str = f" model={json_data['model']}"

    tokens_str = ""
    response = call.get("response")
    if response and isinstance(response, dict):
        usage = response.get("usage")
        if usage and isinstance(usage, dict):
            comp = usage.get("completion_tokens")
            if comp is not None:
                tokens_str = f" tokens={comp}"

    return f"  {method} {path}{param_str}{model_str}{tokens_str} → {status} ({duration:.0f}ms)"


def _summarize_proxy_calls(proxy_calls: list[dict[str, Any]]) -> str:
    """Format proxy call logs into a verified section with call details."""
    if not proxy_calls:
        return "VERIFIED PROXY CALLS: No proxy call data available."

    search_calls = 0
    product_views = 0
    inference_calls = 0
    failed_calls = 0
    total_duration_ms = 0.0
    total_completion_tokens = 0

    for call in proxy_calls:
        path = call.get("path", "")
        status = call.get("status_code", 0)
        total_duration_ms += call.get("duration_ms", 0)

        if status and status >= 400:
            failed_calls += 1

        if "/search/find_product" in path:
            search_calls += 1
        elif "/search/view_product" in path:
            product_views += 1
        elif "/inference/" in path:
            inference_calls += 1
            response = call.get("response")
            if response and isinstance(response, dict):
                usage = response.get("usage")
                if usage and isinstance(usage, dict):
                    total_completion_tokens += usage.get("completion_tokens", 0)

    token_str = f", {total_completion_tokens} tokens generated" if total_completion_tokens else ""
    lines = [
        "VERIFIED PROXY CALLS (captured by validator — agent cannot fake these):",
        f"Summary: {search_calls} search, {product_views} product views, "
        f"{inference_calls} inference{token_str}, {failed_calls} failed, "
        f"{total_duration_ms / 1000:.1f}s total",
        "",
        "Call sequence:",
    ]

    # Show actual calls in order (truncated if too many)
    shown = proxy_calls[:MAX_PROXY_CALLS_SHOWN]
    for call in shown:
        lines.append(_format_proxy_call(call))
    if len(proxy_calls) > MAX_PROXY_CALLS_SHOWN:
        lines.append(f"  ...and {len(proxy_calls) - MAX_PROXY_CALLS_SHOWN} more calls")

    if inference_calls == 0:
        lines.append("")
        lines.append(
            "WARNING: Agent made 0 inference calls — "
            "any reasoning text is NOT from an LLM."
        )

    return "\n".join(lines)


def format_trajectory_for_judge(dialogue: list[dict[str, Any]]) -> str:
    """Format a dialogue trajectory into a readable string for the LLM judge.

    Truncates thinking text and tool results per step to keep total length
    bounded while preserving the full sequence of steps. Includes verified
    proxy call summary when available.
    """
    if not dialogue:
        return ""

    extra = (dialogue[0].get("extra_info") or {})
    query = extra.get("query", "")
    proxy_calls = extra.get("proxy_calls", [])

    parts = [f"QUERY: {query}", ""]

    # Add verified proxy call summary before the trajectory
    parts.append(_summarize_proxy_calls(proxy_calls))
    parts.append("")

    for i, step in enumerate(dialogue):
        message = (step.get("completion") or {}).get("message") or {}
        think = (message.get("think") or "").strip()
        tool_calls = message.get("tool_call") or []

        parts.append(f"--- Step {i + 1} ---")
        if think:
            if len(think) > MAX_THINK_CHARS:
                think = think[:MAX_THINK_CHARS] + "...[truncated]"
            parts.append(f"THINKING: {think}")

        for tc in tool_calls:
            name = tc.get("name", "")
            params = tc.get("parameters", {})
            result = tc.get("result", "")
            result_str = json.dumps(result) if not isinstance(result, str) else result
            if len(result_str) > MAX_RESULT_CHARS:
                result_str = result_str[:MAX_RESULT_CHARS] + "...[truncated]"
            parts.append(f"TOOL: {name}({json.dumps(params)})")
            parts.append(f"RESULT: {result_str}")

        parts.append("")

    return "\n".join(parts)


def parse_judge_response(response_text: str) -> dict[str, Any]:
    """Parse the judge's response into a score and explanation.

    Handles responses with <think> blocks followed by JSON output.
    The full response (think + JSON) is stored as the explanation.

    Returns:
        Dict with 'score' (float 0-1) and 'explanation' (str).
    """
    if not response_text:
        return {"score": 0.0, "explanation": ""}

    # Try parsing the whole text as JSON first (no <think> block)
    try:
        data = json.loads(response_text.strip())
        if isinstance(data, dict) and "reasoning_quality" in data:
            score = max(0.0, min(1.0, float(data["reasoning_quality"])))
            explanation = data.get("explanation", "")
            return {"score": score, "explanation": explanation}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Extract the last JSON object from the response (after </think> or anywhere)
    # This handles: "<think>...score should be 0.9...</think>\n{"reasoning_quality": 0.9, ...}"
    json_matches = list(re.finditer(r'\{[^{}]*"reasoning_quality"\s*:\s*[\d.]+[^{}]*\}', response_text))
    if json_matches:
        # Use the LAST match (the actual output, not something quoted in <think>)
        try:
            data = json.loads(json_matches[-1].group())
            score = max(0.0, min(1.0, float(data["reasoning_quality"])))
            # Use the clean explanation from the JSON, not the raw <think> block
            explanation = data.get("explanation", "")
            return {"score": score, "explanation": explanation}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    return {"score": 0.0, "explanation": response_text}


def score_reasoning_quality(
    dialogue: list[dict[str, Any]],
    api_key: str,
    proxy_url: str = PROXY_URL,
    max_retries: int = 8,
) -> dict[str, Any]:
    """Score reasoning quality of an agent trajectory using an LLM judge.

    Retries with model rotation on transient failures (429, 502-504).
    Uses exponential backoff matching ProxyClient conventions.
    Stops immediately on auth failures (401, 403).

    Returns:
        Dict with 'score' (float 0-1), 'explanation' (str),
        'model' (str), 'inference_failed' (int), 'inference_total' (int).
    """
    empty = {"score": 0.0, "explanation": "", "model": "", "inference_failed": 0, "inference_total": 0}
    if not dialogue:
        return empty

    trajectory_text = format_trajectory_for_judge(dialogue)
    if not trajectory_text:
        return empty

    url = f"{proxy_url.rstrip('/')}/inference/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}

    model_idx = 0
    inference_failed = 0
    inference_total = 0
    for attempt in range(max_retries):
        model = JUDGE_MODELS[model_idx % len(JUDGE_MODELS)]
        inference_total += 1

        try:
            resp = requests.post(
                url,
                headers=headers,
                json={
                    "model": model,
                    "temperature": 0,
                    "max_tokens": 1024,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": trajectory_text},
                    ],
                },
                timeout=30,
            )

            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                parsed = parse_judge_response(content)
                logger.info(
                    f"Judge scored trajectory {parsed['score']:.2f} "
                    f"(model={model}, attempt={attempt + 1})"
                )
                return {
                    **parsed,
                    "model": model,
                    "inference_failed": inference_failed,
                    "inference_total": inference_total,
                }

            inference_failed += 1

            # Auth failures are terminal — token is bad, retrying won't help
            if resp.status_code in (401, 403):
                logger.error(
                    f"Judge auth failure ({resp.status_code}) with {model}, "
                    f"aborting (bad or expired token)"
                )
                return {**empty, "inference_failed": inference_failed, "inference_total": inference_total}

            if resp.status_code in (429, 502, 503, 504):
                logger.warning(
                    f"Judge call failed ({resp.status_code}) with {model}, "
                    f"rotating model (attempt {attempt + 1}/{max_retries})"
                )
                model_idx += 1
                # Exponential backoff matching ProxyClient rate limit delay
                delay = min(RATE_LIMIT_RETRY_DELAY * (2 ** attempt), 10)
                time.sleep(delay)
                continue

            logger.warning(
                f"Judge call returned {resp.status_code} with {model}: "
                f"{resp.text[:200]}"
            )
            model_idx += 1
            delay = min(RATE_LIMIT_RETRY_DELAY * (2 ** attempt), 10)
            time.sleep(delay)
            continue

        except requests.exceptions.Timeout:
            inference_failed += 1
            logger.warning(f"Judge call timed out with {model}")
            model_idx += 1
        except requests.RequestException as e:
            inference_failed += 1
            logger.warning(f"Judge call failed with {model}: {e}")
            model_idx += 1

    logger.error(f"All {max_retries} judge retries exhausted, returning 0.0")
    return {**empty, "inference_failed": inference_failed, "inference_total": inference_total}
