#!/usr/bin/env python3
"""
Escalating concurrency benchmark to find Chutes rate limit threshold.

Patches ProxyClient._make_request_with_retries to capture ALL HTTP status codes
including retried 429s. Starts at low concurrency and escalates until rate limiting
is detected.

Usage:
    cd /Users/sethschilbe/oro/ShoppingBench
    PYTHONPATH=. SANDBOX_PROXY_URL=http://localhost:8080 \
    .venv/bin/python scripts/bench_concurrency.py
"""

import json
import logging
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bench")

# ── Thread-safe telemetry ────────────────────────────────────────────────────

_global_lock = threading.Lock()
_global_calls = []  # all HTTP requests, including retries


def record_call(call_type, path, elapsed, status_code, tokens=0, is_retry=False):
    entry = {
        "type": call_type,
        "path": path,
        "elapsed_s": round(elapsed, 2),
        "timestamp": time.time(),
        "status_code": status_code,
        "input_tokens_est": tokens,
        "is_retry": is_retry,
    }
    with _global_lock:
        _global_calls.append(entry)


# ── Patch at the HTTP request level (inside retry loop) ──────────────────────

import src.agent.proxy_client as pc  # noqa: E402
import requests as req_lib  # noqa: E402

_orig_make_request = pc.ProxyClient._make_request_with_retries


def patched_make_request(self, method, url, **kwargs):
    """Intercept every HTTP request including retries to capture status codes."""
    path = url.replace(self.proxy_url.rstrip("/"), "")
    is_inference = "/inference/" in path
    call_type = "inference" if is_inference else "search"

    # Estimate input tokens for inference calls
    tokens = 0
    if is_inference and "json" in kwargs and kwargs["json"]:
        json_data = kwargs["json"]
        if "messages" in json_data:
            for msg in json_data["messages"]:
                content = msg.get("content", "")
                tokens += len(content) // 4

    max_retries = self.max_retries
    retry_delay = self.retry_delay

    for attempt in range(max_retries + 1):
        is_retry = attempt > 0
        start = time.time()
        try:
            response = req_lib.request(
                method, url, timeout=self.timeout, **kwargs
            )
            elapsed = time.time() - start
            record_call(call_type, path, elapsed, response.status_code, tokens, is_retry)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                log.warning(f"  429 rate limited: {path} (attempt {attempt+1}/{max_retries+1})")
                if attempt < max_retries:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    log.error(f"  429 exhausted retries: {path}")
                    return None
            else:
                log.warning(f"  HTTP {response.status_code}: {path} (attempt {attempt+1})")
                if attempt < max_retries:
                    time.sleep(retry_delay)
                    continue
                return None

        except req_lib.exceptions.Timeout:
            elapsed = time.time() - start
            record_call(call_type, path, elapsed, 408, tokens, is_retry)
            log.warning(f"  Timeout: {path} ({elapsed:.1f}s, attempt {attempt+1})")
            if attempt < max_retries:
                time.sleep(retry_delay)
                continue
            return None

        except Exception as e:
            elapsed = time.time() - start
            record_call(call_type, path, elapsed, 0, tokens, is_retry)
            log.warning(f"  Error: {path}: {e} (attempt {attempt+1})")
            if attempt < max_retries:
                time.sleep(retry_delay)
                continue
            return None

    return None


# Replace the retry method entirely
pc.ProxyClient._make_request_with_retries = patched_make_request

# Also patch post/get to go through our patched method
def patched_post(self, path, json_data=None):
    url = self._build_url(path)
    kwargs = {}
    if json_data:
        kwargs["json"] = json_data
    if self.api_key and "/inference/" in path:
        kwargs["headers"] = {"Authorization": f"Bearer {self.api_key}"}
    return self._make_request_with_retries("POST", url, **kwargs)


def patched_get(self, path, params=None):
    url = self._build_url(path, params)
    return self._make_request_with_retries("GET", url)


pc.ProxyClient.post = patched_post
pc.ProxyClient.get = patched_get

# ── Import agent after patching ──────────────────────────────────────────────

agent_path = Path("/Users/sethschilbe/Downloads/quick-agent.py")
if not agent_path.exists():
    log.error(f"Agent not found: {agent_path}")
    sys.exit(1)

import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("quick_agent", agent_path)
quick_agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quick_agent)

# Allow model override via env var
BENCH_MODEL = os.environ.get("BENCH_MODEL")
if BENCH_MODEL:
    _orig_agent_main = quick_agent.agent_main
    def _model_override_agent_main(problem_data):
        problem_data = dict(problem_data)  # don't mutate original
        # The model is hardcoded in agent_main, so we monkey-patch the module-level reference
        return _orig_agent_main(problem_data)
    # Patch at the inference function level instead
    _orig_inference = quick_agent.inference
    def _model_override_inference(model, messages, temperature=0.0):
        return _orig_inference(BENCH_MODEL, messages, temperature)
    quick_agent.inference = _model_override_inference
    log.info(f"Model override: {BENCH_MODEL}")

# ── Load problems ────────────────────────────────────────────────────────────

problem_file = Path("data/full_suite_90.jsonl")
if not problem_file.exists():
    log.error(f"Problem file not found: {problem_file}")
    sys.exit(1)

all_problems = []
with open(problem_file) as f:
    for line in f:
        line = line.strip()
        if line:
            all_problems.append(json.loads(line))

log.info(f"Loaded {len(all_problems)} problems from {problem_file}")


# ── Run a single problem ────────────────────────────────────────────────────

def run_problem(i, problem, total):
    query = problem.get("query", "")[:60]
    cat = problem.get("category", "?")
    pid = problem.get("problem_id", str(i))[:12]

    log.info(f"  [{i+1}/{total}] START [{cat}] {query}...")
    start = time.time()

    try:
        output = quick_agent.agent_main(problem)
        elapsed = time.time() - start
        num_steps = len(output) if output else 0
        log.info(f"  [{i+1}/{total}] DONE  [{cat}] {elapsed:.1f}s | {num_steps} steps")
        return {"problem_id": pid, "category": cat, "time_s": round(elapsed, 2), "steps": num_steps, "ok": True}
    except Exception as e:
        elapsed = time.time() - start
        log.error(f"  [{i+1}/{total}] FAIL  [{cat}] {elapsed:.1f}s: {e}")
        return {"problem_id": pid, "category": cat, "time_s": round(elapsed, 2), "error": str(e), "ok": False}


# ── Compute RPM from timestamped calls ───────────────────────────────────────

def compute_peak_rpm(calls, window_s=60):
    """Compute peak RPM using a sliding window over inference calls."""
    inf_calls = sorted([c for c in calls if c["type"] == "inference"], key=lambda c: c["timestamp"])
    if len(inf_calls) < 2:
        return len(inf_calls)

    max_rpm = 0
    for i, call in enumerate(inf_calls):
        window_end = call["timestamp"] + window_s
        count = sum(1 for c in inf_calls[i:] if c["timestamp"] <= window_end)
        max_rpm = max(max_rpm, count)

    return max_rpm


# ── Run a concurrency level ──────────────────────────────────────────────────

def run_batch(concurrency, problems):
    global _global_calls

    # Reset telemetry
    with _global_lock:
        _global_calls = []

    n = len(problems)
    log.info(f"\n{'='*70}")
    log.info(f"CONCURRENCY: {concurrency} workers, {n} problems")
    log.info(f"{'='*70}")

    batch_start = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(run_problem, i, p, n): i for i, p in enumerate(problems)}
        for future in as_completed(futures):
            results.append(future.result())

    batch_elapsed = time.time() - batch_start

    # Analyze ALL calls including retries
    with _global_lock:
        calls = list(_global_calls)

    inf_calls = [c for c in calls if c["type"] == "inference"]
    search_calls = [c for c in calls if c["type"] == "search"]

    inf_ok = [c for c in inf_calls if c["status_code"] == 200]
    inf_429 = [c for c in inf_calls if c["status_code"] == 429]
    inf_other_err = [c for c in inf_calls if c["status_code"] not in (200, 429)]
    search_ok = [c for c in search_calls if c["status_code"] == 200]
    search_429 = [c for c in search_calls if c["status_code"] == 429]
    search_other_err = [c for c in search_calls if c["status_code"] not in (200, 429)]

    avg_inf_latency = sum(c["elapsed_s"] for c in inf_ok) / len(inf_ok) if inf_ok else 0
    peak_rpm = compute_peak_rpm(calls)
    avg_rpm = len(inf_calls) / (batch_elapsed / 60) if batch_elapsed > 0 else 0

    ok_count = sum(1 for r in results if r.get("ok"))

    summary = {
        "concurrency": concurrency,
        "problems": n,
        "wall_clock_s": round(batch_elapsed, 1),
        "wall_clock_min": round(batch_elapsed / 60, 1),
        "problems_ok": ok_count,
        "problems_failed": n - ok_count,
        "total_http_requests": len(calls),
        "inference_total": len(inf_calls),
        "inference_ok": len(inf_ok),
        "inference_429": len(inf_429),
        "inference_other_err": len(inf_other_err),
        "search_total": len(search_calls),
        "search_ok": len(search_ok),
        "search_429": len(search_429),
        "search_other_err": len(search_other_err),
        "avg_inference_latency_s": round(avg_inf_latency, 1),
        "avg_rpm": round(avg_rpm, 1),
        "peak_rpm_60s": peak_rpm,
    }

    log.info(f"\n--- Results for {concurrency} workers ---")
    log.info(f"  Wall clock:         {summary['wall_clock_min']} min")
    log.info(f"  Problems OK/Fail:   {ok_count}/{n - ok_count}")
    log.info(f"  HTTP requests:      {len(calls)} total")
    log.info(f"  Inference:          {len(inf_ok)} ok / {len(inf_429)} 429 / {len(inf_other_err)} other err")
    log.info(f"  Search:             {len(search_ok)} ok / {len(search_429)} 429 / {len(search_other_err)} other err")
    log.info(f"  Avg inf latency:    {avg_inf_latency:.1f}s (successful only)")
    log.info(f"  Avg RPM (all inf):  {avg_rpm:.1f}")
    log.info(f"  Peak RPM (60s):     {peak_rpm}")
    log.info(f"  429 rate:           {len(inf_429)}/{len(inf_calls)} = {len(inf_429)/max(len(inf_calls),1)*100:.0f}% of inference calls")

    return summary


# ── Main: escalating concurrency ─────────────────────────────────────────────

CONCURRENCY_LEVELS = [3, 6, 9, 12, 18, 30, 45]
all_summaries = []

for level in CONCURRENCY_LEVELS:
    # Use enough problems to saturate the workers (at least 2x concurrency, capped at 90)
    n_problems = min(max(level * 2, 18), len(all_problems))
    problems = all_problems[:n_problems]

    summary = run_batch(level, problems)
    all_summaries.append(summary)

    # Stop if 429 rate exceeds 30% of inference calls
    total_inf = summary["inference_total"]
    rate_429 = summary["inference_429"]
    if total_inf > 0 and (rate_429 / total_inf) > 0.30:
        log.info(f"\n*** STOPPING: 429 rate is {rate_429/total_inf*100:.0f}% at concurrency {level} ***")
        break

    if rate_429 > 0:
        log.info(f"\n  Some 429s detected ({rate_429}), but under threshold. Escalating...\n")
    else:
        log.info(f"\n  No rate limiting at {level} workers. Escalating...\n")

# ── Final report ─────────────────────────────────────────────────────────────

log.info(f"\n{'='*70}")
log.info("FINAL CONCURRENCY REPORT")
log.info(f"{'='*70}")
log.info(f"{'Workers':>8} | {'Probs':>5} | {'Time':>7} | {'Inf OK':>6} | {'Inf 429':>7} | {'429%':>5} | {'Srch OK':>7} | {'Peak RPM':>8} | {'Avg Lat':>7}")
log.info("-" * 95)
for s in all_summaries:
    total_inf = s['inference_total']
    pct_429 = (s['inference_429'] / total_inf * 100) if total_inf > 0 else 0
    log.info(
        f"{s['concurrency']:>8} | {s['problems']:>5} | {s['wall_clock_min']:>5.1f}m | "
        f"{s['inference_ok']:>6} | {s['inference_429']:>7} | {pct_429:>4.0f}% | "
        f"{s['search_ok']:>7} | {s['peak_rpm_60s']:>8} | "
        f"{s['avg_inference_latency_s']:>5.1f}s"
    )

# Recommendation
max_safe = 0
for s in all_summaries:
    if s["inference_429"] == 0:
        max_safe = s["concurrency"]

if max_safe == 0:
    log.info("\nRate limiting detected at ALL concurrency levels.")
    log.info(f"Even {all_summaries[0]['concurrency']} workers triggers 429s.")
else:
    log.info(f"\nMax concurrency without 429s: {max_safe} workers")

# Save results
output_path = Path("scripts/concurrency_results.json")
with open(output_path, "w") as f:
    json.dump(all_summaries, f, indent=2)
log.info(f"Results saved to {output_path}")
