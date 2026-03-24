#!/usr/bin/env python3
"""
Benchmark agent execution with per-step telemetry.

Runs the staging suite (9 problems) with 6 parallel workers
to match the validator setup. Logs timing for each inference
call, search call, and step.

Usage:
    PYTHONPATH=. SANDBOX_PROXY_URL=http://localhost:8080 \
    CHUTES_ACCESS_TOKEN=<token> \
    .venv/bin/python scripts/bench_agent.py
"""

import json
import logging
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

# Thread-safe per-thread call tracking
_thread_calls = threading.local()
_all_results_lock = threading.Lock()
_all_problem_calls = {}  # problem_index -> list of call entries


def get_thread_calls():
    if not hasattr(_thread_calls, "calls"):
        _thread_calls.calls = []
    return _thread_calls.calls


# Patch ProxyClient BEFORE importing agent
import src.agent.proxy_client as pc  # noqa: E402

_orig_post = pc.ProxyClient.post
_orig_get = pc.ProxyClient.get


def timed_post(self, path, json_data=None):
    start = time.time()
    tokens = 0
    if json_data and "messages" in json_data:
        for msg in json_data["messages"]:
            content = msg.get("content", "")
            tokens += len(content) // 4  # rough: 4 chars per token
    result = _orig_post(self, path, json_data)
    elapsed = time.time() - start
    entry = {
        "type": "inference",
        "path": path,
        "input_tokens_est": tokens,
        "elapsed_s": round(elapsed, 2),
    }
    get_thread_calls().append(entry)
    log.info(f"    INF {elapsed:.2f}s | ~{tokens} input tok")
    return result


def timed_get(self, path, params=None):
    start = time.time()
    result = _orig_get(self, path, params)
    elapsed = time.time() - start
    get_thread_calls().append({
        "type": "search",
        "path": path,
        "elapsed_s": round(elapsed, 2),
    })
    return result


pc.ProxyClient.post = timed_post
pc.ProxyClient.get = timed_get

# NOW import agent (it creates _proxy = ProxyClient() which gets patched methods)
from src.agent.agent import agent_main  # noqa: E402

# Load staging suite problems
problem_file = Path("data/staging_suite_9.jsonl")
if not problem_file.exists():
    log.error(f"Problem file not found: {problem_file}")
    sys.exit(1)

problems = []
with open(problem_file) as f:
    for line in f:
        line = line.strip()
        if line:
            problems.append(json.loads(line))

categories = [p.get("category", "?") for p in problems]
log.info(f"Loaded {len(problems)} problems: {categories}")

MAX_WORKERS = 6


def run_problem(i, problem):
    """Run a single problem and return telemetry."""
    query = problem.get("query", "")[:80]
    cat = problem.get("category", "?")
    pid = problem.get("problem_id", str(i))[:16]

    # Reset thread-local call log
    _thread_calls.calls = []

    log.info(f"[{i+1}/{len(problems)}] START [{cat}]: {query}...")
    start = time.time()

    try:
        output = agent_main(problem)
        elapsed = time.time() - start
        num_steps = len(output) if output else 0

        calls = list(get_thread_calls())

        inference_calls = [c for c in calls if c["type"] == "inference"]
        search_calls = [c for c in calls if c["type"] == "search"]
        total_inference_time = sum(c["elapsed_s"] for c in inference_calls)
        total_search_time = sum(c["elapsed_s"] for c in search_calls)
        total_input_tokens = sum(c.get("input_tokens_est", 0) for c in inference_calls)
        avg_inference = total_inference_time / len(inference_calls) if inference_calls else 0

        # Context growth per inference call
        token_growth = [c.get("input_tokens_est", 0) for c in inference_calls]
        # Latency per inference call
        latency_growth = [c["elapsed_s"] for c in inference_calls]

        result = {
            "problem_id": pid,
            "category": cat,
            "total_time_s": round(elapsed, 2),
            "steps": num_steps,
            "inference_calls": len(inference_calls),
            "search_calls": len(search_calls),
            "total_inference_time_s": round(total_inference_time, 2),
            "total_search_time_s": round(total_search_time, 2),
            "avg_inference_time_s": round(avg_inference, 2),
            "total_input_tokens_est": total_input_tokens,
            "token_growth": token_growth,
            "latency_growth": latency_growth,
            "overhead_s": round(elapsed - total_inference_time - total_search_time, 2),
        }

        log.info(
            f"[{i+1}/{len(problems)}] DONE [{cat}] {elapsed:.1f}s | "
            f"{len(inference_calls)} inf ({total_inference_time:.1f}s, avg {avg_inference:.1f}s) | "
            f"{len(search_calls)} search ({total_search_time:.1f}s) | "
            f"~{total_input_tokens} tok | {num_steps} steps"
        )
        return result

    except Exception as e:
        elapsed = time.time() - start
        log.error(f"[{i+1}/{len(problems)}] FAILED [{cat}] {elapsed:.1f}s: {e}")
        return {"problem_id": pid, "category": cat, "error": str(e), "total_time_s": round(elapsed, 2)}


log.info(f"Running {len(problems)} problems with {MAX_WORKERS} workers (matching validator)")
log.info("=" * 70)

overall_start = time.time()
results = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(run_problem, i, p): i for i, p in enumerate(problems)}
    for future in as_completed(futures):
        results.append(future.result())

overall_elapsed = time.time() - overall_start

log.info(f"\n{'='*70}")
log.info("OVERALL SUMMARY")
log.info(f"{'='*70}")
log.info(f"  Wall-clock time:  {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min)")
log.info(f"  Workers:          {MAX_WORKERS}")
log.info(f"  Problems:         {len(problems)}")
log.info("")

results.sort(key=lambda r: r.get("category", ""))

total_inference_calls = 0
total_search_calls = 0
total_tokens = 0
for r in results:
    if "error" in r:
        log.info(f"  [{r['category']}] FAILED in {r['total_time_s']}s: {r['error']}")
    else:
        total_inference_calls += r["inference_calls"]
        total_search_calls += r["search_calls"]
        total_tokens += r["total_input_tokens_est"]
        growth = r.get("token_growth", [])
        latencies = r.get("latency_growth", [])
        growth_str = " → ".join(str(t) for t in growth[:8])
        if len(growth) > 8:
            growth_str += f" ... ({len(growth)} calls)"
        latency_str = " → ".join(f"{lat:.1f}s" for lat in latencies[:8])
        if len(latencies) > 8:
            latency_str += f" ... ({len(latencies)} calls)"
        log.info(
            f"  [{r['category']:8s}] {r['total_time_s']:6.1f}s | "
            f"{r['inference_calls']:2d} inf (avg {r['avg_inference_time_s']:.1f}s) | "
            f"{r['search_calls']:2d} srch | "
            f"{r['steps']:2d} steps | "
            f"~{r['total_input_tokens_est']:6d} tok"
        )
        log.info(f"            ctx growth: [{growth_str}]")
        log.info(f"            latencies:  [{latency_str}]")

log.info("")
log.info(f"  Total inference calls: {total_inference_calls}")
log.info(f"  Total search calls:    {total_search_calls}")
log.info(f"  Total input tokens:    ~{total_tokens}")
if overall_elapsed > 0:
    log.info(f"  Inference RPM:         ~{total_inference_calls / (overall_elapsed / 60):.0f}")
    log.info(f"  Total RPM (inf+srch):  ~{(total_inference_calls + total_search_calls) / (overall_elapsed / 60):.0f}")

print(json.dumps(results, indent=2))
