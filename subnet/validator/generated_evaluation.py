"""Inputs and score contract for generated-environment evaluations."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Collection
from decimal import Decimal
from pathlib import Path
from typing import Any



GENERATED_SCORE_SCHEMA = "oro.generated_run_score.v1"
GENERATED_PROBLEM_SCHEMA = "oro.generated_environment_problem.v1"

# A validator-side quarantine (no verdict) or a world error is a harness failure,
# not an agent one. Up to this fraction of the roster scores zero like an agent
# failure; beyond it the run is an infrastructure failure and re-queues. A
# deterministic bound trip is the agent's "did not finish": it scores zero and
# keeps its review flag. Any episode left unscored by an environment fault fails
# the run: it can never count as zero.
_INFRA_FAILURE_THRESHOLD = 0.30
_INFRA_FAILURE_PREFIX = "generated evaluation infrastructure failure: "


def claim_roster(pack: Any, delivery_id: str | None, nonces: Any) -> dict[str, str]:
    """Bind a claim to its loaded delivery: the roster and each task's nonce."""

    if delivery_id is None or not isinstance(nonces, dict) or not nonces:
        raise ValueError("claim is not bound to a world delivery (legacy work)")
    if pack.manifest["delivery"]["id"] != delivery_id:
        raise ValueError("loaded delivery differs from the claimed delivery_id")
    if set(nonces) != set(pack.tasks):
        raise ValueError("claim episode_nonces differ from the delivery tasks")
    if not all(isinstance(nonce, str) and nonce for nonce in nonces.values()):
        raise ValueError("claim episode_nonces must be nonempty strings")
    return {task_id: nonces[task_id] for task_id in sorted(pack.tasks)}


def write_problem_file(path: Path, sessions: list[dict[str, Any]]) -> None:
    """Expose only the public session bootstrap as sandbox work."""

    rows = [
        {
            "problem_id": str(session["session_id"]),
            "query": session["policy_view"]["query"],
            "category": "generated_environment",
            "environment": {
                "schema_version": GENERATED_PROBLEM_SCHEMA,
                "binding": session["binding"],
                "policy_view": session["policy_view"],
            },
        }
        for session in sessions
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def aggregate_results(records: list[dict[str, Any]], roster: Collection[str]) -> float:
    """Average verdict scores over the roster; a missing episode scores zero."""

    if not roster:
        raise ValueError("generated evaluation has an empty roster")
    task_ids = [record["task_id"] for record in records]
    if len(set(task_ids)) != len(task_ids) or not set(task_ids) <= set(roster):
        raise ValueError("generated results differ from the delivery roster")
    environment = sum(record["verdict"] is None and record.get("unscored") == "environment" for record in records)
    quarantined = sum(record["verdict"] is None for record in records) - environment
    world = sum(record["verdict"] is not None and "world_error" in record["verdict"]["flags"] for record in records)
    if environment or quarantined + world > len(roster) * _INFRA_FAILURE_THRESHOLD:
        # The Backend's failure classifier reads these outcome counts.
        counts = {"environment_error": environment + quarantined, "verifier_error": world}
        raise ValueError(_INFRA_FAILURE_PREFIX + ", ".join(f"{k}={n}" for k, n in sorted(counts.items()) if n))
    total = sum(
        Decimal(str(record["verdict"]["score"]))
        for record in records
        if record["verdict"] is not None
    )
    return float(total / Decimal(len(roster)))


def _operation_counts(result: dict[str, Any]) -> Counter[str]:
    """Count the validator-observed tool steps without copying trace data."""

    traces = result.get("call_trace")
    return Counter(
        f"environment.{trace['tool']}"
        for trace in (traces if isinstance(traces, list) else [])
        if isinstance(trace, dict) and isinstance(trace.get("tool"), str) and trace["tool"]
    )


def summarize_episode_resource_usage(
    results: list[dict[str, Any]],
    stats_by_session: dict[str, dict],
    provider: str,
) -> dict[str, dict]:
    """Map private inference and logical-operation counters to task IDs."""

    episodes = {}
    for result in results:
        stats = stats_by_session.get(str(result.get("session_id")))
        has_stats = stats is not None
        stats = stats or {}
        usage = {
            "inference_requests": int(stats.get("inference_total", 0)),
            "inference_failed_requests": int(stats.get("inference_failed", 0)),
            "prompt_tokens": int(stats.get("prompt_tokens", 0)),
            "cached_tokens": int(stats.get("cached_tokens", 0)),
            "completion_tokens": int(stats.get("completion_tokens", 0)),
            "requested_models": dict(stats.get("requested_models") or {}),
            "served_models": dict(stats.get("served_models") or {}),
        }
        operation_counts = _operation_counts(result)
        usage["operation_requests"] = sum(operation_counts.values())
        usage["operations"] = dict(sorted(operation_counts.items()))
        if provider != "openrouter":
            usage["inference_cost_status"] = "unsupported"
        elif not has_stats:
            usage["inference_cost_status"] = "missing"
        else:
            missing = int(stats.get("inference_cost_missing", 0))
            usage["inference_cost_status"] = (
                "complete" if missing == 0 else "partial"
            )
            usage["inference_cost_usd"] = float(
                stats.get("inference_cost_usd", 0)
            )
        episodes[str(result["task_id"])] = usage
    return episodes


def summarize_agent_inference_usage(
    stats_by_session: dict[str, dict], provider: str
) -> dict[str, Any]:
    """Aggregate miner-only counters separately from simulator usage."""

    totals: dict[str, Any] = {
        "inference_requests": 0,
        "inference_failed_requests": 0,
        "prompt_tokens": 0,
        "cached_tokens": 0,
        "completion_tokens": 0,
        "requested_models": {},
        "served_models": {},
    }
    cost_usd = 0.0
    cost_missing = 0
    for stats in stats_by_session.values():
        totals["inference_requests"] += int(stats.get("inference_total", 0))
        totals["inference_failed_requests"] += int(stats.get("inference_failed", 0))
        totals["prompt_tokens"] += int(stats.get("prompt_tokens", 0))
        totals["cached_tokens"] += int(stats.get("cached_tokens", 0))
        totals["completion_tokens"] += int(stats.get("completion_tokens", 0))
        cost_usd += float(stats.get("inference_cost_usd", 0))
        cost_missing += int(stats.get("inference_cost_missing", 0))
        for distribution in ("requested_models", "served_models"):
            for model, counters in dict(stats.get(distribution) or {}).items():
                model_total = totals[distribution].setdefault(model, {})
                for key, value in counters.items():
                    model_total[key] = model_total.get(key, 0) + value

    if provider != "openrouter":
        totals["inference_cost_status"] = "unsupported"
    else:
        totals["inference_cost_status"] = "complete" if cost_missing == 0 else "partial"
        totals["inference_cost_usd"] = cost_usd
    return totals


__all__ = [
    "GENERATED_PROBLEM_SCHEMA",
    "GENERATED_SCORE_SCHEMA",
    "aggregate_results",
    "claim_roster",
    "summarize_agent_inference_usage",
    "summarize_episode_resource_usage",
    "write_problem_file",
]
