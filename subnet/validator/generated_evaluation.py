"""Inputs and score contract for generated-environment evaluations."""

from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any


GENERATED_SCORE_SCHEMA = "oro.generated_run_score.v1"
GENERATED_PROBLEM_SCHEMA = "oro.generated_environment_problem.v1"

# Cheating outcomes always hard-fail the whole pack: we never want to score
# a run where the agent broke the sealed contract.
_CHEATING_OUTCOMES = frozenset({"leakage", "exploit"})

# Harness failures (environment_error / verifier_error) score as zero
# reward alongside agent_error UP TO a fraction of the roster. Beyond the
# threshold the whole run is treated as an infrastructure failure: the
# upstream (simulator, verifier, provider) is degraded enough that the
# episodes we DID get are unrepresentative, so we hard-fail the run and
# let it re-queue when infra recovers. Prior behavior only hard-failed at
# 100% infra which meant a rate-limited upstream day would ship dozens of
# 0.05-score partial completions and count them against miners.
_HARNESS_OUTCOMES = frozenset({"environment_error", "verifier_error"})
_INFRA_FAILURE_THRESHOLD = 0.30


def select_run_task_roster(
    problems: list[dict[str, Any]], sealed_task_families: dict[str, str]
) -> dict[str, str]:
    """Bind the existing Backend run selection to the unchanged sealed pack."""
    if not problems:
        raise ValueError("generated run requires a nonempty authoritative roster")
    selected: dict[str, str] = {}
    for problem in problems:
        task_id = problem.get("task_id")
        family = problem.get("family")
        if not isinstance(task_id, str) or not task_id or task_id in selected:
            raise ValueError("run roster requires unique nonempty task IDs")
        if task_id not in sealed_task_families:
            raise ValueError("run roster task is absent from the sealed pack")
        if family != sealed_task_families[task_id]:
            raise ValueError("run roster family differs from the sealed pack")
        selected[task_id] = family
    return selected


def validate_run_results(
    results: list[dict[str, Any]],
    expected: dict[str, str],
    *,
    evaluation_run_id: str,
    agent_version_id: str,
    pack_sha256: str,
    require_complete: bool = True,
) -> None:
    """Validate a duplicate-free result batch against the selected roster."""

    task_ids = [result.get("task_id") for result in results]
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("generated result batch requires unique task IDs")
    if require_complete and set(task_ids) != set(expected):
        raise ValueError("generated results differ from the authoritative run roster")
    if any(task_id not in expected for task_id in task_ids):
        raise ValueError(
            "generated result batch differs from the authoritative run roster"
        )
    for result in results:
        if result.get("family") != expected[result["task_id"]]:
            raise ValueError("generated result has the wrong task family")
        if any(
            result.get(field) != value
            for field, value in (
                ("evaluation_run_id", evaluation_run_id),
                ("agent_version_id", agent_version_id),
                ("pack_sha256", pack_sha256),
            )
        ):
            raise ValueError("generated result differs from the claimed run binding")


def write_problem_file(path: Path, sessions: list[dict[str, Any]]) -> None:
    """Expose only the public session bootstrap as sandbox work."""

    rows = []
    for session in sessions:
        session_id = str(session["session_id"])
        policy_view = session["policy_view"]
        rows.append(
            {
                "problem_id": session_id,
                "query": policy_view["query"],
                "category": "generated_environment",
                "environment": {
                    "schema_version": GENERATED_PROBLEM_SCHEMA,
                    "binding": {
                        "session_id": session_id,
                        "tool_contract_version": policy_view["tool_contract_version"],
                    },
                    "policy_view": policy_view,
                },
            }
        )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def aggregate_results(results: list[dict[str, Any]]) -> float:
    """Average paid task rewards, counting agent failures as zero."""

    if not results:
        raise ValueError("generated evaluation produced no episode results")

    task_ids = [str(result.get("task_id") or "") for result in results]
    if any(not task_id for task_id in task_ids) or len(set(task_ids)) != len(task_ids):
        raise ValueError("generated evaluation must produce one result per task")

    outcomes = Counter(str(result.get("outcome") or "") for result in results)
    cheating_failures = {
        outcome: count
        for outcome, count in outcomes.items()
        if outcome in _CHEATING_OUTCOMES
    }
    if cheating_failures:
        detail = ", ".join(
            f"{outcome}={count}" for outcome, count in sorted(cheating_failures.items())
        )
        raise ValueError(f"generated evaluation integrity failure: {detail}")

    harness_count = sum(outcomes.get(o, 0) for o in _HARNESS_OUTCOMES)
    # Strict > matches Backend's `_is_generated_infra_failure` classifier
    # (`failed * 10 > total * 3`). A `>=` here would trip at exactly 3/10
    # while the classifier does NOT recognize that as infra, letting the
    # miner's auto-discard counter increment for infra we own.
    if harness_count * 10 > len(results) * 3:
        detail = ", ".join(
            f"{outcome}={count}" for outcome, count in sorted(outcomes.items())
        )
        raise ValueError(
            f"generated evaluation infrastructure failure: {detail}"
        )

    # A subset of harness failures is scored the same as agent failures:
    # each counts as zero reward against the roster denominator. Prior
    # behavior hard-failed the whole run on any environment_error /
    # verifier_error episode, so a random tool-timeout on 1 of 90 tasks
    # threw away the other 89 completed episodes.
    reward_total = Decimal("0")
    for result in results:
        verdict = result.get("verdict") or {}
        if verdict.get("correct"):
            reward_total += Decimal(str(verdict.get("paid_reward") or 0))

    return float(reward_total / Decimal(len(results)))


def _operation_counts(result: dict[str, Any]) -> Counter[str]:
    """Count validator-observed logical operations without copying trace data."""

    counts: Counter[str] = Counter()
    traces = result.get("call_trace")
    if not isinstance(traces, list):
        return counts
    for trace in traces:
        request = trace.get("request") if isinstance(trace, dict) else None
        if not isinstance(request, dict):
            continue
        grouped_calls = request.get("calls")
        calls = grouped_calls if isinstance(grouped_calls, list) else [request]
        for call in calls:
            action = call.get("action") if isinstance(call, dict) else None
            name = action.get("name") if isinstance(action, dict) else None
            if isinstance(name, str) and name:
                counts[f"environment.{name}"] += 1
    return counts


def summarize_episode_resource_usage(
    results: list[dict[str, Any]],
    stats_by_session: dict[str, dict],
    provider: str,
) -> dict[str, dict]:
    """Map private inference and logical-operation counters to task IDs."""

    episodes = {}
    for result in results:
        stats = stats_by_session.get(str(result.get("session_id")))
        usage = {
            "inference_requests": int((stats or {}).get("inference_total", 0)),
            "inference_failed_requests": int((stats or {}).get("inference_failed", 0)),
            "prompt_tokens": int((stats or {}).get("prompt_tokens", 0)),
            "completion_tokens": int((stats or {}).get("completion_tokens", 0)),
            "requested_models": dict((stats or {}).get("requested_models") or {}),
            "served_models": dict((stats or {}).get("served_models") or {}),
        }
        operation_counts = _operation_counts(result)
        usage["operation_requests"] = sum(operation_counts.values())
        usage["operations"] = dict(sorted(operation_counts.items()))
        if provider != "openrouter":
            usage["inference_cost_status"] = "unsupported"
        elif stats is None:
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
    "select_run_task_roster",
    "summarize_agent_inference_usage",
    "summarize_episode_resource_usage",
    "validate_run_results",
    "write_problem_file",
]
