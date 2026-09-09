"""Inputs and score contract for generated-environment evaluations."""

from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any


GENERATED_SCORE_SCHEMA = "oro.generated_run_score.v1"
GENERATED_PROBLEM_SCHEMA = "oro.generated_environment_problem.v1"
_INFRASTRUCTURE_OUTCOMES = frozenset(
    {"environment_error", "verifier_error", "leakage", "exploit"}
)


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
    infrastructure_failures = {
        outcome: count
        for outcome, count in outcomes.items()
        if outcome in _INFRASTRUCTURE_OUTCOMES
    }
    if infrastructure_failures:
        detail = ", ".join(
            f"{outcome}={count}"
            for outcome, count in sorted(infrastructure_failures.items())
        )
        raise ValueError(f"generated evaluation infrastructure failure: {detail}")

    reward_total = Decimal("0")
    for result in results:
        verdict = result.get("verdict") or {}
        if verdict.get("correct"):
            reward_total += Decimal(str(verdict.get("paid_reward") or 0))

    return float(reward_total / Decimal(len(results)))


__all__ = [
    "GENERATED_PROBLEM_SCHEMA",
    "GENERATED_SCORE_SCHEMA",
    "aggregate_results",
    "select_run_task_roster",
    "validate_run_results",
    "write_problem_file",
]
