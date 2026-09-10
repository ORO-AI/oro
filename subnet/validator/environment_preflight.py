"""Feature-flagged, validator-owned environment startup preflight.

The preflight uses a deterministic local policy and zero model inference. It
proves the real sandbox -> proxy -> validator session path before the validator
begins claiming work, without attaching experimental work to a miner run.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

from oro_env_runtime.schema import CandidateRef

from . import environment_preflight_agent
from .env_pack_loader import PACK_VERSION_IDENTITIES, LoadedPack
from .session_registry import SessionRegistry
from .session_service import SessionRuntime


POSITIVE_POLICY = "oro.reference_policy.v1.positive"
NEGATIVE_POLICY = "oro.reference_policy.v1.negative"
INFERENCE_USD_CAP = "0.00"
_PREFLIGHT_NAMESPACE = UUID("840c8091-3cd2-4dd0-9b7a-c7fd47d51c5c")
_SIMULATOR_REPLY = {
    "content": "Please continue with the in-stock option that meets the request.",
    "action": "continue",
    "reason": "deterministic environment preflight",
}

SandboxRunner = Callable[
    [Path, str, Path],
    tuple[Path | None, dict[str, Any]],
]


class EnvironmentPreflightError(RuntimeError):
    """The feature-flagged environment path failed its startup gate."""


class _DeterministicSimulator:
    async def respond(
        self,
        _transcript: list[dict[str, Any]],
        _signal: dict[str, Any] | None,
    ) -> dict[str, str]:
        return dict(_SIMULATOR_REPLY)


def _candidate_ref(key: str) -> CandidateRef | None:
    product_id, _, sku = key.partition("::")
    if not product_id or not sku:
        return None
    return CandidateRef(product_id=product_id, sku=sku)


def _reference_targets(task: Any, probe: Any) -> tuple[list[CandidateRef], CandidateRef] | None:
    """Build ``(positive_targets, negative)`` from the task's accepted keys and
    the first in-stock, in-budget, non-accepted catalog candidate; ``None`` if
    unusable."""
    if task.acceptance is None or not task.acceptance.acceptable_keys:
        return None
    accepted: list[CandidateRef] = []
    for key in task.acceptance.acceptable_keys:
        ref = _candidate_ref(key)
        if ref is None:
            return None
        accepted.append(ref)
    positive_targets = [accepted[0]]
    second = next(
        (c for c in accepted[1:] if c.product_id != accepted[0].product_id),
        None,
    )
    if second is not None:
        positive_targets.append(second)
    accepted_keys = set(task.acceptance.acceptable_keys)
    negative = next(
        (
            ref
            for ref in probe.catalog.purchasable(
                max_price=task.hard.budget,
                limit=150,
            )
            if ref.key() not in accepted_keys
        ),
        None,
    )
    if negative is None:
        return None
    return positive_targets, negative


def _drive_replay(
    loaded_pack: LoadedPack,
    task_id: str,
    groups: list[list[dict[str, Any]]],
) -> Any:
    """Drive action groups through a fresh replay session, mirroring the
    deterministic simulator fallback so it reproduces the real run's verdict."""
    replay = loaded_pack.open_session(task_id)
    for turn, actions in enumerate(groups, start=1):
        observations = replay.step_parallel(actions)
        if turn == 1 and not any(item.get("done") is True for item in observations):
            replay.env.ledger.append(
                turn=replay.env._turn,
                kind="user_message",
                actor="user_sim",
                payload={"step": turn, **_SIMULATOR_REPLY, "fallback": False},
                state_hash=replay.env.state_hash_now(),
            )
    return replay


def _case_discriminates(loaded_pack: LoadedPack, task_id: str, task: Any, targets: Any) -> bool:
    """True iff the scripted positive verdict is correct and the negative is not,
    proven against the real verifier (skips tasks whose family gate the scripted
    policy can't satisfy, no family-specific knowledge)."""
    groups = _action_groups(task, targets)
    positive = _drive_replay(loaded_pack, task_id, groups[POSITIVE_POLICY]).verdict()
    negative = _drive_replay(loaded_pack, task_id, groups[NEGATIVE_POLICY]).verdict()
    return bool(positive.correct) and not bool(negative.correct)


def _select_reference_case(loaded_pack: LoadedPack) -> tuple[str, Any, Any]:
    """Select any task with a self-verified positive/negative case.

    Task-family-agnostic (the scripted policy still assumes the commerce tool
    contract). Eventless tasks preferred so an event can't invalidate the
    positive order mid-episode; each candidate is validated against the real
    verifier so a case that would false-fail the preflight is never shipped.
    """
    tasks = list(zip(loaded_pack.task_ids, loaded_pack.task_specs, strict=True))
    ordered = [t for t in tasks if t[1].event_rule is None] + [
        t for t in tasks if t[1].event_rule is not None
    ]
    for task_id, task in ordered:
        probe = loaded_pack.open_session(task_id)
        targets = _reference_targets(task, probe)
        if targets is None:
            continue
        if _case_discriminates(loaded_pack, task_id, task, targets):
            return task_id, task, targets
    raise EnvironmentPreflightError(
        "sealed pack has no task with a self-verifiable positive/negative "
        "reference case for the environment preflight"
    )


def _action_groups(task: Any, targets: Any) -> dict[str, list[list[dict[str, Any]]]]:
    positive_targets, negative = targets
    theme = task.family_payload.get("theme") or []
    query = " ".join(str(token) for token in theme) or task.goal_text
    first_turn = [
        {"name": "search", "args": {"query": query, "k": 10, "in_stock": True}},
        {
            "name": "message",
            "args": {"content": "I found options. I will check stock before I order."},
        },
    ]
    return {
        POSITIVE_POLICY: [
            first_turn,
            [
                *[
                    {
                        "name": "add_to_cart",
                        "args": target.model_dump(mode="json"),
                    }
                    for target in positive_targets
                ],
                {
                    "name": "place_test_order",
                    "args": positive_targets[-1].model_dump(mode="json"),
                },
            ],
        ],
        NEGATIVE_POLICY: [
            first_turn,
            [
                {"name": "add_to_cart", "args": negative.model_dump(mode="json")},
                {"name": "place_test_order", "args": negative.model_dump(mode="json")},
            ],
        ],
    }


def _binding(started: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": started["session_id"],
        "tool_contract_version": started["policy_view"]["tool_contract_version"],
    }


def _write_sandbox_inputs(
    run_dir: Path,
    *,
    task: Any,
    starts: dict[str, dict[str, Any]],
    groups: dict[str, list[list[dict[str, Any]]]],
) -> tuple[Path, Path]:
    agent_path = run_dir / "agent.py"
    problem_path = run_dir / "problems.jsonl"
    output_path = run_dir / "output.jsonl"
    output_path.unlink(missing_ok=True)
    agent_path.write_text(
        Path(environment_preflight_agent.__file__).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    rows = [
        {
            "problem_id": policy.rsplit(".", 1)[-1],
            "query": task.goal_text,
            "category": "environment_preflight",
            "environment": {
                "policy_protocol_version": (
                    environment_preflight_agent.POLICY_PROTOCOL_VERSION
                ),
                "reference_policy_version": policy,
                "binding": _binding(starts[policy]),
                "policy_view": starts[policy]["policy_view"],
                # This is validator-owned test input, never miner-supplied work.
                "action_groups": groups[policy],
            },
        }
        for policy in (POSITIVE_POLICY, NEGATIVE_POLICY)
    ]
    problem_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (run_dir / "environment_sessions.json").write_text(
        json.dumps(
            {
                "schema_version": "oro.session_bootstrap.v1",
                "sessions": [starts[POSITIVE_POLICY], starts[NEGATIVE_POLICY]],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return agent_path, problem_path


def _read_sandbox_results(output_path: Path | None) -> list[dict[str, Any]]:
    if output_path is None or not output_path.is_file():
        return []
    return [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _replay_checks(
    loaded_pack: LoadedPack,
    result: dict[str, Any],
    groups: list[list[dict[str, Any]]],
) -> dict[str, bool]:
    replay = _drive_replay(loaded_pack, result["task_id"], groups)
    replayed_verdict = replay.verdict().model_dump(mode="json")
    expected_verdict = result["verdict"]
    return {
        "terminal_state": result["terminal_state_hash"] == replay.env.state_hash_now(),
        "ledger": result["ledger"]
        == [entry.model_dump(mode="json") for entry in replay.env.ledger.entries()],
        "verdict_checks": expected_verdict.get("checks")
        == replayed_verdict.get("checks"),
        "reward_components": expected_verdict.get("reward_record")
        == replayed_verdict.get("reward_record"),
    }


def _sandbox_passed(rows: list[dict[str, Any]]) -> bool:
    return (
        {row.get("problem_id") for row in rows} == {"positive", "negative"}
        and all(row.get("status") == "SUCCESS" for row in rows)
        and all(int(row.get("inference_total") or 0) == 0 for row in rows)
    )


def environment_preflight_run_id(env_pack_sha256: str) -> str:
    """Return the stable evaluation-shaped ID used for logs and sandbox naming.

    Keyed on the content-addressed pack SHA alone: the preflight is
    deterministic per pack, so the sha is a stable, unique run identity.
    """

    return str(uuid5(_PREFLIGHT_NAMESPACE, env_pack_sha256))


def run_environment_preflight(
    *,
    loaded_pack: LoadedPack,
    runtime: SessionRuntime,
    run_dir: Path,
    sandbox_runner: SandboxRunner,
) -> dict[str, Any]:
    """Run the deterministic preflight through the validator's active runtime."""

    preflight_started = time.perf_counter()
    run_id = environment_preflight_run_id(loaded_pack.pack_sha256)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_dir.chmod(0o777)
    registry = SessionRegistry(
        loaded_pack,
        simulator_factory=lambda _session: _DeterministicSimulator(),
    )
    installed = False
    try:
        task_id, task, targets = _select_reference_case(loaded_pack)
        groups = _action_groups(task, targets)
        starts = {
            policy: registry.start(
                evaluation_run_id=run_id,
                agent_version_id=policy,
                task_id=task_id,
                session_id=f"preflight-{policy.rsplit('.', 1)[-1]}",
            )
            for policy in (POSITIVE_POLICY, NEGATIVE_POLICY)
        }
        agent_path, problem_path = _write_sandbox_inputs(
            run_dir,
            task=task,
            starts=starts,
            groups=groups,
        )
        runtime.install(registry)
        installed = True
        sandbox_started = time.perf_counter()
        output_path, sandbox_metadata = sandbox_runner(
            agent_path,
            run_id,
            problem_path,
        )
        sandbox_duration_ms = round(
            (time.perf_counter() - sandbox_started) * 1000.0,
            3,
        )
        sandbox_rows = _read_sandbox_results(output_path)

        policies = []
        for policy in (POSITIVE_POLICY, NEGATIVE_POLICY):
            result = registry.verdict(_binding(starts[policy]))
            policies.append(
                {
                    "reference_policy_version": policy,
                    "task_id": task_id,
                    "terminal_state_hash": result["terminal_state_hash"],
                    "terminal_reason": result["terminal_reason"],
                    "verdict_correct": bool(result["verdict"].get("correct")),
                    "solver_turn_count": result["solver_turn_count"],
                    "action_count": result["action_count"],
                    "replay_checks": _replay_checks(
                        loaded_pack,
                        result,
                        groups[policy],
                    ),
                }
            )

        positive, negative = policies
        expected_action_counts = {
            policy: sum(len(turn) for turn in groups[policy])
            for policy in (POSITIVE_POLICY, NEGATIVE_POLICY)
        }
        passed = all(
            (
                sandbox_metadata.get("exit_code") == 0,
                _sandbox_passed(sandbox_rows),
                positive["verdict_correct"],
                not negative["verdict_correct"],
                positive["solver_turn_count"] == 2,
                positive["action_count"] == expected_action_counts[POSITIVE_POLICY],
                negative["solver_turn_count"] == 2,
                negative["action_count"] == expected_action_counts[NEGATIVE_POLICY],
                all(
                    all(policy["replay_checks"].values()) for policy in policies
                ),
            )
        )
        summary = {
            "schema_version": "oro.environment_preflight.v1",
            "status": "pass" if passed else "fail",
            "run_id": run_id,
            "env_pack_sha256": loaded_pack.pack_sha256,
            "versions": dict(PACK_VERSION_IDENTITIES),
            "inference_usd_cap": INFERENCE_USD_CAP,
            "timing_ms": {
                "sandbox": sandbox_duration_ms,
                "total": round(
                    (time.perf_counter() - preflight_started) * 1000.0,
                    3,
                ),
            },
            "sandbox": {
                "exit_code": sandbox_metadata.get("exit_code"),
                "problem_count": len(sandbox_rows),
                "inference_call_count": sum(
                    int(row.get("inference_total") or 0) for row in sandbox_rows
                ),
            },
            "policies": policies,
        }
        (run_dir / "environment_preflight_result.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not passed:
            raise EnvironmentPreflightError(json.dumps(summary, sort_keys=True))
        return summary
    finally:
        if installed:
            runtime.clear(registry)
        else:
            registry.close()


__all__ = [
    "EnvironmentPreflightError",
    "NEGATIVE_POLICY",
    "POSITIVE_POLICY",
    "environment_preflight_run_id",
    "run_environment_preflight",
]
