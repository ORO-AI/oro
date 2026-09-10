from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from validator.generated_evaluation import (
    GENERATED_PROBLEM_SCHEMA,
    aggregate_results,
    select_run_task_roster,
    validate_run_results,
    write_problem_file,
)
from validator.main import Validator
from validator import main as validator_main


def _result(task_id: str, *, correct: bool, reward: float = 0, outcome="completed"):
    return {
        "task_id": task_id,
        "outcome": outcome,
        "verdict": {"correct": correct, "paid_reward": reward},
    }


def test_problem_file_contains_only_public_session_contract(tmp_path) -> None:
    policy_view = {
        "query": "Find a blue mug",
        "max_steps": 5,
        "tool_contract_version": "v1",
        "tools": [],
        "max_calls_per_turn": 16,
    }
    path = tmp_path / "problems.jsonl"

    write_problem_file(path, [{"session_id": "session-1", "policy_view": policy_view}])

    row = json.loads(path.read_text())
    assert row == {
        "problem_id": "session-1",
        "query": "Find a blue mug",
        "category": "generated_environment",
        "environment": {
            "schema_version": GENERATED_PROBLEM_SCHEMA,
            "binding": {
                "session_id": "session-1",
                "tool_contract_version": "v1",
            },
            "policy_view": policy_view,
        },
    }


def test_score_is_mean_reward_with_agent_failures_as_zero() -> None:
    score = aggregate_results(
        [
            _result("one", correct=True, reward=0.75),
            _result("two", correct=False),
            _result("three", correct=False, outcome="agent_error"),
        ]
    )

    assert score == 0.25


@pytest.mark.parametrize(
    "outcome", ["environment_error", "verifier_error", "leakage", "exploit"]
)
def test_infrastructure_and_integrity_outcomes_reject_the_run(outcome) -> None:
    with pytest.raises(ValueError, match=outcome):
        aggregate_results([_result("one", correct=False, outcome=outcome)])


def test_duplicate_task_result_is_rejected() -> None:
    with pytest.raises(ValueError, match="one result per task"):
        aggregate_results(
            [_result("one", correct=True, reward=1), _result("one", correct=False)]
        )


@pytest.mark.parametrize(
    ("pack_sha256", "expected"),
    [(None, "legacy"), ("a" * 64, "generated")],
)
def test_validator_selects_evaluator_from_claim_binding(
    pack_sha256: str | None, expected: str
) -> None:
    validator = Validator.__new__(Validator)
    validator._run_legacy_evaluation = MagicMock(return_value="legacy")
    validator._run_generated_evaluation = MagicMock(return_value="generated")
    work = SimpleNamespace(env_pack_sha256=pack_sha256)

    result = validator._run_claimed_evaluation(
        work,
        MagicMock(),
        inference_access_token="token",
        inference_provider="openrouter",
        inference_base_url="https://example.test/v1",
    )

    assert result == expected
    selected = getattr(validator, f"_run_{expected}_evaluation")
    selected.assert_called_once()


def test_validator_rejects_invalid_claim_binding() -> None:
    validator = Validator.__new__(Validator)

    with pytest.raises(ValueError, match="64 lowercase hex"):
        validator._run_claimed_evaluation(
            SimpleNamespace(env_pack_sha256="invalid"),
            MagicMock(),
            inference_access_token="token",
            inference_provider="openrouter",
            inference_base_url="https://example.test/v1",
        )


@pytest.mark.parametrize(
    "selected_ids",
    [[f"task-{i}" for i in range(35)], ["task-35", "task-48", "task-80"]],
)
def test_generated_sessions_use_exact_authoritative_subset_without_hidden_bank(
    tmp_path, monkeypatch, selected_ids
):
    all_ids = [f"task-{i}" for i in range(105)]
    families = {task_id: f"family-{i % 7}" for i, task_id in enumerate(all_ids)}
    pack = SimpleNamespace(
        task_ids=all_ids,
        task_specs=[SimpleNamespace(family=families[task]) for task in all_ids],
        close=MagicMock(),
    )
    monkeypatch.setattr(
        validator_main, "fetch_and_validate_pack", AsyncMock(return_value=pack)
    )
    registry = MagicMock()
    registry.start.side_effect = lambda **kw: {
        "session_id": kw["task_id"],
        "policy_view": {"query": "public query", "tool_contract_version": "v1"},
    }
    monkeypatch.setattr(
        validator_main, "SessionRegistry", MagicMock(return_value=registry)
    )
    validator = Validator.__new__(Validator)
    validator.config = SimpleNamespace(
        backend_url="unused",
        session_tool_timeout=1,
        session_simulator_timeout=1,
        sandbox_max_workers=1,
    )
    validator.wallet = SimpleNamespace(hotkey="unused")
    validator.backend_client = MagicMock()
    validator.backend_client.get_run_problems.return_value = [
        {"task_id": task, "family": families[task]} for task in selected_ids
    ]
    validator.session_runtime = MagicMock()
    validator._eval_dir = MagicMock(return_value=tmp_path)
    work = SimpleNamespace(
        env_pack_sha256="a" * 64, eval_run_id="run", agent_version_id="agent"
    )
    _, sessions, roster = validator._create_environment_sessions(
        work, inference_access_token="unused"
    )
    validator.backend_client.get_run_problems.assert_called_once_with(work.eval_run_id)
    assert [
        call.kwargs["task_id"] for call in registry.start.call_args_list
    ] == selected_ids
    assert list(roster) == selected_ids
    write_problem_file(tmp_path / "problems.jsonl", sessions)
    emitted = [
        json.loads(line)["problem_id"]
        for line in (tmp_path / "problems.jsonl").read_text().splitlines()
    ]
    assert emitted == selected_ids
    assert set(emitted).isdisjoint(set(all_ids) - set(selected_ids))
    assert len(pack.task_ids) == 105


@pytest.mark.parametrize(
    "problems",
    [
        [],
        [{"task_id": "public", "family": "right"}] * 2,
        [{"task_id": "fabricated", "family": "right"}],
        [{"task_id": "public", "family": "wrong"}],
        [{"task_id": "", "family": "right"}],
        [{"family": "right"}],
    ],
)
def test_invalid_authoritative_roster_fails_closed(problems):
    with pytest.raises(ValueError):
        select_run_task_roster(problems, {"public": "right", "hidden": "right"})


def test_selection_preserves_backend_order_and_allows_selected_race_subset():
    assert list(
        select_run_task_roster(
            [
                {"task_id": "hidden-b", "family": "right"},
                {"task_id": "hidden-a", "family": "right"},
            ],
            {"public": "right", "hidden-a": "right", "hidden-b": "right"},
        )
    ) == ["hidden-b", "hidden-a"]


@pytest.mark.parametrize(
    "change",
    [
        {"task_id": "hidden"},
        {"family": "wrong"},
        {"evaluation_run_id": "wrong"},
        {"agent_version_id": "wrong"},
        {"pack_sha256": "wrong"},
    ],
)
def test_result_identity_cannot_be_replaced_at_matching_cardinality(change):
    result = {
        "task_id": "public",
        "family": "right",
        "evaluation_run_id": "run",
        "agent_version_id": "agent",
        "pack_sha256": "pack",
    }
    kwargs = dict(evaluation_run_id="run", agent_version_id="agent", pack_sha256="pack")
    validate_run_results([result], {"public": "right"}, **kwargs)
    with pytest.raises(ValueError):
        validate_run_results([{**result, **change}], {"public": "right"}, **kwargs)
    with pytest.raises(ValueError):
        validate_run_results([], {"public": "right"}, **kwargs)
    with pytest.raises(ValueError):
        validate_run_results([result, result], {"public": "right"}, **kwargs)


def test_wrong_roster_cannot_be_emitted(monkeypatch):
    emitter = AsyncMock()
    monkeypatch.setattr(validator_main, "emit_finalized_results", emitter)
    validator = Validator.__new__(Validator)
    registry = MagicMock()
    registry.finalized_results.return_value = [{"task_id": "hidden"}]
    work = SimpleNamespace(
        env_pack_sha256="a" * 64, eval_run_id="run", agent_version_id="agent"
    )
    with pytest.raises(ValueError, match="authoritative run roster"):
        validator._emit_environment_results(
            work, registry, expected_task_roster={"public": "right"}
        )
    emitter.assert_not_called()


def test_incremental_result_batch_accepts_only_selected_bound_tasks():
    result = {
        "task_id": "public",
        "family": "right",
        "evaluation_run_id": "run",
        "agent_version_id": "agent",
        "pack_sha256": "pack",
    }
    kwargs = dict(evaluation_run_id="run", agent_version_id="agent", pack_sha256="pack")

    validate_run_results(
        [result],
        {"public": "right", "later": "right"},
        require_complete=False,
        **kwargs,
    )

    with pytest.raises(ValueError, match="authoritative run roster"):
        validate_run_results(
            [{**result, "task_id": "hidden"}],
            {"public": "right"},
            require_complete=False,
            **kwargs,
        )


def test_generated_runner_flush_failure_does_not_change_score(
    tmp_path, monkeypatch
):
    result = {
        "task_id": "public",
        "family": "right",
        "evaluation_run_id": "run",
        "agent_version_id": "agent",
        "pack_sha256": "a" * 64,
        "outcome": "completed",
        "verdict": {"correct": True, "paid_reward": 0.5},
    }
    registry = MagicMock()
    registry.finalized_results.return_value = [result]
    reporter = MagicMock()
    reporter.flush.side_effect = RuntimeError("backend unavailable")
    reporter_type = MagicMock(return_value=reporter)
    monkeypatch.setattr(validator_main, "GeneratedProgressReporter", reporter_type)

    validator = Validator.__new__(Validator)
    validator._create_environment_sessions = MagicMock(
        return_value=(
            registry,
            [
                {
                    "session_id": "session",
                    "policy_view": {
                        "query": "query",
                        "tool_contract_version": "v1",
                    },
                }
            ],
            {"public": "right"},
        )
    )
    validator._eval_dir = MagicMock(return_value=tmp_path)
    validator.run_sandbox = MagicMock(return_value=(tmp_path / "output.jsonl", {}))
    validator.session_runtime = MagicMock()
    validator._emit_environment_result_batch = MagicMock()
    work = SimpleNamespace(
        env_pack_sha256="a" * 64,
        eval_run_id="run",
        agent_version_id="agent",
    )

    completion = validator._run_generated_evaluation(
        work,
        tmp_path / "agent.py",
        inference_access_token="token",
        inference_provider="openrouter",
        inference_base_url="https://example.test/v1",
    )

    reporter.start.assert_called_once_with()
    reporter.stop.assert_called()
    reporter.flush.assert_called_once_with([result])
    validator.session_runtime.clear.assert_called_once_with(registry)
    assert completion is not None
    assert completion.score == 0.5
