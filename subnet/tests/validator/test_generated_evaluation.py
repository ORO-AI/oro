from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from validator.generated_evaluation import (
    GENERATED_PROBLEM_SCHEMA,
    aggregate_results,
    select_run_task_roster,
    summarize_agent_inference_usage,
    summarize_episode_resource_usage,
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
    ("provider", "stats", "expected_status"),
    [
        ("openrouter", {"inference_cost_usd": 0.25}, "complete"),
        (
            "openrouter",
            {"inference_cost_usd": 0.25, "inference_cost_missing": 1},
            "partial",
        ),
        ("openrouter", None, "missing"),
        ("chutes", {"inference_total": 1}, "unsupported"),
    ],
)
def test_episode_inference_usage_status(provider, stats, expected_status) -> None:
    by_session = {"session": stats} if stats is not None else {}

    usage = summarize_episode_resource_usage(
        [{"session_id": "session", "task_id": "task"}], by_session, provider
    )["task"]

    assert usage["inference_cost_status"] == expected_status
    assert ("inference_cost_usd" in usage) is (
        stats is not None and provider == "openrouter"
    )


def test_agent_inference_summary_keeps_multi_model_usage_separate() -> None:
    summary = summarize_agent_inference_usage(
        {
            "agent-1": {
                "inference_total": 2,
                "inference_failed": 0,
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "inference_cost_usd": 0.3,
                "requested_models": {"requested/a": {"requests": 2}},
                "served_models": {"served/a": {"requests": 2, "cost_usd": 0.3}},
            },
            "agent-2": {
                "inference_total": 1,
                "inference_failed": 1,
                "requested_models": {
                    "requested/b": {"requests": 1, "failed_requests": 1}
                },
            },
        },
        "openrouter",
    )

    assert summary["inference_requests"] == 3
    assert summary["inference_failed_requests"] == 1
    assert summary["requested_models"] == {
        "requested/a": {"requests": 2},
        "requested/b": {"requests": 1, "failed_requests": 1},
    }
    assert summary["served_models"] == {
        "served/a": {"requests": 2, "cost_usd": 0.3},
    }
    assert summary["inference_cost_status"] == "complete"
    assert summary["inference_cost_usd"] == 0.3


def test_episode_resource_usage_counts_single_and_grouped_operations() -> None:
    result = {
        "session_id": "session",
        "task_id": "task",
        "call_trace": [
            {"request": {"action": {"name": "search", "args": {}}}},
            {
                "request": {
                    "calls": [
                        {"action": {"name": "view", "args": {}}},
                        {"action": {"name": "search", "args": {}}},
                    ]
                }
            },
        ],
    }

    usage = summarize_episode_resource_usage([result], {}, "openrouter")["task"]

    assert usage["operation_requests"] == 3
    assert usage["operations"] == {
        "environment.search": 2,
        "environment.view": 1,
    }


@pytest.mark.parametrize(
    "call_trace",
    [None, {}, [None, {}, {"request": None}, {"request": {"calls": [None]}}]],
)
def test_episode_resource_usage_ignores_malformed_trace_entries(call_trace) -> None:
    usage = summarize_episode_resource_usage(
        [{"session_id": "session", "task_id": "task", "call_trace": call_trace}],
        {},
        "openrouter",
    )["task"]

    assert usage["operation_requests"] == 0
    assert usage["operations"] == {}


@pytest.mark.parametrize("outcome", ["leakage", "exploit"])
def test_integrity_outcomes_reject_the_run(outcome) -> None:
    """Cheating still hard-fails the whole pack, at any count."""
    with pytest.raises(ValueError, match=outcome):
        aggregate_results([_result("one", correct=False, outcome=outcome)])


@pytest.mark.parametrize("outcome", ["environment_error", "verifier_error"])
@pytest.mark.parametrize("count", [10, 30, 90])
def test_partial_harness_failure_scores_completed_episodes(outcome, count) -> None:
    """A subset of environment_error / verifier_error episodes counts as 0
    reward against the roster denominator, same as agent_error. Prior
    behavior hard-failed the whole run on a single infra episode, throwing
    away every completed task."""
    results = [_result(f"task-{i}", correct=True, reward=1.0) for i in range(count - 1)]
    results.append(_result("hiccup", correct=False, outcome=outcome))
    # (count-1) correct at 1.0 each, 1 harness failure at 0.0, over `count` total
    expected = (count - 1) / count
    assert aggregate_results(results) == pytest.approx(expected)


@pytest.mark.parametrize("outcome", ["environment_error", "verifier_error"])
@pytest.mark.parametrize("failed", [2, 3, 8, 9])
def test_at_or_below_threshold_harness_failures_score_normally(outcome, failed) -> None:
    """Harness failures AT or BELOW the 30% infra threshold score at
    zero reward against the full roster. Strict > matches Backend's
    classifier — 9/30 = 30% exactly is NOT infra."""
    total = 30
    completed = total - failed
    results = [_result(f"good-{i}", correct=True, reward=1.0) for i in range(completed)]
    results.extend(
        _result(f"bad-{i}", correct=False, outcome=outcome) for i in range(failed)
    )
    assert aggregate_results(results) == pytest.approx(completed / total)


@pytest.mark.parametrize("outcome", ["environment_error", "verifier_error"])
@pytest.mark.parametrize("failed", [10, 15, 27, 30])
def test_above_threshold_harness_failures_hard_fail(outcome, failed) -> None:
    """When > 30% of episodes are harness failures the completed sample
    isn't representative of miner performance — hard-fail so the run
    re-queues once infra recovers. First fail boundary: 10/30 = 33.3%."""
    total = 30
    completed = total - failed
    results = [_result(f"good-{i}", correct=True, reward=1.0) for i in range(completed)]
    results.extend(
        _result(f"bad-{i}", correct=False, outcome=outcome) for i in range(failed)
    )
    with pytest.raises(ValueError, match="infrastructure failure"):
        aggregate_results(results)


@pytest.mark.parametrize("outcome", ["environment_error", "verifier_error"])
def test_all_harness_failures_still_hard_fail(outcome) -> None:
    """Boundary case: 100% harness → still hard-fail (unchanged)."""
    results = [_result(f"bad-{i}", correct=False, outcome=outcome) for i in range(5)]
    with pytest.raises(ValueError) as caught:
        aggregate_results(results)
    assert str(caught.value) == (
        f"generated evaluation infrastructure failure: {outcome}=5"
    )


@pytest.mark.parametrize("integrity", ["leakage", "exploit"])
@pytest.mark.parametrize("infra", ["environment_error", "verifier_error"])
def test_integrity_takes_precedence_over_infrastructure(integrity, infra):
    with pytest.raises(ValueError) as caught:
        aggregate_results([
            _result("infra", correct=False, outcome=infra),
            _result("cheat", correct=False, outcome=integrity),
        ])
    assert str(caught.value) == f"generated evaluation integrity failure: {integrity}=1"


def test_valid_all_zero_and_agent_error_run_is_successful_zero_score():
    assert aggregate_results([
        _result("wrong", correct=False),
        _result("agent", correct=False, outcome="agent_error"),
    ]) == 0.0


def test_cheating_outcome_rejects_even_at_low_count() -> None:
    """A single cheating outcome hard-fails regardless of pack size."""
    results = [_result(f"good-{i}", correct=True, reward=1.0) for i in range(9)]
    results.append(_result("bad", correct=False, outcome="leakage"))

    with pytest.raises(ValueError, match="integrity failure"):
        aggregate_results(results)


def test_duplicate_task_result_is_rejected() -> None:
    with pytest.raises(ValueError, match="one result per task"):
        aggregate_results(
            [_result("one", correct=True, reward=1), _result("one", correct=False)]
        )


@pytest.mark.parametrize("outcome,expected_reason,flush_failure", [
    ("environment_error", "generated evaluation infrastructure failure: environment_error=1", False),
    ("verifier_error", "generated evaluation infrastructure failure: verifier_error=1", False),
    ("exploit", "generated evaluation integrity failure: exploit=1", False),
    ("completed", "generated evaluation infrastructure failure: environment_error=1", True),
])
def test_generated_runner_delivers_failed_completion(
    tmp_path, monkeypatch, outcome, expected_reason, flush_failure
):
    result = {
        **_result("task", correct=False, outcome=outcome),
        "family": "right", "evaluation_run_id": "run", "agent_version_id": "agent",
        "pack_sha256": "a" * 64,
    }
    registry = MagicMock()
    registry.key_exhausted = threading.Event()
    registry.finalized_results.return_value = [result]
    reporter = MagicMock()
    if flush_failure:
        reporter.flush.side_effect = RuntimeError("artifact upload failed")
    monkeypatch.setattr(validator_main, "GeneratedProgressReporter", MagicMock(return_value=reporter))
    validator = Validator.__new__(Validator)
    validator._create_environment_sessions = MagicMock(return_value=(
        registry,
        [{"session_id": "session", "policy_view": {"query": "query", "tool_contract_version": "v1"}}],
        {"task": "right"},
    ))
    validator._eval_dir = MagicMock(return_value=tmp_path)
    validator.run_sandbox = MagicMock(return_value=(tmp_path / "output.jsonl", {}))
    validator.session_runtime = MagicMock()
    validator.backend_client = MagicMock()
    work = SimpleNamespace(env_pack_sha256="a" * 64, eval_run_id="run", agent_version_id="agent")
    completion = validator._run_generated_evaluation(
        work, tmp_path / "agent.py", inference_access_token="synthetic",
        inference_provider="openrouter", inference_base_url="https://example.test/v1",
    )
    assert completion is None
    validator.backend_client.complete_run.assert_called_once_with(
        eval_run_id="run", status=validator_main.TerminalStatus.FAILED,
        failure_reason=expected_reason, sandbox_metadata={},
    )
    reporter.flush.assert_called_once_with([result])
    validator.session_runtime.clear.assert_called_once_with(registry)


def test_generated_runner_partial_scores_on_miner_key_exhaustion(tmp_path, monkeypatch):
    """When the miner's per-run inference key hits its cap mid-run, the
    completed episodes are scored and the run is marked SUCCESS with a
    partial score. Prior behavior discarded all completed work and marked
    the run FAILED, wasting validator compute and cliff-dropping the miner's
    score to 0."""
    # Fixture: 2 tasks completed correctly (0.5 reward each), 2 tasks
    # quarantined as agent_error when the key was exhausted. Expected score
    # = 1.0 total paid_reward / 4 tasks = 0.25.
    results = [
        {**_result("t1", correct=True, reward=0.5), "family": "right",
         "evaluation_run_id": "run", "agent_version_id": "agent",
         "pack_sha256": "a" * 64},
        {**_result("t2", correct=True, reward=0.5), "family": "right",
         "evaluation_run_id": "run", "agent_version_id": "agent",
         "pack_sha256": "a" * 64},
        {**_result("t3", correct=False, outcome="agent_error"), "family": "right",
         "evaluation_run_id": "run", "agent_version_id": "agent",
         "pack_sha256": "a" * 64},
        {**_result("t4", correct=False, outcome="agent_error"), "family": "right",
         "evaluation_run_id": "run", "agent_version_id": "agent",
         "pack_sha256": "a" * 64},
    ]
    registry = MagicMock()
    registry.key_exhausted = threading.Event()
    registry.key_exhausted.set()
    registry.finalized_results.return_value = results
    reporter = MagicMock()
    monkeypatch.setattr(validator_main, "GeneratedProgressReporter", MagicMock(return_value=reporter))
    validator = Validator.__new__(Validator)
    validator._create_environment_sessions = MagicMock(return_value=(
        registry,
        [{"session_id": "session", "policy_view": {"query": "query", "tool_contract_version": "v1"}}],
        {"t1": "right", "t2": "right", "t3": "right", "t4": "right"},
    ))
    validator._eval_dir = MagicMock(return_value=tmp_path)
    validator._simulator_inference_stats_file = MagicMock(return_value=tmp_path / "sim.jsonl")
    # sandbox stopped early via stop_event → empty output is expected on this path
    validator.run_sandbox = MagicMock(return_value=(None, {}))
    validator.session_runtime = MagicMock()
    validator.backend_client = MagicMock()
    work = SimpleNamespace(env_pack_sha256="a" * 64, eval_run_id="run", agent_version_id="agent")

    completion = validator._run_generated_evaluation(
        work, tmp_path / "agent.py", inference_access_token="synthetic",
        inference_provider="openrouter", inference_base_url="https://example.test/v1",
    )
    assert validator.run_sandbox.call_args.kwargs["stop_event"] is registry.key_exhausted
    # Not a run-level failure — the completion is returned to the caller so
    # the outer scoring path finalizes SUCCESS with the partial score.
    assert completion is not None
    assert completion.score == pytest.approx(0.25)
    assert completion.sandbox_metadata["_miner_inference_key_exhausted"] is True
    validator.backend_client.complete_run.assert_not_called()


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
        "session_id": "session",
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
        "session_id": "session",
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


@pytest.mark.parametrize(
    ("sidecar_stats", "output_stats", "expected_cost"),
    [
        (
            {
                "problem_id": "session",
                "inference_total": 2,
                "inference_failed": 0,
                "inference_cost_usd": 0.125,
                "inference_cost_missing": 0,
                "prompt_tokens": 10,
                "completion_tokens": 4,
            },
            None,
            0.125,
        ),
        ({"problem_id": "session", "inference_total": "invalid"}, None, None),
        (
            None,
            {
                "problem_id": "session",
                "inference_total": 3,
                "inference_failed": 0,
                "inference_cost_usd": 0.25,
                "inference_cost_missing": 0,
                "prompt_tokens": 12,
                "completion_tokens": 5,
            },
            0.25,
        ),
        (
            {
                "problem_id": "session",
                "inference_total": 2,
                "inference_cost_usd": 0.125,
            },
            {
                "problem_id": "session",
                "inference_total": 3,
                "inference_cost_usd": 0.25,
            },
            0.25,
        ),
    ],
)
def test_generated_runner_retains_agent_inference_summary(
    tmp_path, monkeypatch, sidecar_stats, output_stats, expected_cost
):
    result = {
        "task_id": "public",
        "session_id": "session",
        "family": "right",
        "evaluation_run_id": "run",
        "agent_version_id": "agent",
        "pack_sha256": "a" * 64,
        "outcome": "completed",
        "verdict": {"correct": True, "paid_reward": 0.5},
    }
    registry = MagicMock()
    registry.key_exhausted = threading.Event()
    registry.finalized_results.return_value = [result]
    reporter = MagicMock()
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
    if sidecar_stats is not None:
        (tmp_path / "inference_stats.jsonl").write_text(
            json.dumps(sidecar_stats) + "\n"
        )
    if output_stats is not None:
        (tmp_path / "output.jsonl").write_text(
            json.dumps(
                {
                    "problem_id": "session",
                    "_shadow_inference_usage": output_stats,
                    "dialogue": [],
                }
            )
            + "\n"
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
    usage = completion.sandbox_metadata["_shadow_resource_usage"]["by_episode"][
        "public"
    ]
    if expected_cost is None:
        assert usage["inference_cost_status"] == "missing"
        assert "inference_cost_usd" not in usage
    else:
        assert usage["inference_cost_usd"] == expected_cost
