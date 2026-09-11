from __future__ import annotations

import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from subnet import local_generated_validator
from subnet.local_generated_validator import (
    GENERATED_FAMILIES,
    LocalGeneratedConfig,
    LocalGeneratedValidatorError,
    run_local_generated_validator,
    validate_local_pack,
)
from subnet.validator.env_pack_loader import PackValidationError
from subnet.validator.session_service import SessionRuntime

pytest_plugins = ("tests.compat_fixture",)


def _pack(families: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        task_specs=[SimpleNamespace(family=family) for family in families],
        task_ids=[f"task-{index}" for index in range(len(families))],
        pack_sha256="a" * 64,
        manifest={"models": {"user_simulator": "vendor/sim"}},
        metadata={"search_index_sha256": "b" * 64},
        close=MagicMock(),
    )


def _episode(
    index: int,
    family: str,
    *,
    outcome: str = "completed",
    correct: bool = True,
    reward: float = 1.0,
) -> dict:
    return {
        "evaluation_run_id": "local-test-run",
        "agent_version_id": "test-agent",
        "task_id": f"task-{index}",
        "session_id": f"session-{index}",
        "pack_sha256": "a" * 64,
        "family": family,
        "outcome": outcome,
        "terminal_reason": "environment_done" if outcome == "completed" else None,
        "error_detail": None if outcome == "completed" else f"{outcome} detail",
        "verdict": (
            {"correct": correct, "paid_reward": reward}
            if outcome == "completed"
            else None
        ),
        "terminal_state_hash": "c" * 64 if outcome == "completed" else None,
        "step_count": 1,
        "solver_turn_count": 1,
        "action_count": 1,
        "render_budget": None,
        "bootstrap": {"session_id": f"session-{index}", "policy_view": {}},
        "call_trace": [],
        "ledger": [],
        "provenance": {"pack_sha256": "a" * 64},
        "environment_error": outcome == "environment_error",
        "verifier_error": outcome == "verifier_error",
    }


def _config(tmp_path: Path, *, timeout: float = 120.0) -> LocalGeneratedConfig:
    agent_path = tmp_path / "agent.py"
    pack_path = tmp_path / "pack.tar.gz"
    agent_path.write_text("def agent_main(problem):\n    return []\n", encoding="utf-8")
    pack_path.write_bytes(b"pack")
    return LocalGeneratedConfig(
        agent_path=agent_path,
        pack_path=pack_path,
        output_root=tmp_path / "runs",
        inference_access_token="test-token",
        inference_provider="openrouter",
        inference_base_url="https://openrouter.test/api/v1",
        model="vendor/test-model",
        pack_sha256="a" * 64,
        problem_count=7,
        max_workers=7,
        timeout=timeout,
    )


def _install_runtime_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    results: list[dict],
) -> tuple[SimpleNamespace, MagicMock, list[list[str]], SessionRuntime, MagicMock]:
    pack = _pack(sorted(GENERATED_FAMILIES))
    registry = MagicMock()
    registry.start.side_effect = [
        {
            "session_id": f"session-{index}",
            "policy_view": {
                "query": family,
                "tool_contract_version": "v1",
                "tools": [],
                "max_steps": 4,
                "max_calls_per_turn": 16,
            },
        }
        for index, family in enumerate(sorted(GENERATED_FAMILIES))
    ]
    registry.finalized_results.return_value = results
    registry.close.side_effect = pack.close
    runtime = SessionRuntime()
    server = MagicMock()
    commands: list[list[str]] = []

    monkeypatch.setattr(
        local_generated_validator, "load_local_pack", lambda *_a, **_k: pack
    )
    monkeypatch.setattr(local_generated_validator, "SessionRuntime", lambda: runtime)
    monkeypatch.setattr(
        local_generated_validator, "SessionServer", lambda *_a, **_k: server
    )
    monkeypatch.setattr(
        local_generated_validator, "SessionRegistry", lambda *_a, **_k: registry
    )

    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        commands.append(command)
        run_dirs = list((tmp_path / "runs").glob("local-*"))
        assert len(run_dirs) == 1
        rows = [
            {
                "problem_id": f"session-{index}",
                "status": "SUCCESS",
                "execution_time": index + 0.5,
                "inference_failure_count": 0,
            }
            for index in range(7)
        ]
        (run_dirs[0] / "sandbox" / "sandbox_output.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(local_generated_validator.subprocess, "run", run)
    return pack, registry, commands, runtime, server


def _failure_summary(error: LocalGeneratedValidatorError) -> dict:
    assert error.summary_path.is_file()
    summary = json.loads(error.summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["error"] == {
        "classification": error.classification,
        "message": str(error),
    }
    return summary


def test_local_pack_selects_first_five_tasks_from_each_generated_family() -> None:
    families = sorted(GENERATED_FAMILIES)
    roster = [family for family in families for _ in range(6)]
    pack = _pack(roster)

    assert validate_local_pack(pack) == [
        f"task-{family_index * 6 + task_index}"
        for family_index in range(7)
        for task_index in range(5)
    ]

    missing = sorted(GENERATED_FAMILIES)[:-1]
    with pytest.raises(ValueError, match="missing="):
        validate_local_pack(_pack(missing))

    unexpected = sorted(GENERATED_FAMILIES) + ["unsupported"]
    with pytest.raises(ValueError, match="unexpected="):
        validate_local_pack(_pack(unexpected))

    with pytest.raises(ValueError, match="insufficient="):
        validate_local_pack(_pack(families * 4))


def test_problem_count_samples_across_every_family_it_can_reach() -> None:
    families = sorted(GENERATED_FAMILIES)
    pack = _pack([family for family in families for _ in range(5)])
    family_of = {
        f"task-{index}": family
        for index, family in enumerate(family for family in families for _ in range(5))
    }

    seven = validate_local_pack(pack, problem_count=7, seed=1)
    assert len(seven) == 7
    assert sorted(family_of[task] for task in seven) == families

    three = validate_local_pack(pack, problem_count=3, seed=1)
    assert len(three) == 3
    assert len({family_of[task] for task in three}) == 3

    ten = validate_local_pack(pack, problem_count=10, seed=1)
    assert len(ten) == 10
    assert sorted(Counter(family_of[task] for task in ten).values()) == [
        1,
        1,
        1,
        1,
        2,
        2,
        2,
    ]

    assert validate_local_pack(pack, problem_count=35, seed=1) == pack.task_ids
    # Archive order, so problems.jsonl and the roster stay readable.
    assert seven == sorted(seven, key=lambda task: int(task.removeprefix("task-")))


def test_problem_selection_repeats_only_for_a_repeated_seed() -> None:
    pack = _pack([family for family in sorted(GENERATED_FAMILIES) for _ in range(5)])

    assert validate_local_pack(pack, problem_count=7, seed=7) == validate_local_pack(
        pack, problem_count=7, seed=7
    )
    assert validate_local_pack(pack, problem_count=7, seed=7) != validate_local_pack(
        pack, problem_count=7, seed=8
    )


def test_problem_count_outside_the_pack_names_the_bounds() -> None:
    pack = _pack([family for family in sorted(GENERATED_FAMILIES) for _ in range(5)])

    for count in (0, 36):
        with pytest.raises(ValueError, match="--problems must be between 1 and 35"):
            validate_local_pack(pack, problem_count=count)


def test_a_short_run_does_not_require_a_full_qualifying_family(tmp_path: Path) -> None:
    # The per-family minimum is a qualifying rule; a sampled run should not inherit it.
    pack = _pack(sorted(GENERATED_FAMILIES))

    assert len(validate_local_pack(pack, problem_count=3, seed=1)) == 3
    with pytest.raises(ValueError, match="insufficient="):
        validate_local_pack(pack)


def test_failed_run_still_writes_the_trajectory_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A failed run is exactly when a miner needs the trajectories.
    results = [
        _episode(index, family)
        for index, family in enumerate(sorted(GENERATED_FAMILIES))
    ]
    _install_runtime_fakes(monkeypatch, tmp_path, results=results)
    monkeypatch.setattr(
        local_generated_validator,
        "aggregate_results",
        lambda _results: (_ for _ in ()).throw(ValueError("boom")),
    )

    with pytest.raises(
        local_generated_validator.LocalGeneratedValidatorError
    ) as excinfo:
        run_local_generated_validator(_config(tmp_path))

    report = excinfo.value.report_path
    assert report is not None and report.is_file()
    assert "TF" in report.read_text() or "episode" in report.read_text()
    summary = json.loads(excinfo.value.summary_path.read_text())
    assert summary["status"] == "failed"


def test_summary_averages_rewards_within_each_family(tmp_path: Path) -> None:
    family = min(GENERATED_FAMILIES)
    path = tmp_path / "summary.json"
    results = [
        _episode(0, family, reward=1.0),
        _episode(1, family, correct=False),
    ]

    local_generated_validator._write_summary(
        path,
        run_id="local-test-run",
        pack=None,
        configured_pack_sha256="a" * 64,
        task_ids=["task-0", "task-1"],
        results=results,
        aggregate_score=0.5,
        status="completed",
    )

    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["family_rewards"] == {family: 0.5}


@pytest.mark.parametrize("termination", ["nonzero", "timeout", "missing_output"])
def test_sandbox_interruption_preserves_finalized_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    termination: str,
) -> None:
    families = sorted(GENERATED_FAMILIES)
    results = [
        _episode(0, families[0]),
        *[
            _episode(index, family, outcome="agent_error")
            for index, family in enumerate(families[1:], 1)
        ],
    ]
    _pack_value, _registry, _commands, _runtime, _server = _install_runtime_fakes(
        monkeypatch, tmp_path, results=results
    )
    original_run = local_generated_validator.subprocess.run
    sandbox_stopped = False

    def interrupt(command: list[str], **kwargs: object) -> SimpleNamespace:
        nonlocal sandbox_stopped
        if command[:2] == ["docker", "rm"]:
            sandbox_stopped = True
            return SimpleNamespace(returncode=0)
        if termination == "missing_output":
            return SimpleNamespace(returncode=17)
        original_run(command, **kwargs)
        if termination == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        sandbox_stopped = True
        return SimpleNamespace(returncode=17)

    def finalize() -> list[dict]:
        if termination == "timeout":
            assert sandbox_stopped, "stop the sandbox before reading final state"
        return results

    _registry.finalized_results.side_effect = finalize
    monkeypatch.setattr(local_generated_validator.subprocess, "run", interrupt)

    with pytest.raises(LocalGeneratedValidatorError) as caught:
        run_local_generated_validator(_config(tmp_path))
    summary = _failure_summary(caught.value)
    artifact_dir = caught.value.artifact_dir
    assert caught.value.classification == "infrastructure"
    if termination == "nonzero":
        assert "sandbox exited with code 17" in str(caught.value)
    elif termination == "timeout":
        assert "timed out after 120 seconds" in str(caught.value)
    else:
        assert "sandbox exited with code 17" in str(caught.value)

    assert summary["task_count"] == 7
    assert summary["aggregate_score"] == pytest.approx(1 / 7)
    assert summary["tasks"][0]["reward"] == 1.0
    assert [task["reward"] for task in summary["tasks"][1:]] == [0.0] * 6
    receipts = [
        json.loads(line)
        for line in (artifact_dir / "episode_results.jsonl").read_text().splitlines()
    ]
    assert len(receipts) == 7
    assert receipts[0]["episode"]["terminal_state_hash"] == "c" * 64


def test_timeout_retries_container_removal_after_failed_forced_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loaded_pack,
) -> None:
    from oro_env_runtime.runtime import TOOL_CONTRACT_VERSION

    from subnet.validator.session_registry import InvalidSessionError, SessionRegistry

    results = [
        _episode(index, family)
        for index, family in enumerate(sorted(GENERATED_FAMILIES))
    ]
    _, _, _, runtime, _ = _install_runtime_fakes(monkeypatch, tmp_path, results=results)
    # This test fixes the roster to one task per family; the fixture has two.
    selected = {
        task.family: (task_id, task)
        for task_id, task in zip(
            loaded_pack.task_ids, loaded_pack.task_specs, strict=True
        )
    }
    loaded_pack.task_ids = [task_id for task_id, _ in selected.values()]
    loaded_pack.task_specs = [task for _, task in selected.values()]
    registry = SessionRegistry(loaded_pack)
    monkeypatch.setattr(
        local_generated_validator, "load_local_pack", lambda *_: loaded_pack
    )
    monkeypatch.setattr(
        local_generated_validator, "SessionRegistry", lambda *_a, **_k: registry
    )
    write_results = local_generated_validator._write_episode_results

    def write_sealed_results(path: Path, rows: list[dict]) -> None:
        # Docker removal has failed; the live HTTP bridge must already be sealed.
        assert removal_attempts == 1
        assert len(rows) == len(loaded_pack.task_ids)
        with pytest.raises(InvalidSessionError, match="finalized"):
            runtime.call(
                {
                    "session_id": rows[0]["session_id"],
                    "tool_contract_version": TOOL_CONTRACT_VERSION,
                    "call_id": "late",
                    "idempotency_key": "late",
                    "turn": 1,
                    "action": {"name": "inspect_cart", "args": {}},
                }
            )
        write_results(path, rows)

    monkeypatch.setattr(
        local_generated_validator, "_write_episode_results", write_sealed_results
    )
    removal_attempts = 0

    def time_out(command: list[str], **kwargs: object) -> SimpleNamespace:
        nonlocal removal_attempts
        if command[:2] == ["docker", "rm"]:
            removal_attempts += 1
            return SimpleNamespace(returncode=1 if removal_attempts == 1 else 0)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(local_generated_validator.subprocess, "run", time_out)

    with pytest.raises(LocalGeneratedValidatorError) as caught:
        run_local_generated_validator(_config(tmp_path))

    assert removal_attempts == 2, str(caught.value)


def test_sandbox_failure_takes_precedence_when_receipts_cannot_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    families = sorted(GENERATED_FAMILIES)
    # Enough verifier_errors to exceed the aggregate's harness tolerance so
    # receipts truly cannot aggregate; the sandbox exit code must still take
    # precedence over the aggregate failure.
    results = [
        _episode(0, families[0]),
        *[
            _episode(index, family, outcome="verifier_error")
            for index, family in enumerate(families[1:4], 1)
        ],
        *[
            _episode(index, family, outcome="agent_error")
            for index, family in enumerate(families[4:], 4)
        ],
    ]
    _pack_value, _registry, _commands, _runtime, _server = _install_runtime_fakes(
        monkeypatch, tmp_path, results=results
    )
    original_run = local_generated_validator.subprocess.run

    def fail_sandbox(command: list[str], **kwargs: object) -> SimpleNamespace:
        original_run(command, **kwargs)
        return SimpleNamespace(returncode=17)

    monkeypatch.setattr(local_generated_validator.subprocess, "run", fail_sandbox)

    with pytest.raises(LocalGeneratedValidatorError) as caught:
        run_local_generated_validator(_config(tmp_path))

    assert caught.value.classification == "infrastructure"
    assert "sandbox exited with code 17" in str(caught.value)
    summary = _failure_summary(caught.value)
    assert summary["aggregate_score"] is None


def test_sandbox_failure_takes_precedence_when_receipt_roster_is_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    families = sorted(GENERATED_FAMILIES)
    results = [_episode(index, family) for index, family in enumerate(families[:-1])]
    _pack_value, _registry, _commands, _runtime, _server = _install_runtime_fakes(
        monkeypatch, tmp_path, results=results
    )

    monkeypatch.setattr(
        local_generated_validator.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(returncode=17),
    )

    with pytest.raises(LocalGeneratedValidatorError) as caught:
        run_local_generated_validator(_config(tmp_path))

    assert caught.value.classification == "infrastructure"
    assert "sandbox exited with code 17" in str(caught.value)
    summary = _failure_summary(caught.value)
    assert summary["aggregate_score"] is None


def test_sandbox_cannot_mount_evaluator_artifacts_writable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = [
        _episode(index, family)
        for index, family in enumerate(sorted(GENERATED_FAMILIES))
    ]
    _pack_value, _registry, commands, _runtime, _server = _install_runtime_fakes(
        monkeypatch, tmp_path, results=results
    )
    completed = run_local_generated_validator(_config(tmp_path))
    mounts = [
        commands[0][index + 1]
        for index, argument in enumerate(commands[0])
        if argument == "-v"
    ]
    assert f"{completed.artifact_dir}:/app/logs:ro" in mounts
    writable = [mount for mount in mounts if not mount.endswith(":ro")]
    assert writable == [f"{completed.artifact_dir}/sandbox:/app/output"]
    assert not completed.artifact_dir.stat().st_mode & 0o022


def test_sandbox_report_cannot_change_evaluator_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = [
        _episode(index, family, outcome="agent_error")
        for index, family in enumerate(sorted(GENERATED_FAMILIES))
    ]
    _install_runtime_fakes(monkeypatch, tmp_path, results=results)
    original_run = local_generated_validator.subprocess.run

    def forge(command: list[str], **kwargs: object) -> object:
        completed = original_run(command, **kwargs)
        run_dir = next((tmp_path / "runs").glob("local-*"))
        (run_dir / "sandbox" / "sandbox_output.jsonl").write_text(
            json.dumps(
                {
                    "problem_id": "session-0",
                    "execution_time": float("nan"),
                    "inference_failure_count": "forged",
                    "error": "forged environment failure",
                }
            )
            + "\n"
        )
        return completed

    monkeypatch.setattr(local_generated_validator.subprocess, "run", forge)
    completed = run_local_generated_validator(_config(tmp_path))
    tasks = json.loads(completed.summary_path.read_text())["tasks"]
    assert tasks[0]["error_classification"] == "agent"
    assert tasks[0]["error_detail"] == "agent_error detail"
    assert "duration_seconds" not in tasks[0]
    json.dumps(tasks, allow_nan=False)


@pytest.mark.parametrize("kind", ["fifo", "directory", "hardlink", "parent_symlink"])
def test_sandbox_output_reader_rejects_unsafe_files(tmp_path: Path, kind: str) -> None:
    output = tmp_path / "sandbox_output.jsonl"
    if kind == "fifo":
        os.mkfifo(output)
    elif kind == "directory":
        output.mkdir()
    elif kind == "hardlink":
        target = tmp_path / "evaluator.json"
        target.write_text("{}\n")
        os.link(target, output)
    else:
        target = tmp_path / "actual"
        target.mkdir()
        (target / output.name).write_text("{}\n")
        alias = tmp_path / "alias"
        alias.symlink_to(target, target_is_directory=True)
        output = alias / output.name
    with pytest.raises(ValueError, match="sandbox output"):
        local_generated_validator._read_sandbox_rows(output)


def test_sandbox_output_reader_rejects_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "evaluator.json"
    target.write_text('{"trusted": true}\n')
    output = tmp_path / "sandbox_output.jsonl"
    output.symlink_to(target)
    with pytest.raises(ValueError, match="sandbox output"):
        local_generated_validator._read_sandbox_rows(output)


def test_sandbox_output_reader_bounds_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "sandbox_output.jsonl"
    output.write_text('{"problem_id": "session-0"}\n')
    assert local_generated_validator._read_sandbox_rows(output) == [
        {"problem_id": "session-0"}
    ]
    monkeypatch.setattr(local_generated_validator, "MAX_SANDBOX_OUTPUT_BYTES", 8)
    with pytest.raises(ValueError, match="sandbox output"):
        local_generated_validator._read_sandbox_rows(output)


@pytest.mark.parametrize("content", ["[]\n", "null\n", '"agent text"\n'])
def test_sandbox_output_reader_requires_object_rows(
    tmp_path: Path, content: str
) -> None:
    output = tmp_path / "sandbox_output.jsonl"
    output.write_text(content)
    with pytest.raises(ValueError, match="sandbox output"):
        local_generated_validator._read_sandbox_rows(output)


def test_run_composes_generated_components_and_writes_per_family_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    families = sorted(GENERATED_FAMILIES)
    results = [
        _episode(index, family, correct=index != 0, reward=0.5)
        for index, family in enumerate(families)
    ]
    _pack_value, registry, commands, _runtime, _server = _install_runtime_fakes(
        monkeypatch,
        tmp_path,
        results=results,
    )

    completed = run_local_generated_validator(_config(tmp_path))

    assert completed.aggregate_score == pytest.approx(3 / 7)
    assert completed.results == results
    assert commands[0][:2] == ["docker", "run"]
    assert registry.start.call_count == 7
    summary = json.loads(completed.summary_path.read_text(encoding="utf-8"))
    assert summary["aggregate_score"] == pytest.approx(3 / 7)
    assert summary["family_rewards"] == {
        family: (0.0 if index == 0 else 0.5) for index, family in enumerate(families)
    }
    assert [task["family"] for task in summary["tasks"]] == families
    assert summary["tasks"][0] == {
        "task_id": "task-0",
        "family": families[0],
        "outcome": "completed",
        "correct": False,
        "reward": 0.0,
        "error_classification": None,
        "error_detail": None,
    }
    assert (
        len((completed.artifact_dir / "problems.jsonl").read_text().splitlines()) == 7
    )
    bootstraps = json.loads(
        (completed.artifact_dir / "environment_sessions.json").read_text()
    )
    assert len(bootstraps["sessions"]) == 7
    episode_artifacts = [
        json.loads(line)
        for line in (completed.artifact_dir / "episode_results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert episode_artifacts == [
        {"schema_version": "oro.environment_episode.v1", "episode": result}
        for result in results
    ]


def test_sandbox_inherits_provider_tokens_without_putting_values_in_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        _episode(index, family)
        for index, family in enumerate(sorted(GENERATED_FAMILIES))
    ]
    _pack_value, _registry, commands, _runtime, _server = _install_runtime_fakes(
        monkeypatch,
        tmp_path,
        results=results,
    )
    original_run = local_generated_validator.subprocess.run
    sandbox_environment: dict[str, str] = {}

    def capture_environment(command: list[str], **kwargs: object) -> object:
        environment = kwargs.get("env")
        if isinstance(environment, dict):
            sandbox_environment.update(environment)
        return original_run(command, **kwargs)

    monkeypatch.setattr(
        local_generated_validator.subprocess,
        "run",
        capture_environment,
    )

    run_local_generated_validator(_config(tmp_path))

    sandbox_command = commands[0]
    assert all("test-token" not in argument for argument in sandbox_command)
    assert not any("CHUTES_ACCESS_TOKEN" in argument for argument in sandbox_command)
    assert ["-e", "INFERENCE_ACCESS_TOKEN"] == sandbox_command[
        sandbox_command.index("INFERENCE_ACCESS_TOKEN") - 1 : sandbox_command.index(
            "INFERENCE_ACCESS_TOKEN"
        )
        + 1
    ]
    assert "CHUTES_ACCESS_TOKEN" not in sandbox_environment
    assert sandbox_environment["INFERENCE_ACCESS_TOKEN"] == "test-token"
    assert sandbox_environment["SANDBOX_MODEL"] == "vendor/test-model"
    assert ["-e", "SANDBOX_MODEL"] == sandbox_command[
        sandbox_command.index("SANDBOX_MODEL") - 1 : sandbox_command.index(
            "SANDBOX_MODEL"
        )
        + 1
    ]


def test_summary_ignores_sandbox_inference_failure_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    families = sorted(GENERATED_FAMILIES)
    results = [
        _episode(0, families[0], outcome="agent_error"),
        _episode(1, families[1], outcome="agent_error"),
        *[_episode(index, family) for index, family in enumerate(families[2:], 2)],
    ]
    _pack_value, _registry, _commands, _runtime, _server = _install_runtime_fakes(
        monkeypatch,
        tmp_path,
        results=results,
    )
    original_run = local_generated_validator.subprocess.run

    def run_with_inference_failure(command: list[str], **kwargs: object) -> object:
        completed = original_run(command, **kwargs)
        run_dir = next((tmp_path / "runs").glob("local-*"))
        rows = [
            json.loads(line)
            for line in (run_dir / "sandbox" / "sandbox_output.jsonl")
            .read_text()
            .splitlines()
        ]
        rows[0]["inference_failure_count"] = 1
        (run_dir / "sandbox" / "sandbox_output.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return completed

    monkeypatch.setattr(
        local_generated_validator.subprocess, "run", run_with_inference_failure
    )

    completed = run_local_generated_validator(_config(tmp_path))

    summary = json.loads(completed.summary_path.read_text())
    assert summary["tasks"][0]["error_classification"] == "agent"
    assert summary["tasks"][1]["error_classification"] == "agent"
    assert completed.aggregate_score == pytest.approx(5 / 7)


def test_isolated_verifier_failure_completes_run_scoring_task_as_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One verifier_error among otherwise-good tasks stays inside the
    aggregate's harness tolerance: the run completes, the failing task
    counts as zero reward in the average, and the summary still labels
    the task with an error classification for diagnostics."""
    families = sorted(GENERATED_FAMILIES)
    results = [
        _episode(0, families[0], outcome="verifier_error"),
        *[_episode(index, family) for index, family in enumerate(families[1:], 1)],
    ]
    _install_runtime_fakes(monkeypatch, tmp_path, results=results)

    completed = run_local_generated_validator(_config(tmp_path))

    summary = json.loads(completed.summary_path.read_text())
    assert summary["tasks"][0]["error_classification"] == "verifier"
    assert completed.aggregate_score == pytest.approx(
        sum(1.0 for _ in families[1:]) / len(families)
    )


def test_verifier_failures_above_tolerance_still_fail_the_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Above the aggregate's harness tolerance the whole run still
    hard-fails and the diagnostic summary marks each failing task."""
    families = sorted(GENERATED_FAMILIES)
    results = [
        *[
            _episode(index, family, outcome="verifier_error")
            for index, family in enumerate(families[:3])
        ],
        *[_episode(index, family) for index, family in enumerate(families[3:], 3)],
    ]
    _install_runtime_fakes(monkeypatch, tmp_path, results=results)

    with pytest.raises(LocalGeneratedValidatorError) as caught:
        run_local_generated_validator(_config(tmp_path))

    assert caught.value.classification == "verifier"
    summary = _failure_summary(caught.value)
    assert summary["tasks"][0]["error_classification"] == "verifier"


def test_missing_finalized_task_is_an_infrastructure_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        _episode(index, family)
        for index, family in enumerate(sorted(GENERATED_FAMILIES)[:-1])
    ]
    _install_runtime_fakes(monkeypatch, tmp_path, results=results)

    with pytest.raises(LocalGeneratedValidatorError) as caught:
        run_local_generated_validator(_config(tmp_path))

    assert caught.value.classification == "infrastructure"
    assert "one finalized result per pack task" in str(caught.value)
    _failure_summary(caught.value)


@pytest.mark.parametrize("failure_point", ["session_start", "problem_write"])
def test_preinstall_failure_closes_registry_and_pack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    results = [
        _episode(index, family)
        for index, family in enumerate(sorted(GENERATED_FAMILIES))
    ]
    pack, registry, _commands, runtime, _server = _install_runtime_fakes(
        monkeypatch,
        tmp_path,
        results=results,
    )
    if failure_point == "session_start":
        registry.start.side_effect = RuntimeError("session creation failed")
    else:
        monkeypatch.setattr(
            local_generated_validator,
            "write_problem_file",
            MagicMock(side_effect=OSError("problem write failed")),
        )

    with pytest.raises(LocalGeneratedValidatorError) as caught:
        run_local_generated_validator(_config(tmp_path))

    assert runtime.ready is False
    registry.close.assert_called_once_with()
    pack.close.assert_called_once_with()
    assert caught.value.classification == "infrastructure"
    _failure_summary(caught.value)


def test_problem_execution_budget_does_not_cut_off_finalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        _episode(index, family)
        for index, family in enumerate(sorted(GENERATED_FAMILIES))
    ]
    _pack_value, registry, _commands, _runtime, _server = _install_runtime_fakes(
        monkeypatch,
        tmp_path,
        results=results,
    )
    original_run = local_generated_validator.subprocess.run

    def finish_sandbox(command: list[str], **kwargs: object) -> object:
        completed = original_run(command, **kwargs)
        time.sleep(0.02)
        return completed

    monkeypatch.setattr(
        local_generated_validator.subprocess,
        "run",
        finish_sandbox,
    )

    completed = run_local_generated_validator(_config(tmp_path, timeout=0.01))

    registry.finalized_results.assert_called_once_with()
    assert completed.aggregate_score == pytest.approx(1.0)


def test_finalization_and_cleanup_complete_synchronously_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        _episode(index, family)
        for index, family in enumerate(sorted(GENERATED_FAMILIES))
    ]
    pack, registry, _commands, _runtime, _server = _install_runtime_fakes(
        monkeypatch,
        tmp_path,
        results=results,
    )
    lifecycle: list[str] = []

    def finalize() -> list[dict]:
        lifecycle.append("finalization-started")
        time.sleep(0.03)
        lifecycle.append("finalization-finished")
        return results

    def close() -> None:
        lifecycle.append("cleanup-started")
        time.sleep(0.03)
        pack.close()
        lifecycle.append("cleanup-finished")

    registry.finalized_results.side_effect = finalize
    registry.close.side_effect = close
    started_at = time.monotonic()
    completed = run_local_generated_validator(_config(tmp_path, timeout=0.01))
    elapsed = time.monotonic() - started_at

    assert completed.aggregate_score == pytest.approx(1.0)
    assert lifecycle == [
        "finalization-started",
        "finalization-finished",
        "cleanup-started",
        "cleanup-finished",
    ]
    assert elapsed >= 0.06
    registry.close.assert_called_once_with()
    pack.close.assert_called_once_with()


def test_interrupted_sandbox_run_cleans_up_before_propagating_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        _episode(index, family)
        for index, family in enumerate(sorted(GENERATED_FAMILIES))
    ]
    pack, registry, _commands, _runtime, server = _install_runtime_fakes(
        monkeypatch,
        tmp_path,
        results=results,
    )
    commands: list[list[str]] = []

    def interrupt(command: list[str], **_kwargs: object) -> SimpleNamespace:
        commands.append(command)
        if command[:2] == ["docker", "run"]:
            raise KeyboardInterrupt()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(local_generated_validator.subprocess, "run", interrupt)

    with pytest.raises(KeyboardInterrupt):
        run_local_generated_validator(_config(tmp_path))

    registry.finalized_results.assert_called_once_with()
    server.stop.assert_called_once_with()
    registry.close.assert_called_once_with()
    pack.close.assert_called_once_with()
    assert commands[1][:3] == ["docker", "rm", "--force"]


def test_registry_cleanup_failure_fails_run_and_writes_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        _episode(index, family)
        for index, family in enumerate(sorted(GENERATED_FAMILIES))
    ]
    _pack_value, registry, _commands, _runtime, _server = _install_runtime_fakes(
        monkeypatch,
        tmp_path,
        results=results,
    )
    registry.close.side_effect = [RuntimeError("registry cleanup failed"), None]

    with pytest.raises(LocalGeneratedValidatorError) as caught:
        run_local_generated_validator(_config(tmp_path))

    assert caught.value.classification == "infrastructure"
    assert "registry cleanup failed" in str(caught.value)
    _failure_summary(caught.value)


@pytest.mark.parametrize(
    ("failure_point", "expected_message"),
    [
        ("server", "session server failed"),
        ("sandbox_launch", "docker executable failed"),
        ("sandbox_nonzero", "sandbox exited with code 17"),
        ("missing_output", "sandbox did not produce output"),
        ("malformed_output", "could not read sandbox output"),
        ("finalization", "finalization failed"),
    ],
)
def test_started_run_failures_write_classified_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
    expected_message: str,
) -> None:
    results = [
        _episode(index, family)
        for index, family in enumerate(sorted(GENERATED_FAMILIES))
    ]
    _pack_value, registry, _commands, _runtime, server = _install_runtime_fakes(
        monkeypatch,
        tmp_path,
        results=results,
    )
    original_run = local_generated_validator.subprocess.run
    if failure_point == "server":
        server.start.side_effect = RuntimeError("session server failed")
    elif failure_point == "sandbox_launch":
        monkeypatch.setattr(
            local_generated_validator.subprocess,
            "run",
            MagicMock(side_effect=OSError("docker executable failed")),
        )
    elif failure_point == "sandbox_nonzero":
        monkeypatch.setattr(
            local_generated_validator.subprocess,
            "run",
            lambda command, **kwargs: SimpleNamespace(returncode=17),
        )
    elif failure_point == "missing_output":
        monkeypatch.setattr(
            local_generated_validator.subprocess,
            "run",
            lambda command, **kwargs: SimpleNamespace(returncode=0),
        )
    elif failure_point == "malformed_output":

        def malformed(command: list[str], **kwargs: object) -> object:
            completed = original_run(command, **kwargs)
            run_dir = next((tmp_path / "runs").glob("local-*"))
            (run_dir / "sandbox" / "sandbox_output.jsonl").write_text("not-json\n")
            return completed

        monkeypatch.setattr(local_generated_validator.subprocess, "run", malformed)
    else:
        registry.finalized_results.side_effect = RuntimeError("finalization failed")

    with pytest.raises(LocalGeneratedValidatorError) as caught:
        run_local_generated_validator(_config(tmp_path))

    assert expected_message in str(caught.value)
    assert caught.value.classification == (
        "verifier" if failure_point == "finalization" else "infrastructure"
    )
    _failure_summary(caught.value)


def test_session_server_construction_failure_writes_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local_generated_validator,
        "SessionServer",
        MagicMock(side_effect=RuntimeError("session server construction failed")),
    )

    with pytest.raises(LocalGeneratedValidatorError) as caught:
        run_local_generated_validator(_config(tmp_path))

    assert caught.value.classification == "infrastructure"
    assert "session server construction failed" in str(caught.value)
    _failure_summary(caught.value)


def test_pack_search_failure_writes_search_classified_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local_generated_validator,
        "load_local_pack",
        MagicMock(
            side_effect=PackValidationError(
                "sealed epoch validation failed: search identity mismatch"
            )
        ),
    )
    monkeypatch.setattr(
        local_generated_validator,
        "SessionServer",
        lambda *_args, **_kwargs: MagicMock(),
    )
    sandbox_run = MagicMock()
    monkeypatch.setattr(local_generated_validator.subprocess, "run", sandbox_run)

    with pytest.raises(LocalGeneratedValidatorError) as caught:
        run_local_generated_validator(_config(tmp_path))

    assert caught.value.classification == "search"
    sandbox_run.assert_not_called()
    summary = _failure_summary(caught.value)
    assert summary["aggregate_score"] is None


def test_problem_execution_timeout_uses_one_budget_across_worker_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(local_generated_validator, "QUALIFYING_TASKS_PER_FAMILY", 1)
    results = [
        _episode(index, family)
        for index, family in enumerate(sorted(GENERATED_FAMILIES))
    ]
    pack = _pack(sorted(GENERATED_FAMILIES))
    registry = MagicMock()
    registry.start.side_effect = [
        {
            "session_id": f"session-{index}",
            "policy_view": {"query": family, "tool_contract_version": "v1"},
        }
        for index, family in enumerate(sorted(GENERATED_FAMILIES))
    ]
    registry.finalized_results.return_value = results

    def load_pack(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return pack

    monkeypatch.setattr(local_generated_validator, "load_local_pack", load_pack)
    monkeypatch.setattr(local_generated_validator, "SessionRuntime", MagicMock)
    monkeypatch.setattr(
        local_generated_validator, "SessionServer", lambda *_a, **_k: MagicMock()
    )
    monkeypatch.setattr(
        local_generated_validator, "SessionRegistry", lambda *_a, **_k: registry
    )
    calls: list[tuple[list[str], dict[str, object]]] = []

    def time_out(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        if command[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(local_generated_validator.subprocess, "run", time_out)

    with pytest.raises(LocalGeneratedValidatorError) as caught:
        run_local_generated_validator(_config(tmp_path, timeout=10.0))

    sandbox_command, sandbox_kwargs = calls[0]
    timeout_index = sandbox_command.index("--timeout")
    assert float(sandbox_command[timeout_index + 1]) == pytest.approx(10.0)
    assert sandbox_kwargs["timeout"] == pytest.approx(70.0)
    assert calls[1][0][:3] == ["docker", "rm", "--force"]
    assert calls[1][1]["timeout"] == 30
    assert caught.value.classification == "infrastructure"
    _failure_summary(caught.value)
