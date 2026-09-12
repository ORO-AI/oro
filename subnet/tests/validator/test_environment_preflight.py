"""Validator-owned environment preflight coverage."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from validator import environment_preflight, environment_preflight_agent, main
from validator.env_pack_loader import PACK_VERSION_IDENTITIES, LoadedPack
from validator.environment_preflight import run_environment_preflight
from validator.session_service import SessionRuntime

pytest_plugins = ("tests.compat_fixture",)


def _in_process_sandbox_runner(runtime: SessionRuntime, *, exit_code: int = 0):
    def run(
        _agent_path: Path,
        _run_id: str,
        problem_path: Path,
    ) -> tuple[Path, dict]:
        output_path = problem_path.with_name("output.jsonl")
        output_rows = []
        for line in problem_path.read_text(encoding="utf-8").splitlines():
            problem = json.loads(line)
            binding = problem["environment"]["binding"]
            for turn, actions in enumerate(
                problem["environment"]["action_groups"], start=1
            ):
                group_id = f"{problem['problem_id']}-{turn}"
                response = runtime.call(
                    {
                        **binding,
                        "call_id": group_id,
                        "idempotency_key": group_id,
                        "turn": turn,
                        "calls": [
                            {
                                "call_id": f"{group_id}-{index}",
                                "action": action,
                            }
                            for index, action in enumerate(actions, start=1)
                        ],
                    }
                )
                if turn == 1:
                    assert response["user_message"]["content"]
            output_rows.append(
                {
                    "problem_id": problem["problem_id"],
                    "status": "SUCCESS",
                    "inference_total": 0,
                    "dialogue": [],
                }
            )
        output_path.write_text(
            "".join(json.dumps(row) + "\n" for row in output_rows),
            encoding="utf-8",
        )
        return output_path, {"exit_code": exit_code}

    return run


def test_preflight_proves_positive_negative_and_replay(
    loaded_pack: LoadedPack,
    tmp_path: Path,
) -> None:
    runtime = SessionRuntime()

    summary = run_environment_preflight(
        loaded_pack=loaded_pack,
        runtime=runtime,
        run_dir=tmp_path / "preflight",
        sandbox_runner=_in_process_sandbox_runner(runtime),
    )

    assert summary["status"] == "pass"
    assert summary["sandbox"] == {
        "exit_code": 0,
        "problem_count": 2,
        "inference_call_count": 0,
    }
    assert summary["versions"] == PACK_VERSION_IDENTITIES
    assert 0 <= summary["timing_ms"]["sandbox"] <= summary["timing_ms"]["total"]
    positive, negative = summary["policies"]
    assert len(positive["terminal_state_hash"]) == 64
    assert len(negative["terminal_state_hash"]) == 64
    assert positive["terminal_state_hash"] != negative["terminal_state_hash"]
    assert positive["verdict_correct"] is True
    assert positive["solver_turn_count"] == 2
    assert positive["action_count"] == 4
    assert negative["verdict_correct"] is False
    assert negative["solver_turn_count"] == 2
    assert negative["action_count"] == 4
    assert all(positive["replay_checks"].values())
    assert all(negative["replay_checks"].values())
    assert runtime.ready is False
    saved = json.loads(
        (tmp_path / "preflight/environment_preflight_result.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved == summary
    bootstrap = json.loads(
        (tmp_path / "preflight/environment_sessions.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(bootstrap) == {"schema_version", "sessions"}
    assert all(
        set(session) == {"session_id", "policy_view"}
        for session in bootstrap["sessions"]
    )


def test_select_reference_case_is_family_agnostic(loaded_pack: LoadedPack) -> None:
    """The selector picks any family, not only retrieval_recall. The
    fixture's first task is intent_decomposition; the old filter skipped it."""
    assert loaded_pack.task_specs[0].family != "retrieval_recall"

    task_id, task, targets = environment_preflight._select_reference_case(loaded_pack)

    assert task.family == loaded_pack.task_specs[0].family
    assert task.family != "retrieval_recall"
    positive_targets, negative = targets
    assert positive_targets and negative is not None


def test_select_reference_case_requires_a_discriminating_case(
    loaded_pack: LoadedPack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No discriminating case -> the selector refuses rather than false-pass."""
    monkeypatch.setattr(
        environment_preflight, "_case_discriminates", lambda *_args: False
    )
    with pytest.raises(
        environment_preflight.EnvironmentPreflightError,
        match="self-verifiable positive/negative",
    ):
        environment_preflight._select_reference_case(loaded_pack)


def test_preflight_clears_runtime_when_sandbox_fails(
    loaded_pack: LoadedPack,
    tmp_path: Path,
) -> None:
    runtime = SessionRuntime()

    def fail(*_args):  # noqa: ANN002, ANN202
        raise RuntimeError("sandbox unavailable")

    with pytest.raises(RuntimeError, match="sandbox unavailable"):
        run_environment_preflight(
            loaded_pack=loaded_pack,
            runtime=runtime,
            run_dir=tmp_path / "preflight",
            sandbox_runner=fail,
        )

    assert runtime.ready is False


def test_preflight_rejects_nonzero_sandbox_exit(
    loaded_pack: LoadedPack,
    tmp_path: Path,
) -> None:
    runtime = SessionRuntime()

    with pytest.raises(environment_preflight.EnvironmentPreflightError):
        run_environment_preflight(
            loaded_pack=loaded_pack,
            runtime=runtime,
            run_dir=tmp_path / "preflight",
            sandbox_runner=_in_process_sandbox_runner(runtime, exit_code=1),
        )

    assert runtime.ready is False


def test_reference_agent_groups_calls_and_requires_shopper_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"user_message": {"content": "continue"}}

    def post(url, *, json, timeout):  # noqa: ANN001, ANN201
        requests.append((url, json, timeout))
        return Response()

    monkeypatch.setattr(environment_preflight_agent.requests, "post", post)
    dialogue = environment_preflight_agent.agent_main(
        {
            "problem_id": "positive",
            "environment": {
                "binding": {"session_id": "session"},
                "action_groups": [
                    [
                        {"name": "search", "args": {"query": "shoes"}},
                        {"name": "message", "args": {"content": "Continue?"}},
                    ]
                ],
            },
        }
    )

    assert len(dialogue) == 1
    assert dialogue[0]["environment_result"]["user_message"] == {
        "content": "continue"
    }
    assert requests[0][0].endswith("/environment/call")
    assert requests[0][1]["turn"] == 1
    assert [call["call_id"] for call in requests[0][1]["calls"]] == [
        "positive-1-1",
        "positive-1-2",
    ]
    assert requests[0][2] == 70


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, False), ("true", True), ("YES", True), ("0", False), ("no", False)],
)
def test_environment_flag_parser(
    monkeypatch: pytest.MonkeyPatch,
    raw: str | None,
    expected: bool,
) -> None:
    if raw is None:
        monkeypatch.delenv("TEST_ENVIRONMENT_FLAG", raising=False)
    else:
        monkeypatch.setenv("TEST_ENVIRONMENT_FLAG", raw)
    assert main._parse_env_bool("TEST_ENVIRONMENT_FLAG") is expected


def test_disabled_feature_flag_does_not_fetch_pack() -> None:
    validator = main.Validator.__new__(main.Validator)
    validator.config = SimpleNamespace(environment_runtime_enabled=False)

    assert validator._run_environment_preflight_if_enabled() is None


def test_enabled_feature_flag_uses_validator_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pack_sha256 = "b" * 64
    loaded_pack = object()
    runtime = object()
    sandbox_runner = MagicMock()
    expected = {"status": "pass"}
    log_messages = []

    async def fetch(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        assert args[:2] == (pack_sha256, "https://backend.example")
        assert kwargs["download_url_rewriter"]("http://localhost:4566/pack") == (
            "http://host.docker.internal:4566/pack"
        )
        return loaded_pack

    def run(**kwargs):  # noqa: ANN003, ANN201
        assert kwargs["loaded_pack"] is loaded_pack
        assert kwargs["runtime"] is runtime
        assert kwargs["sandbox_runner"] is sandbox_runner
        return expected

    monkeypatch.setattr(main, "fetch_and_validate_pack", fetch)
    monkeypatch.setattr(main, "run_environment_preflight", run)
    monkeypatch.setattr(main.logging, "info", log_messages.append)
    validator = main.Validator.__new__(main.Validator)
    validator.config = SimpleNamespace(
        environment_runtime_enabled=True,
        environment_preflight_pack_sha256=pack_sha256,
        backend_url="https://backend.example",
        workspace_dir="/tmp/workspace",
    )
    validator.wallet = SimpleNamespace(hotkey=object())
    validator.session_runtime = runtime
    validator.run_sandbox = sandbox_runner
    validator._eval_dir = MagicMock(return_value=Path("/tmp/preflight"))

    assert validator._run_environment_preflight_if_enabled() is expected
    validator._eval_dir.assert_called_once_with(
        environment_preflight.environment_preflight_run_id(pack_sha256)
    )
    assert (
        f"Environment preflight receipt: {json.dumps(expected, sort_keys=True)}"
        in log_messages
    )


def test_enabled_feature_flag_requires_valid_pack_sha() -> None:
    validator = main.Validator.__new__(main.Validator)
    validator.config = SimpleNamespace(
        environment_runtime_enabled=True,
        environment_preflight_pack_sha256="bad",
    )

    with pytest.raises(RuntimeError, match="64 lowercase hex"):
        validator._run_environment_preflight_if_enabled()


def test_finalize_submits_all_sessions_and_clears_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pack_sha256 = "b" * 64
    work = SimpleNamespace(
        eval_run_id="aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
        env_pack_sha256=pack_sha256,
    )
    registry = MagicMock()
    results = [{"task_id": "task-1"}]
    registry.finalized_results.return_value = results
    emit = AsyncMock(return_value={"counts": {"201": 1}})
    monkeypatch.setattr(main, "emit_finalized_results", emit)

    validator = main.Validator.__new__(main.Validator)
    validator.config = SimpleNamespace(backend_url="https://backend.example")
    validator.wallet = SimpleNamespace(hotkey=object())
    validator.session_runtime = MagicMock()

    validator.finalize_environment_sessions(work, registry)
    emit.assert_awaited_once_with(
        backend_url="https://backend.example",
        validator_keypair=validator.wallet.hotkey,
        env_pack_sha256=pack_sha256,
        results=results,
        download_url_rewriter=main._rewrite_localhost_url,
    )
    validator.session_runtime.clear.assert_called_once_with(registry)


def test_finalize_failure_cannot_change_legacy_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = SimpleNamespace(
        eval_run_id="aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
        env_pack_sha256="b" * 64,
    )
    registry = MagicMock()
    monkeypatch.setattr(
        main,
        "emit_finalized_results",
        AsyncMock(side_effect=TypeError("schema drift")),
    )
    validator = main.Validator.__new__(main.Validator)
    validator.config = SimpleNamespace(backend_url="https://backend.example")
    validator.wallet = SimpleNamespace(hotkey=object())
    validator.session_runtime = MagicMock()
    validator.session_runtime.clear.side_effect = RuntimeError("cleanup failed")

    validator.finalize_environment_sessions(work, registry)
    validator.session_runtime.clear.assert_called_once_with(registry)
