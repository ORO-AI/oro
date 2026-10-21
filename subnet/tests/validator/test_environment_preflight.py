"""Environment startup preflight on the bundled, freshly signed neutral world."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from oro_env_runtime import RUNTIME_CONTRACT
from oro_env_runtime.pack import PackRefused, load_pack
from oro_pack_tools.build import validate_pack
from validator import environment_preflight, environment_preflight_agent
from validator.environment_preflight import (
    POLICY_ACTIONS,
    build_preflight_pack,
    run_environment_preflight,
)
from validator.session_service import SessionRuntime


def _in_process_sandbox_runner(runtime: SessionRuntime, *, exit_code: int = 0, actions=None):
    def run(_agent_path: Path, _run_id: str, problem_path: Path) -> tuple[Path, dict]:
        output_path = problem_path.with_name("output.jsonl")
        rows = []
        for line in problem_path.read_text(encoding="utf-8").splitlines():
            problem = json.loads(line)
            environment = problem["environment"]
            for step, action in enumerate(actions or environment["actions"], start=1):
                runtime.call({**environment["binding"], "step": step, "action": action})
            rows.append({"problem_id": problem["problem_id"], "status": "SUCCESS", "inference_total": 0})
        output_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return output_path, {"exit_code": exit_code}

    return run


def test_preflight_world_is_a_valid_pack_signed_with_a_fresh_key(tmp_path: Path) -> None:
    key = Ed25519PrivateKey.generate()
    with mock.patch.object(Ed25519PrivateKey, "generate", return_value=key):
        pack = build_preflight_pack(tmp_path / "a")

    assert validate_pack(tmp_path / "a", {"validator-preflight": key.public_key()}).pack_id == pack.pack_id
    assert pack.manifest["runtime_contract"] == RUNTIME_CONTRACT
    with pytest.raises(PackRefused):
        load_pack(tmp_path / "a")  # never trusted by the pinned keyring
    other = build_preflight_pack(tmp_path / "b")
    assert other.manifest["signature"]["value"] != pack.manifest["signature"]["value"]


def test_preflight_passes_both_policies_and_replays(tmp_path: Path) -> None:
    runtime = SessionRuntime()

    summary = run_environment_preflight(
        runtime=runtime,
        run_dir=tmp_path / "preflight",
        sandbox_runner=_in_process_sandbox_runner(runtime),
    )

    assert summary["status"] == "pass"
    assert summary["runtime_contract"] == RUNTIME_CONTRACT
    assert summary["sandbox"] == {"exit_code": 0, "problem_count": 2, "inference_call_count": 0}
    assert 0 <= summary["timing_ms"]["sandbox"] <= summary["timing_ms"]["total"]
    assert summary["policies"] == [
        {"policy": "positive", "score": 1.0, "category": None, "finish_reason": "order", "steps": 2, "replay_matches": True},
        {"policy": "negative", "score": 0.0, "category": "request not met", "finish_reason": "order", "steps": 2,
         "replay_matches": True},
    ]
    assert runtime.ready is False
    saved = json.loads((tmp_path / "preflight/environment_preflight_result.json").read_text(encoding="utf-8"))
    assert saved == summary
    problems = [
        json.loads(line)
        for line in (tmp_path / "preflight/problems.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [p["environment"]["actions"] for p in problems] == list(POLICY_ACTIONS.values())
    assert all(p["environment"]["policy_view"]["max_steps"] == 4 for p in problems)


def test_preflight_fails_when_a_verdict_does_not_discriminate(tmp_path: Path) -> None:
    runtime = SessionRuntime()

    with pytest.raises(environment_preflight.EnvironmentPreflightError, match='"status": "fail"'):
        run_environment_preflight(
            runtime=runtime,
            run_dir=tmp_path / "preflight",
            sandbox_runner=_in_process_sandbox_runner(runtime, actions=POLICY_ACTIONS["positive"]),
        )
    assert runtime.ready is False


def test_preflight_clears_runtime_when_sandbox_fails(tmp_path: Path) -> None:
    runtime = SessionRuntime()

    def fail(*_args):  # noqa: ANN002, ANN202
        raise RuntimeError("sandbox unavailable")

    with pytest.raises(RuntimeError, match="sandbox unavailable"):
        run_environment_preflight(runtime=runtime, run_dir=tmp_path / "preflight", sandbox_runner=fail)

    assert runtime.ready is False


def test_preflight_rejects_nonzero_sandbox_exit(tmp_path: Path) -> None:
    runtime = SessionRuntime()

    with pytest.raises(environment_preflight.EnvironmentPreflightError):
        run_environment_preflight(
            runtime=runtime,
            run_dir=tmp_path / "preflight",
            sandbox_runner=_in_process_sandbox_runner(runtime, exit_code=1),
        )

    assert runtime.ready is False


class _Response:
    def __init__(self, body: dict) -> None:
        self.body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.body


def test_reference_agent_posts_one_step_per_action(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = []

    def post(url, *, json, timeout):  # noqa: ANN001, ANN201
        requests.append((url, json, timeout))
        return _Response({"observation": {}, "finished": json["step"] == 2})

    monkeypatch.setattr(environment_preflight_agent.requests, "post", post)
    dialogue = environment_preflight_agent.agent_main(
        {"problem_id": "positive", "environment": {"binding": {"session_id": "s"}, "actions": POLICY_ACTIONS["positive"]}}
    )

    assert len(dialogue) == 2
    assert [r[0].rsplit("/", 2)[-2:] for r in requests] == [["environment", "call"]] * 2
    assert [r[1] for r in requests] == [
        {"session_id": "s", "step": step, "action": action}
        for step, action in enumerate(POLICY_ACTIONS["positive"], start=1)
    ]
    assert requests[0][2] == 70


def test_reference_agent_requires_the_last_step_to_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        environment_preflight_agent.requests,
        "post",
        lambda *_a, **_k: _Response({"observation": {}, "finished": False}),
    )
    with pytest.raises(RuntimeError, match="step 2 finished=False"):
        environment_preflight_agent.agent_main(
            {"problem_id": "p", "environment": {"binding": {}, "actions": POLICY_ACTIONS["negative"]}}
        )
