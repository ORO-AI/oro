from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from subnet import local_generated_validator as local
from subnet.tests.world_pack_fixture import (
    StubModelClient,
    StubSearchClient,
    archive_pack,
    build_world_pack,
    world_keyring,  # noqa: F401
)
from subnet.validator import session_registry

# The test world's answers: buying them scores 1.0.
ANSWERS = {"t1": "a", "t2": "b"}


@pytest.fixture
def services(monkeypatch, world_keyring):  # noqa: F811
    """Stub the model proxy and search server the way the e2e driver wraps them."""

    models = StubModelClient()
    search = StubSearchClient()
    built = []

    class CountingClient(session_registry.ProxyModelClient):
        def __init__(self, token, **kwargs):
            built.append((token, kwargs))

        def complete(self, model, prompt, params, timeout):
            return models.complete(model, prompt, params, timeout)

        def decide(self, model, state, questions, params, timeout):
            return models.decide(model, state, questions, params, timeout)

    # Patched after import, exactly as the driver does: looked up at call time.
    monkeypatch.setattr(session_registry, "ProxyModelClient", CountingClient)
    urls = []
    monkeypatch.setattr(
        session_registry,
        "SearchServerClient",
        lambda url, *args: urls.append(url) or search,
    )
    return SimpleNamespace(models=models, search=search, built=built, urls=urls)


@pytest.fixture
def sandbox(monkeypatch):
    """Fake the sandbox: an agent that buys each task's answer (or a chosen item)."""

    state = SimpleNamespace(runtime=None, buys=dict(ANSWERS), returncode=0, commands=[],
                            write_output=True, raise_=None, output_listings=[])

    class FakeServer:
        def __init__(self, runtime, *, host, port):
            state.runtime = runtime

        def start(self, *, timeout):
            pass

        def stop(self):
            pass

    def run(command, **kwargs):
        if command[:2] == ["docker", "rm"]:
            return subprocess.CompletedProcess(command, 0)
        state.commands.append((command, kwargs))
        if state.raise_ is not None:
            raise state.raise_
        logs = Path(next(arg.split(":")[0] for arg in command if arg.endswith(":/app/logs:ro")))
        state.output_listings.append(sorted(p.name for p in (logs / "sandbox").iterdir()))
        (logs / "sandbox" / "left-by-the-agent.txt").write_text("attempt state")
        rows = [json.loads(line) for line in (logs / "problems.jsonl").read_text().splitlines()]
        for row in rows:
            task_id = row["environment"]["policy_view"]["query"].split()[-1].rstrip(".")
            if task_id in state.buys:
                try:
                    state.runtime.call({**row["environment"]["binding"], "step": 1,
                                        "action": {"name": "buy", "args": {"id": state.buys[task_id]}}})
                except Exception:  # noqa: BLE001  (the agent sees a failed call)
                    pass
        if state.write_output:
            (logs / "sandbox" / "sandbox_output.jsonl").write_text(
                "".join(json.dumps({"problem_id": row["problem_id"]}) + "\n" for row in rows)
            )
        return subprocess.CompletedProcess(command, state.returncode)

    monkeypatch.setattr(local, "SessionServer", FakeServer)
    monkeypatch.setattr(local.subprocess, "run", run)
    monkeypatch.setattr(local, "host_path", lambda path: path)
    return state


def _delivery(tmp_path: Path, **kwargs) -> Path:
    name = kwargs.get("delivery_kind", "practice")
    pack = build_world_pack(tmp_path / f"world-{name}", **kwargs)
    return archive_pack(pack, tmp_path / f"{name}.tar.gz")


def _config(tmp_path: Path, **overrides) -> local.LocalGeneratedConfig:
    agent = tmp_path / "agent.py"
    agent.write_text("def run(problem):\n    return {}\n")
    fields = {
        "agent_path": agent,
        "pack_path": _delivery(tmp_path),
        "output_root": tmp_path / "runs",
        "inference_access_token": "or-test-token",
        "inference_provider": "openrouter",
        "inference_base_url": "https://openrouter.ai/api/v1",
        "model": "vendor/model",
        "search_url": "http://search.test:5632",
    }
    return local.LocalGeneratedConfig(**{**fields, **overrides})


def _failure(config) -> local.LocalGeneratedValidatorError:
    with pytest.raises(local.LocalGeneratedValidatorError) as caught:
        local.run_local_generated_validator(config)
    return caught.value


def test_practice_run_writes_the_artifacts_the_e2e_report_reads(
    tmp_path, services, sandbox
) -> None:
    sandbox.buys["t2"] = "a"  # wrong item: "request not met"

    result = local.run_local_generated_validator(_config(tmp_path))

    art = result.artifact_dir
    summary = json.loads((art / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["aggregate_score"] == result.aggregate_score == 0.5
    assert summary["world_id"] == "world-validator-test"
    assert summary["task_count"] == 2 and summary["task_roster"] == ["t1", "t2"]
    assert summary["selection_mode"] == "all_tasks"
    assert summary["error"] is None
    assert [(t["task_id"], t["score"], t["category"], t["outcome"]) for t in summary["tasks"]] == [
        ("t1", 1.0, None, "completed"),
        ("t2", 0.0, "request not met", "completed"),
    ]
    episodes = [json.loads(line)["episode"] for line in (art / "episode_results.jsonl").read_text().splitlines()]
    sessions = json.loads((art / "environment_sessions.json").read_text())["sessions"]
    assert [e["task_id"] for e in episodes] == ["t1", "t2"]
    assert [e["session_id"] for e in episodes] == [s["session_id"] for s in sessions]
    assert all(e["step_count"] == 1 and e["terminal_reason"] == "order" for e in episodes)
    assert all(e["ledger"]["episode"][0]["delivery"]["id"] == "practice-1" for e in episodes)
    assert all(isinstance(e["wall_seconds"], float) for e in episodes)
    problems = [json.loads(line) for line in (art / "problems.jsonl").read_text().splitlines()]
    assert [p["environment"]["binding"] for p in problems] == [s["binding"] for s in sessions]
    assert (art / "sandbox" / "sandbox_output.jsonl").is_file()
    # The registry's model client was the patched class, on the run's key and the
    # local proxy; every index went to the configured search server first.
    assert services.built == [("or-test-token", {
        "proxy_url": local.LOCAL_PROXY_URL, "timeout": 60.0,
        "inference_stats_file": None, "key_exhausted_event": services.built[0][1]["key_exhausted_event"],
    })]
    assert services.urls == ["http://search.test:5632"]
    assert list(services.search.titles)


def test_each_episode_gets_a_fresh_nonce(tmp_path, services, sandbox) -> None:
    config = _config(tmp_path)
    nonces = []
    for _ in range(2):
        art = local.run_local_generated_validator(config).artifact_dir
        nonces.append({
            json.loads(line)["episode"]["episode_nonce"]
            for line in (art / "episode_results.jsonl").read_text().splitlines()
        })
    assert len(nonces[0] | nonces[1]) == 4


def test_unfinished_episodes_are_finished_and_graded(tmp_path, services, sandbox) -> None:
    sandbox.buys = {"t1": "a"}

    result = local.run_local_generated_validator(_config(tmp_path))

    assert result.aggregate_score == 0.5
    assert [t["category"] for t in result.summary["tasks"]] == [None, "did not finish"]


def test_an_environment_fault_reruns_the_episode_with_the_agent(tmp_path, services, sandbox, monkeypatch) -> None:
    real, faults = session_registry.runtime_session.grade_episode, [1]

    def grade(pack, ledger, read):
        verdict = real(pack, ledger, read)
        if faults[0]:
            faults[0] -= 1
            verdict = {**verdict, "score": None, "category": "environment fault", "flags": ["environment_fault"]}
        return verdict

    monkeypatch.setattr(session_registry.runtime_session, "grade_episode", grade)

    result = local.run_local_generated_validator(_config(tmp_path))

    assert len(sandbox.commands) == 2  # the run, then one re-run of the faulted episode
    assert sandbox.output_listings[1] == []  # nothing from the first attempt is visible
    assert result.aggregate_score == 1.0
    rows = (result.artifact_dir / "sandbox" / "sandbox_output.jsonl").read_text().splitlines()
    assert len(rows) == 3  # the first run's rows are kept


@pytest.mark.parametrize("kind", ["qualifying", "race", "full"])
def test_only_a_practice_delivery_runs_locally(tmp_path, services, sandbox, kind) -> None:
    error = _failure(_config(tmp_path, pack_path=_delivery(tmp_path, delivery_kind=kind)))

    assert error.classification == "configuration"
    assert "practice delivery" in str(error)
    assert sandbox.commands == []
    summary = json.loads(error.summary_path.read_text())
    assert summary["status"] == "failed"
    assert summary["error"]["classification"] == "configuration"


def test_untrusted_signature_is_refused(tmp_path, services, sandbox, monkeypatch) -> None:
    from subnet.validator import env_pack_loader
    from oro_env_runtime.pack import pinned_keyring

    monkeypatch.setattr(env_pack_loader, "KEYRING", pinned_keyring())

    error = _failure(_config(tmp_path))

    assert error.classification == "configuration"
    assert "refused" in str(error)


def test_race_configuration_refuses_to_run(tmp_path, services, sandbox, monkeypatch) -> None:
    monkeypatch.setenv("ORO_RACE_ID", "race")

    assert _failure(_config(tmp_path)).classification == "configuration"


def test_problem_count_samples_repeatably(tmp_path, services) -> None:
    from subnet.tests.world_pack_fixture import load_world_pack

    pack = load_world_pack(tmp_path)

    assert local.select_tasks(pack) == ["t1", "t2"]
    picks = {tuple(local.select_tasks(pack, problem_count=1, seed=seed)) for seed in range(20)}
    assert picks == {("t1",), ("t2",)}
    assert local.select_tasks(pack, problem_count=1, seed=3) == local.select_tasks(
        pack, problem_count=1, seed=3
    )
    with pytest.raises(ValueError, match="between 1 and 2"):
        local.select_tasks(pack, problem_count=3)


def test_sampled_run_reports_its_seed(tmp_path, services, sandbox) -> None:
    result = local.run_local_generated_validator(_config(tmp_path, problem_count=1, seed=5))

    assert result.summary["selection_mode"] == "random_sample"
    assert result.summary["selection_seed"] == 5
    assert result.summary["task_count"] == 1


def test_sandbox_failure_is_infrastructure_and_keeps_episodes(tmp_path, services, sandbox) -> None:
    sandbox.returncode = 3

    error = _failure(_config(tmp_path))

    assert error.classification == "infrastructure"
    assert str(error) == "sandbox exited with code 3"
    assert len((error.artifact_dir / "episode_results.jsonl").read_text().splitlines()) == 2
    assert error.report_path is not None


def test_sandbox_timeout_removes_the_container(tmp_path, services, sandbox) -> None:
    sandbox.raise_ = subprocess.TimeoutExpired("docker", 1)

    error = _failure(_config(tmp_path, timeout=1))

    assert error.classification == "infrastructure"
    assert "timed out after 1 seconds" in str(error)


def test_missing_sandbox_output_fails(tmp_path, services, sandbox) -> None:
    sandbox.write_output = False

    error = _failure(_config(tmp_path))

    assert (error.classification, str(error)) == ("infrastructure", "sandbox did not produce output")


def test_quarantined_episodes_beyond_the_threshold_fail_as_environment(
    tmp_path, services, sandbox, monkeypatch
) -> None:
    def broken_step(self, tool, args):
        raise RuntimeError("world crashed")

    monkeypatch.setattr(session_registry.Session, "step", broken_step)

    error = _failure(_config(tmp_path))

    assert error.classification == "environment"
    assert "infrastructure failure: environment_error=2" in str(error)
    summary = json.loads(error.summary_path.read_text())
    assert [t["outcome"] for t in summary["tasks"]] == ["environment_error"] * 2


def test_search_url_defaults_from_the_environment(monkeypatch) -> None:
    monkeypatch.delenv("E2E_SEARCH_URL", raising=False)
    monkeypatch.delenv("SEARCH_SERVER_URL", raising=False)
    assert local._default_search_url() == "http://localhost:5632"
    monkeypatch.setenv("SEARCH_SERVER_URL", "http://search-server:5632")
    assert local._default_search_url() == "http://search-server:5632"
    monkeypatch.setenv("E2E_SEARCH_URL", "http://e2e:1")
    assert local._default_search_url() == "http://e2e:1"


def test_sandbox_mounts_evaluator_artifacts_read_only_and_keeps_tokens_out_of_argv(
    tmp_path, services, sandbox
) -> None:
    local.run_local_generated_validator(_config(tmp_path))

    ((command, kwargs),) = sandbox.commands
    assert "or-test-token" not in " ".join(command)
    assert kwargs["env"]["INFERENCE_ACCESS_TOKEN"] == "or-test-token"
    assert any(arg.endswith(":/app/logs:ro") for arg in command)
