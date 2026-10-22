from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from oro_sdk.types import UNSET

from subnet.tests.world_pack_fixture import (
    StubModelClient,
    StubSearchClient,
    archive_pack,
    build_world_pack,
    load_world_pack,
    world_keyring,  # noqa: F401
)
from validator import env_pack_loader, session_registry
from validator import main as validator_main
from validator.generated_evaluation import (
    GENERATED_PROBLEM_SCHEMA,
    aggregate_results,
    claim_roster,
    summarize_agent_inference_usage,
    summarize_episode_resource_usage,
    write_problem_file,
)
from validator.main import Validator

REAL_FETCH_PACK = env_pack_loader.fetch_and_validate_pack
REAL_FETCH_RACE_PACK = env_pack_loader.fetch_and_validate_race_pack

NONCES = {"t1": "nonce-1", "t2": "nonce-2"}


def _record(task_id: str, score: float = 0.0, *, flags=(), verdict=True, session_id=None):
    return {
        "task_id": task_id,
        "session_id": session_id or f"s-{task_id}",
        "episode_nonce": f"n-{task_id}",
        "ledger": {"episode": [{"kind": "open"}, {"reason": "order"}]},
        "verdict": (
            {"score": score, "category": None, "flags": list(flags)} if verdict else None
        ),
        "error": None if verdict else "runtime step failed: RuntimeError",
        "call_trace": [],
        "steps": 1,
    }


def test_problem_file_contains_only_public_session_contract(tmp_path) -> None:
    session = {
        "session_id": "session",
        "binding": {"session_id": "session"},
        "policy_view": {"query": "find a mug", "tools": [], "max_steps": 6},
        "private": "never exposed",
    }
    path = tmp_path / "problems.jsonl"

    write_problem_file(path, [session])

    assert json.loads(path.read_text()) == {
        "problem_id": "session",
        "query": "find a mug",
        "category": "generated_environment",
        "environment": {
            "schema_version": GENERATED_PROBLEM_SCHEMA,
            "binding": {"session_id": "session"},
            "policy_view": session["policy_view"],
        },
    }


def test_claim_roster_binds_delivery_tasks_to_claim_nonces(tmp_path) -> None:
    pack = load_world_pack(tmp_path, delivery_id="qual-1", delivery_kind="qualifying")

    assert claim_roster(pack, "qual-1", dict(reversed(NONCES.items()))) == NONCES


@pytest.mark.parametrize(
    ("delivery_id", "nonces", "match"),
    [
        (None, NONCES, "legacy work"),
        ("qual-1", None, "legacy work"),
        ("qual-1", {}, "legacy work"),
        ("other", NONCES, "claimed delivery_id"),
        ("qual-1", {"t1": "nonce-1"}, "differ from the delivery tasks"),
        ("qual-1", {**NONCES, "t3": "x"}, "differ from the delivery tasks"),
        ("qual-1", {"t1": "nonce-1", "t2": ""}, "nonempty strings"),
    ],
)
def test_claim_roster_fails_closed(tmp_path, delivery_id, nonces, match) -> None:
    pack = load_world_pack(tmp_path, delivery_id="qual-1", delivery_kind="qualifying")

    with pytest.raises(ValueError, match=match):
        claim_roster(pack, delivery_id, nonces)


def test_score_is_mean_verdict_score_over_the_roster() -> None:
    records = [_record("a", 1.0), _record("b", 0.25)]

    # A roster task without a graded episode scores zero.
    assert aggregate_results(records, ["a", "b", "c", "d"]) == pytest.approx(0.3125)


def test_quarantines_and_world_errors_score_zero_up_to_the_threshold() -> None:
    roster = [str(i) for i in range(10)]
    records = [_record("0", verdict=False), _record("1", flags=["world_error"]),
               *(_record(t, 1.0) for t in roster[2:])]

    assert aggregate_results(records, roster) == pytest.approx(0.8)
    records[2] = _record("2", verdict=False)
    records[3] = _record("3", flags=["world_error"])
    with pytest.raises(ValueError) as failure:
        aggregate_results(records, roster)
    assert str(failure.value) == (
        "generated evaluation infrastructure failure: environment_error=2, verifier_error=2"
    )


def test_bound_trips_score_zero_and_never_fail_the_run() -> None:
    roster = ["a", "b", "c"]
    records = [_record(t, flags=["possible_world_defect"]) for t in roster]

    assert aggregate_results(records, roster) == 0.0


def test_any_environment_unscored_episode_fails_the_run_as_infrastructure() -> None:
    roster = [str(i) for i in range(10)]
    records = [{**_record("0", verdict=False), "unscored": "environment"}, *(_record(t, 1.0) for t in roster[1:])]

    with pytest.raises(ValueError) as failure:
        aggregate_results(records, roster)
    assert str(failure.value) == "generated evaluation infrastructure failure: environment_error=1"


@pytest.mark.parametrize(
    "records",
    [[_record("a"), _record("a")], [_record("z")]],
)
def test_results_must_match_the_roster(records) -> None:
    with pytest.raises(ValueError, match="differ from the delivery roster"):
        aggregate_results(records, ["a", "b"])


def test_empty_roster_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty roster"):
        aggregate_results([], [])


@pytest.mark.parametrize(
    ("provider", "stats", "expected_status"),
    [
        ("openrouter", {"inference_cost_usd": 0.25, "cached_tokens": 8}, "complete"),
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
    assert usage["cached_tokens"] == (stats or {}).get("cached_tokens", 0)
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
                "cached_tokens": 8,
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
    assert summary["cached_tokens"] == 8
    assert summary["requested_models"] == {
        "requested/a": {"requests": 2},
        "requested/b": {"requests": 1, "failed_requests": 1},
    }
    assert summary["served_models"] == {
        "served/a": {"requests": 2, "cost_usd": 0.3},
    }
    assert summary["inference_cost_status"] == "complete"
    assert summary["inference_cost_usd"] == 0.3


def test_episode_resource_usage_counts_tool_steps() -> None:
    result = {
        "session_id": "session",
        "task_id": "task",
        "call_trace": [
            {"tool": "search", "step": 1, "elapsed_ms": 1.0},
            {"tool": "view", "step": 2, "elapsed_ms": 1.0},
            {"tool": "search", "step": 3, "elapsed_ms": 1.0, "error": "x"},
        ],
    }

    usage = summarize_episode_resource_usage([result], {}, "openrouter")["task"]

    assert usage["operation_requests"] == 3
    assert usage["operations"] == {"environment.search": 2, "environment.view": 1}


@pytest.mark.parametrize("call_trace", [None, {}, [None, {}, {"tool": None}, {"tool": ""}]])
def test_episode_resource_usage_ignores_malformed_trace_entries(call_trace) -> None:
    usage = summarize_episode_resource_usage(
        [{"session_id": "session", "task_id": "task", "call_trace": call_trace}],
        {},
        "openrouter",
    )["task"]

    assert usage["operation_requests"] == 0
    assert usage["operations"] == {}


def _validator(tmp_path: Path) -> Validator:
    validator = Validator.__new__(Validator)
    validator.config = SimpleNamespace(
        backend_url="http://backend.test",
        search_server_url="http://search.test",
        session_model_timeout=5.0,
    )
    validator.wallet = SimpleNamespace(hotkey=object())
    validator._eval_dir = MagicMock(return_value=tmp_path)
    validator.session_runtime = MagicMock()
    validator.backend_client = MagicMock()
    validator.backend_client.get_run_problems.return_value = [
        {"task_id": "t1", "family": "world"}, {"task_id": "t2", "family": "world"},
    ]
    return validator


def _work(**fields):
    return SimpleNamespace(**{
        "eval_run_id": "run",
        "agent_version_id": "agent",
        "env_pack_sha256": "a" * 64,
        "race_id": None,
        "delivery_id": "qual-1",
        "episode_nonces": SimpleNamespace(additional_properties=dict(NONCES)),
        **fields,
    })


@pytest.fixture
def delivery_fetch(tmp_path, monkeypatch, world_keyring):  # noqa: F811
    """Serve the signed test world as the claimed delivery and stub outside services."""

    archive = archive_pack(
        build_world_pack(tmp_path / "world", delivery_id="qual-1", delivery_kind="qualifying"),
        tmp_path / "delivery.tar.gz",
    )
    calls = []

    def fetch(kind):
        async def fetch_delivery(*args, expected_delivery_id, **kwargs):
            calls.append((kind, args[0], expected_delivery_id))
            return env_pack_loader.load_delivery(
                archive, expected_delivery_id=expected_delivery_id, cache_root=tmp_path / "cache"
            )
        return fetch_delivery

    monkeypatch.setattr(env_pack_loader, "fetch_and_validate_pack", fetch("qualifying"))
    monkeypatch.setattr(env_pack_loader, "fetch_and_validate_race_pack", fetch("race"))
    search = StubSearchClient()
    models = StubModelClient()
    monkeypatch.setattr(session_registry, "SearchServerClient", lambda url, *a: search)
    monkeypatch.setattr(session_registry, "ProxyModelClient", lambda token, **kw: models)
    return SimpleNamespace(calls=calls, search=search, models=models)


def test_claimed_delivery_opens_one_session_per_task_with_claim_nonces(
    tmp_path, delivery_fetch
) -> None:
    validator = _validator(tmp_path)

    registry, sessions, roster, delivery_id = validator._create_environment_sessions(
        _work(), inference_access_token="token"
    )
    try:
        assert delivery_fetch.calls == [("qualifying", "a" * 64, "qual-1")]
        assert delivery_id == "qual-1" and roster == NONCES
        # Every index was uploaded to the search server before sessions opened.
        assert list(delivery_fetch.search.titles) == [
            index["id"] for index in registry.pack.indexes.values()
        ]
        assert [s["policy_view"]["query"] for s in sessions] == [
            "Buy the item for t1.", "Buy the item for t2."
        ]
        saved = json.loads((tmp_path / "environment_sessions.json").read_text())
        assert saved["sessions"] == sessions
        validator.session_runtime.install.assert_called_once_with(registry)
        binding = sessions[0]["binding"]
        assert registry.call({**binding, "step": 1, "action": {"name": "buy", "args": {"id": "a"}}})[
            "finished"
        ]
        registry.finish_all()
        records = {r["task_id"]: r for r in registry.terminal_results()}
        assert records["t1"]["episode_nonce"] == "nonce-1"
        assert records["t1"]["verdict"]["score"] == 1.0
        assert records["t2"]["verdict"]["category"] == "did not finish"
    finally:
        registry.close()


def test_race_claim_loads_the_race_delivery(tmp_path, delivery_fetch) -> None:
    validator = _validator(tmp_path)

    registry, *_ = validator._create_environment_sessions(
        _work(race_id="00000000-0000-0000-0000-000000000001"),
        inference_access_token="token",
    )
    registry.close()

    assert delivery_fetch.calls == [("race", "00000000-0000-0000-0000-000000000001", "qual-1")]


@pytest.mark.parametrize("race_id", [None, "00000000-0000-0000-0000-000000000001"])
def test_claims_reach_the_real_fetch_signatures(tmp_path, delivery_fetch, monkeypatch, race_id) -> None:
    """The public fetch functions run as written; only the backend download is stubbed."""

    monkeypatch.setattr(env_pack_loader, "fetch_and_validate_pack", REAL_FETCH_PACK)
    monkeypatch.setattr(env_pack_loader, "fetch_and_validate_race_pack", REAL_FETCH_RACE_PACK)
    archive = archive_pack(
        build_world_pack(tmp_path / "real-world", delivery_id="qual-1", delivery_kind="qualifying"),
        tmp_path / "real.tar.gz",
    )
    seen = []

    async def fetch_delivery(label, binding, fetch, backend_url, keypair, *, expected_delivery_id, **kwargs):
        seen.append((label, binding, backend_url))
        return env_pack_loader.load_delivery(archive, expected_delivery_id=expected_delivery_id, cache_root=tmp_path / "c")

    monkeypatch.setattr(env_pack_loader, "_fetch_delivery", fetch_delivery)
    validator = _validator(tmp_path)

    registry, *_ = validator._create_environment_sessions(_work(race_id=race_id), inference_access_token="token")
    registry.close()

    [(label, binding, backend_url)] = seen
    assert binding["pack_sha256"] == "a" * 64 and backend_url == validator.config.backend_url
    assert binding.get("race_id") == race_id


@pytest.mark.parametrize(
    "unbound",
    [{"delivery_id": UNSET, "episode_nonces": UNSET}, {"delivery_id": None}, {"episode_nonces": None}],
)
def test_unbound_claim_fails_clearly(tmp_path, delivery_fetch, unbound) -> None:
    work = _work(**unbound)

    with pytest.raises(ValueError, match="not bound to a world delivery"):
        _validator(tmp_path)._create_environment_sessions(work, inference_access_token="token")


def test_claim_without_pack_binding_fails_clearly(tmp_path) -> None:
    work = _work()
    work.env_pack_sha256 = None

    with pytest.raises(ValueError, match="legacy work"):
        _validator(tmp_path)._create_environment_sessions(work, inference_access_token="token")


def test_rejected_delivery_fails_the_run(tmp_path, monkeypatch) -> None:
    async def rejected(*args, **kwargs):
        return None

    monkeypatch.setattr(env_pack_loader, "fetch_and_validate_pack", rejected)

    with pytest.raises(RuntimeError, match="delivery qual-1 failed validation"):
        _validator(tmp_path)._create_environment_sessions(_work(), inference_access_token="token")


def _runner(tmp_path, monkeypatch, records, *, roster=("t1", "t2"), sandbox_output=True):
    registry = MagicMock()
    registry.key_exhausted = threading.Event()
    registry.terminal_results.return_value = records
    reporter = MagicMock()
    reporter_type = MagicMock(return_value=reporter)
    monkeypatch.setattr(validator_main, "GeneratedProgressReporter", reporter_type)
    validator = _validator(tmp_path)
    validator._create_environment_sessions = MagicMock(return_value=(
        registry,
        [{"session_id": "s-t1", "binding": {"session_id": "s-t1"}, "policy_view": {"query": "q"}}],
        {task: f"n-{task}" for task in roster},
        "qual-1",
    ))
    validator.run_sandbox = MagicMock(
        return_value=(tmp_path / "output.jsonl" if sandbox_output else None, {})
    )
    return validator, registry, reporter, reporter_type


def _run(validator, tmp_path, token="synthetic"):
    return validator._run_generated_evaluation(
        _work(), tmp_path / "agent.py", inference_access_token=token,
        inference_provider="openrouter", inference_base_url="https://example.test/v1",
    )


def test_generated_runner_scores_and_flushes_every_record(tmp_path, monkeypatch) -> None:
    records = [_record("t1", 1.0), _record("t2", 0.5)]
    validator, registry, reporter, _ = _runner(tmp_path, monkeypatch, records)

    completion = _run(validator, tmp_path)

    assert completion.score == pytest.approx(0.75)
    assert completion.score_components == {"schema_version": "oro.generated_run_score.v1"}
    registry.finish_all.assert_called_once_with()
    registry.terminal_results.assert_called_once_with(final=True)
    reporter.flush.assert_called_once_with(records)
    validator.session_runtime.clear.assert_called_once_with(registry)
    validator.backend_client.complete_run.assert_not_called()
    assert validator.run_sandbox.call_args.kwargs["stop_event"] is registry.key_exhausted


def test_environment_fault_reruns_see_nothing_from_the_first_attempt(tmp_path, monkeypatch) -> None:
    records = [_record("t1", 1.0), _record("t2", 0.5)]
    validator, registry, _reporter, _ = _runner(tmp_path, monkeypatch, records)

    def eval_dir(run_id: str) -> Path:
        (tmp_path / f"eval_{run_id}").mkdir(exist_ok=True)
        return tmp_path / f"eval_{run_id}"

    validator._eval_dir = eval_dir
    (tmp_path / "agent.py").write_text("AGENT = 1\n")
    rerun_session = {"session_id": "s-t2-rerun", "binding": {"session_id": "s-t2-rerun"}, "policy_view": {"query": "q"}}
    seen = []

    def sandbox(agent_path, eval_run_id, problem_file, **kwargs):
        run_dir = problem_file.parent
        problems = [json.loads(row)["problem_id"] for row in problem_file.read_text().splitlines()]
        seen.append((run_dir, sorted(p.name for p in run_dir.iterdir()), agent_path.read_text(), problems))
        (run_dir / "left-by-the-agent.txt").write_text("attempt state")
        (run_dir / "output.jsonl").write_text(json.dumps({"problem_id": f"run-{len(seen)}"}) + "\n")
        return run_dir / "output.jsonl", {}

    validator.run_sandbox.side_effect = sandbox
    registry.rerun_environment_faults.side_effect = lambda run: run([rerun_session])

    assert _run(validator, tmp_path).score == pytest.approx(0.75)

    (dir1, *_), (dir2, files2, agent2, problems2) = seen
    assert dir2 != dir1 and files2 == ["agent.py", "problems.jsonl"]  # nothing from attempt 1
    assert agent2 == "AGENT = 1\n" and problems2 == ["s-t2-rerun"]
    assert not dir2.exists()  # removed once its output is folded in
    output = (dir1 / "output.jsonl").read_text().splitlines()
    assert [json.loads(row)["problem_id"] for row in output] == ["run-1", "run-2"]


def test_no_rerun_after_the_miner_key_runs_out(tmp_path, monkeypatch) -> None:
    validator, registry, _reporter, _ = _runner(tmp_path, monkeypatch, [_record("t1", 1.0), _record("t2", 1.0)])
    registry.key_exhausted.set()

    _run(validator, tmp_path)

    registry.rerun_environment_faults.assert_not_called()


def test_generated_runner_fails_on_world_errors(tmp_path, monkeypatch) -> None:
    records = [_record("t1", flags=["world_error"]), _record("t2", 1.0)]
    validator, registry, reporter, _ = _runner(tmp_path, monkeypatch, records)

    assert _run(validator, tmp_path) is None

    failure = validator.backend_client.complete_run.call_args.kwargs
    assert failure["status"] == validator_main.TerminalStatus.FAILED
    assert failure["failure_reason"] == (
        "generated evaluation infrastructure failure: verifier_error=1"
    )
    validator.session_runtime.clear.assert_called_once_with(registry)


def test_emission_failure_does_not_change_the_score(tmp_path, monkeypatch) -> None:
    validator, _, reporter, _ = _runner(tmp_path, monkeypatch, [_record("t1", 1.0), _record("t2")])
    reporter.flush.side_effect = RuntimeError("artifact upload failed")

    assert _run(validator, tmp_path).score == pytest.approx(0.5)
    validator.backend_client.complete_run.assert_not_called()


def test_sandbox_failure_fails_the_run(tmp_path, monkeypatch) -> None:
    validator, *_ = _runner(tmp_path, monkeypatch, [], sandbox_output=False)

    assert _run(validator, tmp_path) is None
    assert validator.backend_client.complete_run.call_args.kwargs["failure_reason"] == (
        "Sandbox execution failed"
    )


def test_key_exhaustion_scores_finished_episodes(tmp_path, monkeypatch) -> None:
    validator, registry, *_ = _runner(
        tmp_path, monkeypatch, [_record("t1", 1.0)], roster=("t1", "t2", "t3", "t4"),
        sandbox_output=False,
    )
    registry.key_exhausted.set()

    completion = _run(validator, tmp_path)

    assert completion.score == pytest.approx(0.25)
    assert completion.sandbox_metadata["_miner_inference_key_exhausted"] is True


def test_progress_batches_emit_with_the_claim_binding(tmp_path, monkeypatch) -> None:
    records = [_record("t1", 1.0), _record("t2")]
    validator, _, _, reporter_type = _runner(tmp_path, monkeypatch, records)
    emitted = []

    async def emit(**kwargs):
        emitted.append(kwargs)
        return {"results": [], "counts": {"201": len(kwargs["results"])}}

    monkeypatch.setattr(validator_main, "emit_finalized_results", emit)
    _run(validator, tmp_path)

    reporter_type.call_args.args[1](records)

    (call,) = emitted
    assert call["delivery_id"] == "qual-1"
    assert call["eval_run_id"] == "run"
    assert call["env_pack_sha256"] == "a" * 64
    assert call["task_families"] == {"t1": "world", "t2": "world"}
    assert call["results"] == records


def test_claim_episode_nonces_read_the_sdk_model() -> None:
    assert validator_main._claim_episode_nonces(_work()) == NONCES
    assert validator_main._claim_episode_nonces(_work(episode_nonces=UNSET)) is None


@pytest.mark.parametrize(
    ("sidecar_stats", "output_stats", "expected_cost", "expected_requests"),
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
            2,
        ),
        ({"problem_id": "session", "inference_total": "invalid"}, None, None, 0),
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
            3,
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
            3,
        ),
        (
            {
                "problem_id": "session",
                "inference_total": 4,
                "inference_cost_usd": 1.5,
            },
            {
                "inference_total": "invalid",
                "inference_cost_usd": "x",
                "cached_tokens": 5,
            },
            1.5,
            4,
        ),
        (
            {
                "problem_id": "session",
                "inference_total": 4,
                "inference_cost_usd": 1.5,
            },
            {"inference_total": "invalid", "prompt_tokens": 7},
            1.5,
            4,
        ),
    ],
)
def test_generated_runner_retains_agent_inference_summary(
    tmp_path, monkeypatch, sidecar_stats, output_stats, expected_cost, expected_requests
):
    record = {**_record("t1", 0.5), "session_id": "session"}
    validator, _, _, reporter_type = _runner(tmp_path, monkeypatch, [record], roster=("t1",))
    validator._emit_episode_records = MagicMock()
    if sidecar_stats is not None:
        (tmp_path / "inference_stats.jsonl").write_text(json.dumps(sidecar_stats) + "\n")
    if output_stats is not None:
        (tmp_path / "output.jsonl").write_text(
            json.dumps(
                {
                    "problem_id": "session",
                    "_shadow_inference_usage": output_stats,
                    "dialogue": [],
                    "miner_echo": "sk-or-test-miner",
                }
            )
            + "\n"
        )

    completion = _run(validator, tmp_path, token="sk-or-test-miner")

    reporter_type.call_args.args[1]([record])
    transcripts = validator._emit_episode_records.call_args.args[4]
    assert "sk-or-test-miner" not in json.dumps(transcripts)
    assert completion.score == 0.5
    usage = completion.sandbox_metadata["_shadow_resource_usage"]["by_episode"]["t1"]
    if expected_cost is None:
        assert usage["inference_cost_status"] == "missing"
        assert "inference_cost_usd" not in usage
    else:
        assert usage["inference_cost_status"] == "complete"
        assert usage["inference_cost_usd"] == expected_cost
    assert usage["inference_requests"] == expected_requests
