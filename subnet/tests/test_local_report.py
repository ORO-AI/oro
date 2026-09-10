from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from subnet import local_report, sandbox

ROOT = Path(__file__).resolve().parents[2]
VIEWER_DIR = ROOT / "trajectory-viewer"
FIXTURES = VIEWER_DIR / "tests" / "fixtures" / "oro-episodes"
EMBED_TAG = '<script id="oro-episodes" type="application/json">'


def _task(task_id: str, family: str, reward: float, outcome: str = "completed") -> dict:
    return {
        "task_id": task_id,
        "family": family,
        "outcome": outcome,
        "correct": outcome == "completed" and reward > 0,
        "reward": reward,
        "error_classification": None if outcome == "completed" else "environment",
        "error_detail": None
        if outcome == "completed"
        else "tool call exceeded 10.000s",
    }


def _summary() -> dict:
    return {
        "run_id": "local-abc123",
        "pack_sha256": "9e5d11c6" + "0" * 52 + "c73a",
        "models": {
            "user_simulator": "mistralai/mistral-small-2603",
            "judge": "deepseek/deepseek-v4-flash-0731",
            "solver": "nvidia/nemotron-3-ultra-550b-a55b",
        },
        "tasks": [
            _task("TF1-intent_decomposition-1", "intent_decomposition", 1.0),
            _task("TF1-intent_decomposition-2", "intent_decomposition", 0.0),
            _task("TF2-retrieval_recall-1", "retrieval_recall", 0.5),
            _task("TF6-recovery-1", "recovery", 0.0, outcome="environment_error"),
        ],
        "aggregate_score": 0.375,
    }


def _report(**overrides: object) -> str:
    kwargs: dict = dict(
        artifact_dir=Path("/app/logs/environment-runs/local-abc123"),
        report_path=Path("/app/logs/environment-runs/local-abc123/trajectories.html"),
        provider="openrouter",
        agent_model="deepseek-ai/DeepSeek-V3.2-TEE",
        color=False,
    )
    kwargs.update(overrides)
    return local_report.render_console_report(_summary(), **kwargs)


def test_console_report_names_run_models_and_versions() -> None:
    report = _report()
    assert "local-abc123" in report
    assert "9e5d11c6" in report
    assert "runtime" in report and "verifier" in report
    assert "openrouter" in report
    assert "deepseek-ai/DeepSeek-V3.2-TEE" in report
    assert "mistralai/mistral-small-2603" in report
    assert "deepseek/deepseek-v4-flash-0731" in report
    # The generator's solver is not a model the miner's run used.
    assert "nemotron" not in report


def test_console_report_groups_tasks_by_family_with_means() -> None:
    report = _report()
    lines = report.splitlines()
    intent = next(
        i for i, line in enumerate(lines) if line.startswith("intent_decomposition")
    )
    assert "0.50" in lines[intent] and "1/2" in lines[intent]
    assert "TF1-intent_decomposition-1" in lines[intent + 1]
    assert "TF1-intent_decomposition-2" in lines[intent + 2]
    # A graded reward below 1.0 on a correct verdict still counts as passed.
    retrieval = next(line for line in lines if line.startswith("retrieval_recall"))
    assert "0.50" in retrieval and "1/1" in retrieval
    recovery = next(line for line in lines if line.startswith("recovery"))
    assert "0.00" in recovery and "0/1" in recovery
    assert "environment: tool call exceeded 10.000s" in report
    assert "Aggregate score  0.375000" in report
    assert (
        report.index("intent_decomposition")
        < report.index("retrieval_recall")
        < report.index("recovery")
    )


def test_console_report_keeps_a_gap_after_a_long_family_name() -> None:
    summary = _summary()
    summary["tasks"] = [_task("t1", "intent_decomposition", 1.0)]
    report = local_report.render_console_report(
        summary,
        artifact_dir=Path("/x"),
        report_path=None,
        provider="openrouter",
        agent_model="m",
        color=False,
    )
    assert "intent_decomposition    mean 1.00  1/1 passed" in report


def test_console_report_shows_host_paths_when_run_under_compose(monkeypatch) -> None:
    monkeypatch.setattr(sandbox, "HOST_PROJECT_DIR", None)
    assert "/app/logs/environment-runs/local-abc123/trajectories.html" in _report()

    monkeypatch.setattr(sandbox, "HOST_PROJECT_DIR", "/home/miner/oro")
    report = _report()
    assert (
        "/home/miner/oro/logs/environment-runs/local-abc123/trajectories.html" in report
    )
    assert "/app/logs" not in report


def test_console_report_colour_is_opt_in() -> None:
    assert "\x1b[" not in _report(color=False)
    coloured = _report(color=True)
    assert "\x1b[32m" in coloured  # a full-reward task
    assert "\x1b[31m" in coloured  # a zero-reward task


def test_console_report_omits_report_line_when_report_was_not_written() -> None:
    assert "trajectories.html" not in _report(report_path=None)


def _fixture_episodes() -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(FIXTURES.glob("*.json"))]


def _embedded_payload(html: str) -> str:
    start = html.index(EMBED_TAG) + len(EMBED_TAG)
    return html[start : html.index("</script>", start)]


def test_bundle_has_no_module_syntax_and_parses() -> None:
    bundle = local_report.bundle_viewer_script(VIEWER_DIR)
    assert not any(
        line.startswith(("import ", "export ")) for line in bundle.splitlines()
    )
    if shutil.which("node") is None:
        pytest.skip("node not available")
    subprocess.run(["node", "--check", "-"], input=bundle, text=True, check=True)


def test_bundle_refuses_module_syntax_it_cannot_strip(tmp_path: Path) -> None:
    shutil.copytree(
        VIEWER_DIR, tmp_path / "viewer", ignore=shutil.ignore_patterns("tests")
    )
    app = tmp_path / "viewer" / "app.js"
    app.write_text('export * from "./loading.js";\n' + app.read_text())
    with pytest.raises(ValueError, match="module syntax survived"):
        local_report.bundle_viewer_script(tmp_path / "viewer")


def test_bundle_strips_a_wrapped_import(tmp_path: Path) -> None:
    shutil.copytree(
        VIEWER_DIR, tmp_path / "viewer", ignore=shutil.ignore_patterns("tests")
    )
    app = tmp_path / "viewer" / "app.js"
    app.write_text(
        'import {\n  validateFilename,\n} from "./loading.js";\n' + app.read_text()
    )
    bundle = local_report.bundle_viewer_script(tmp_path / "viewer")
    assert 'from "./loading.js"' not in bundle
    if shutil.which("node") is not None:
        subprocess.run(["node", "--check", "-"], input=bundle, text=True, check=True)


def test_trajectory_report_embeds_episodes_and_inlines_assets() -> None:
    html = local_report.render_trajectory_report(
        _fixture_episodes(), run_id="local-abc123", viewer_dir=VIEWER_DIR
    )
    assert '<link rel="stylesheet"' not in html
    assert 'type="module"' not in html
    assert "<style>" in html and "<script>" in html
    payload = json.loads(_embedded_payload(html))
    assert payload["run_id"] == "local-abc123"
    assert [entry["name"] for entry in payload["episodes"]] == [
        "TF5-ranking-200000.json",
        "TF6-recovery-200000.json",
    ]
    assert (
        payload["episodes"][0]["value"]["schema_version"]
        == "oro.environment_episode.v1"
    )


def test_trajectory_report_escapes_script_terminators() -> None:
    episodes = _fixture_episodes()
    episodes[0]["episode"]["bootstrap"]["policy_view"]["query"] = "</script><!-- x"
    html = local_report.render_trajectory_report(
        episodes, run_id="local-abc123", viewer_dir=VIEWER_DIR
    )
    payload = _embedded_payload(html)
    assert "<" not in payload
    query = json.loads(payload)["episodes"][0]["value"]["episode"]["bootstrap"]
    assert query["policy_view"]["query"] == "</script><!-- x"


def test_write_trajectory_report_returns_the_written_path(tmp_path: Path) -> None:
    out = local_report.write_trajectory_report(
        tmp_path / "trajectories.html",
        _fixture_episodes(),
        run_id="local-abc123",
        viewer_dir=VIEWER_DIR,
    )
    assert out == tmp_path / "trajectories.html"
    assert "TF6-recovery-200000" in out.read_text()


def test_write_trajectory_report_missing_viewer_is_not_fatal(
    tmp_path: Path, capsys
) -> None:
    out = local_report.write_trajectory_report(
        tmp_path / "trajectories.html",
        _fixture_episodes(),
        run_id="local-abc123",
        viewer_dir=tmp_path / "nope",
    )
    assert out is None
    assert "trajectory viewer report skipped" in capsys.readouterr().err
    assert not (tmp_path / "trajectories.html").exists()


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_viewer_node_suite_passes() -> None:
    subprocess.run(
        [
            "node",
            "--test",
            *sorted(str(p) for p in (VIEWER_DIR / "tests").glob("*.test.mjs")),
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
    )
