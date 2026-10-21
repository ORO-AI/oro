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


def _task(task_id: str, score: float, category: str | None = None, flags=()) -> dict:
    return {
        "task_id": task_id,
        "score": score,
        "category": category,
        "flags": list(flags),
        "outcome": "completed",
        "error_classification": None,
        "error_detail": None,
    }


def _summary() -> dict:
    return {
        "run_id": "local-abc123",
        "pack_sha256": "9e5d11c6" + "0" * 52 + "c73a",
        "world_id": "w-fd064ac905a6",
        "tasks": [
            _task("t-1", 1.0),
            _task("t-2", 0.0, "request not met"),
            _task("t-3", 0.44, "not the best option"),
            _task("t-4", 0.0, "did not finish", ["possible_world_defect"]),
        ],
        "aggregate_score": 0.36,
        "pack_task_count": 35,
        "agent_file": "my_agent.py",
        "agent_sha256": "c" * 64,
        "selection_mode": "random_sample",
        "selection_seed": 4821993,
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


def test_console_report_names_run_world_and_versions() -> None:
    report = _report()
    assert "local-abc123" in report
    assert "9e5d11c6" in report
    assert "world       w-fd064ac905a6" in report
    assert "runtime" in report
    assert "openrouter" in report
    assert "deepseek-ai/DeepSeek-V3.2-TEE" in report


def _render(summary: dict) -> str:
    return local_report.render_console_report(
        summary,
        artifact_dir=Path("/x"),
        report_path=None,
        provider="openrouter",
        agent_model="m",
        color=False,
    )


def test_console_report_names_the_sample_and_how_to_repeat_it() -> None:
    assert "problems    4 of 35, sampled, repeat with --seed 4821993" in _report()

    full = _summary()
    full["pack_task_count"] = len(full["tasks"])
    full["selection_mode"] = "all_tasks"
    full["selection_seed"] = None
    report = _render(full)
    assert "problems    all 4" in report
    assert "--seed" not in report


def test_console_report_qualifies_the_agent_model_it_prints() -> None:
    report = _report()
    # SANDBOX_MODEL is a request, not a record: the proxy maps it per provider and
    # a custom agent may never read it.
    assert "SANDBOX_MODEL" in report
    assert "mapped per provider" in report
    assert "custom agents choose in code" in report


def test_console_report_names_the_agent_that_ran() -> None:
    report = _report()
    assert "agent       my_agent.py  sha256 cccccccccccc…" in report


def test_console_report_groups_tasks_by_category() -> None:
    lines = _report().splitlines()
    header = {line.split("  ")[0]: i for i, line in enumerate(lines) if "task(s)" in line}
    assert list(header) == ["correct", "did not finish", "not the best option", "request not met"]
    assert lines[header["correct"] + 1].split() == ["t-1", "1.00"]
    assert lines[header["not the best option"] + 1].split() == ["t-3", "0.44"]
    assert lines[header["did not finish"] + 1].endswith("0.00  possible_world_defect")
    assert "Aggregate score  0.360000" in _report()


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
    assert "\x1b[32m" in coloured  # a full score
    assert "\x1b[33m" in coloured  # partial credit
    assert "\x1b[31m" in coloured  # a zero score


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
