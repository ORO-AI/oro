"""Console and HTML reports for a local generated-environment run.

The console report replaces the raw per-task lines the CLI used to print. The
HTML report bundles the vendored Trajectory Room viewer with every episode of
the run into one file, so a miner can open it from any browser, including after
copying it off a VPS.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from subnet.sandbox import host_path
from subnet.validator.env_pack_loader import PACK_VERSION_IDENTITIES

VIEWER_DIR = Path(__file__).resolve().parents[1] / "trajectory-viewer"

# The viewer's ES modules, in dependency order. Only function declarations are
# exported, so once import/export syntax is stripped the concatenation is a
# valid classic script; app.js must come last because it runs on load.
_VIEWER_MODULES = (
    "adapters/atif.js",
    "adapters/oro.js",
    "adapters/index.js",
    "loading.js",
    "render.js",
    "app.js",
)
_IMPORT_STATEMENT_RE = re.compile(r"^import\s[^;]*;[ \t]*\n?", re.MULTILINE)
_EXPORT_PREFIX_RE = re.compile(r"^export ", re.MULTILINE)
_MODULE_SYNTAX_RE = re.compile(
    r"^(?:import|export|default)\b|\bfrom\s+\"\.{1,2}/", re.MULTILINE
)
_STYLESHEET_TAG = '<link rel="stylesheet" href="./styles.css">'
_MODULE_TAG = '<script type="module" src="./app.js"></script>'
_EMBED_TAG = '<script id="oro-episodes" type="application/json">'

_RESET = "\x1b[0m"
_BOLD = "\x1b[1m"
_DIM = "\x1b[2m"
_RED = "\x1b[31m"
_GREEN = "\x1b[32m"
_YELLOW = "\x1b[33m"
_CYAN = "\x1b[36m"


def stdout_supports_color() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def render_console_report(
    summary: dict[str, Any],
    *,
    artifact_dir: Path,
    report_path: Path | None,
    provider: str,
    agent_model: str,
    color: bool,
) -> str:
    def paint(text: str, *codes: str) -> str:
        return f"{''.join(codes)}{text}{_RESET}" if color else text

    def reward_paint(reward: float, outcome: str) -> str:
        text = f"{reward:.2f}"
        if outcome != "completed" or reward <= 0:
            return paint(text, _RED)
        if reward >= 1:
            return paint(text, _GREEN)
        return paint(text, _YELLOW)

    tasks: list[dict[str, Any]] = summary["tasks"]
    models: dict[str, str] = summary.get("models") or {}
    pack_sha256 = str(summary["pack_sha256"])
    lines = [
        f"{paint('ORO Bench local run', _BOLD)}  {paint(str(summary['run_id']), _CYAN)}",
        f"  pack        {pack_sha256[:8]}…{pack_sha256[-4:]}  ({len(tasks)} tasks)",
        (
            f"  runtime     {PACK_VERSION_IDENTITIES['runtime_version']}"
            f"  verifier {PACK_VERSION_IDENTITIES['verifier_version']}"
        ),
        f"  inference   {provider}",
        (
            f"  agent model {agent_model}  "
            f"{paint('(reference agent; custom agents choose in code)', _DIM)}"
        ),
        f"  simulator   {models.get('user_simulator', 'not declared')}",
        f"  judge       {models.get('judge', 'not declared')}",
        "",
    ]

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_family[str(task.get("family") or "unknown")].append(task)
    width = max((len(str(task.get("task_id"))) for task in tasks), default=0)
    label_width = max([width, *(len(family) for family in by_family)]) + 4
    for family, rows in by_family.items():
        rewards = [float(row.get("reward") or 0) for row in rows]
        passed = sum(1 for row in rows if row.get("correct"))
        mean = sum(rewards) / len(rewards)
        lines.append(
            f"{paint(f'{family:<{label_width}}', _BOLD)}"
            f"mean {mean:.2f}  {passed}/{len(rows)} passed"
        )
        for row, reward in zip(rows, rewards):
            outcome = str(row.get("outcome") or "")
            detail = ""
            if outcome != "completed":
                classification = row.get("error_classification") or "error"
                problem = f"{classification}: {row.get('error_detail') or outcome}"
                detail = f"  {paint(problem, _RED)}"
            lines.append(
                f"  {str(row.get('task_id')):<{width}}  {outcome:<18}"
                f"{reward_paint(reward, outcome)}{detail}"
            )
        lines.append("")

    aggregate = f"{float(summary['aggregate_score']):.6f}"
    lines.append(f"{paint('Aggregate score', _BOLD)}  {paint(aggregate, _BOLD, _CYAN)}")
    lines.append(f"Artifacts        {host_path(str(artifact_dir))}")
    if report_path is not None:
        lines.append(
            f"Trajectories     {host_path(str(report_path))}  "
            f"{paint('(open in a browser)', _DIM)}"
        )
    return "\n".join(lines)


def bundle_viewer_script(viewer_dir: Path = VIEWER_DIR) -> str:
    parts = []
    for name in _VIEWER_MODULES:
        source = (viewer_dir / name).read_text(encoding="utf-8")
        source = _EXPORT_PREFIX_RE.sub("", _IMPORT_STATEMENT_RE.sub("", source))
        if _MODULE_SYNTAX_RE.search(source):
            raise ValueError(f"{viewer_dir / name}: module syntax survived bundling")
        parts.append(f"// ---- {name}\n{source}")
    body = "\n".join(parts)
    return f'(() => {{\n"use strict";\n{body}\n}})();\n'


def _embed_json(value: Any) -> str:
    # "\u003c" is plain JSON for "<", so JSON.parse still round-trips it, and the
    # block can no longer contain "</script>" or "<!--", the two sequences that
    # end or confuse an inline <script>.
    return json.dumps(value, separators=(",", ":")).replace("<", "\\u003c")


def render_trajectory_report(
    episodes: list[dict[str, Any]],
    *,
    run_id: str,
    viewer_dir: Path = VIEWER_DIR,
) -> str:
    template = (viewer_dir / "index.html").read_text(encoding="utf-8")
    if _STYLESHEET_TAG not in template or _MODULE_TAG not in template:
        raise ValueError(f"{viewer_dir / 'index.html'}: asset tags to inline not found")
    styles = (viewer_dir / "styles.css").read_text(encoding="utf-8")
    entries = [
        {"name": f"{episode['episode']['task_id']}.json", "value": episode}
        for episode in episodes
    ]
    payload = _embed_json({"run_id": run_id, "episodes": entries})
    scripts = (
        f"{_EMBED_TAG}{payload}</script>\n"
        f"    <script>\n{bundle_viewer_script(viewer_dir)}    </script>"
    )
    return template.replace(_STYLESHEET_TAG, f"<style>\n{styles}</style>").replace(
        _MODULE_TAG, scripts
    )


def write_trajectory_report(
    path: Path,
    episodes: list[dict[str, Any]],
    *,
    run_id: str,
    viewer_dir: Path = VIEWER_DIR,
) -> Path | None:
    """Write the single-file viewer, or warn and return None if that is impossible."""

    try:
        html = render_trajectory_report(episodes, run_id=run_id, viewer_dir=viewer_dir)
        path.write_text(html, encoding="utf-8")
    except (OSError, ValueError, KeyError) as error:
        print(f"Warning: trajectory viewer report skipped ({error!r})", file=sys.stderr)
        return None
    return path


__all__ = [
    "render_console_report",
    "stdout_supports_color",
    "write_trajectory_report",
]
