"""Probe evaluator artifact mounts without inference or environment calls."""

from pathlib import Path


def agent_main(problem):
    report = {}
    for name in (
        "problems.jsonl",
        "environment_sessions.json",
        "summary.json",
        "episode_results.jsonl",
    ):
        target = Path("/app/logs") / name
        for operation in ("write", "unlink"):
            try:
                if operation == "write":
                    target.write_text("forged")
                else:
                    target.unlink()
            except OSError:
                report[f"{name}:{operation}"] = True
            else:
                report[f"{name}:{operation}"] = False
    probe = Path("/app/output") / f"write-probe-{problem['problem_id']}.txt"
    probe.write_text("sandbox output is writable")
    report["output_writable"] = probe.read_text() == "sandbox output is writable"
    return [{"artifact_isolation": report}]
