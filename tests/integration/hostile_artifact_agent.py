"""Probe evaluator artifact mounts without inference or environment calls."""

from pathlib import Path


def agent_main(problem):
    """Each evaluator file must be out of reach: no read, write or unlink."""
    report = {}
    for name in (
        "problems.jsonl",
        "environment_sessions.json",
        "summary.json",
        "episode_results.jsonl",
    ):
        target = Path("/run/oro/logs") / name
        for operation in ("read", "write", "unlink"):
            try:
                if operation == "read":
                    target.read_text()
                elif operation == "write":
                    target.write_text("forged")
                else:
                    target.unlink()
            except OSError:
                report[f"{name}:{operation}"] = True
            else:
                report[f"{name}:{operation}"] = False
    probe = Path("/run/oro/output") / f"write-probe-{problem['problem_id']}.txt"
    try:
        probe.write_text("forged output")
    except OSError:
        report["output_unwritable"] = True
    else:
        report["output_unwritable"] = False
    return [{"artifact_isolation": report}]
