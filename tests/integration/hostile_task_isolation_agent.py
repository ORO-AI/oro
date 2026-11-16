"""Probe whether one task of a run can see or reach the run's other tasks.

Every task writes a canary into its own TMPDIR, waits for the others to start, then
tries each cross-task channel. The run uses network "none", and its one inference
call fails fast but is still recorded.
"""

import os
import tempfile
import time
from pathlib import Path


def _blocked(action) -> bool:
    try:
        action()
    except (OSError, ValueError):
        return True
    return False


def _task_uid(pid: str) -> int:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("Uid:"):
            return int(line.split()[1])
    return -1


def _sibling_pids() -> list[int]:
    """Other tasks' processes: /proc/<pid>/status stays world-readable."""
    pids = []
    for entry in Path("/proc").iterdir():
        try:
            if entry.name.isdigit() and int(entry.name) != os.getpid() and _task_uid(entry.name) >= 20000:
                pids.append(int(entry.name))
        except OSError:
            pass
    return pids


def _plant_request_log_link(own: Path) -> bool:
    """Point our own request log at the run's sessions; the executor must not follow it."""
    log = own / "request_log.jsonl"
    log.unlink(missing_ok=True)
    log.symlink_to("/run/oro/logs/environment_sessions.json")
    return True


def agent_main(problem):
    own = Path(tempfile.gettempdir())
    (own / "canary.txt").write_text(problem["problem_id"])
    deadline = time.time() + 10
    while time.time() < deadline and len(_sibling_pids()) < 2:
        time.sleep(0.2)
    siblings = _sibling_pids()
    try:
        from src.agent.proxy_client import ProxyClient  # real agents import it
    except ImportError:
        ProxyClient = None
    if ProxyClient is not None:  # no network: fails fast, but is still logged
        ProxyClient(timeout=2, max_retries=0).post(
            "/inference/chat/completions", {"model": "probe", "messages": []}
        )
    report = {
        "runtime_importable": ProxyClient is not None,
        "own_uid": os.getuid() >= 20000 and os.getuid() == os.getgid(),
        "private_tmp_writable": (own / "canary.txt").read_text() == problem["problem_id"],
        "saw_siblings": len(siblings) >= 2,
        "shared_tmp_unlistable": _blocked(lambda: list(Path("/tmp").iterdir())),
        "shared_tmp_unwritable": _blocked(lambda: Path("/tmp/shared.txt").write_text("x")),
        "dev_shm_unwritable": _blocked(lambda: Path("/dev/shm/shared.txt").write_text("x")),
        "other_task_dirs_unreadable": all(
            _blocked(lambda p=p: Path(f"/tmp/oro-task-{_task_uid(str(p))}/canary.txt").read_text())
            for p in siblings
        ),
        "request_log_link_planted": problem["problem_id"] != "task-0"
        or _plant_request_log_link(own),
        "sibling_environ_unreadable": all(
            _blocked(lambda p=p: Path(f"/proc/{p}/environ").read_bytes()) for p in siblings
        ),
        "sibling_fds_unreadable": all(
            _blocked(lambda p=p: list(Path(f"/proc/{p}/fd").iterdir())) for p in siblings
        ),
        "sibling_unsignalable": all(_blocked(lambda p=p: os.kill(p, 0)) for p in siblings),
        "parent_environ_unreadable": _blocked(
            lambda: Path(f"/proc/{os.getppid()}/environ").read_bytes()
        ),
        "evaluator_mount_hidden": _blocked(lambda: list(Path("/run/oro").iterdir()))
        and _blocked(lambda: Path("/run/oro/logs/problems.jsonl").read_text()),
    }
    # Hold the slot until the slowest sibling has probed us too.
    time.sleep(2)
    return [{"task_isolation": report}]
