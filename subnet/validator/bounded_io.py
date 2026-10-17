"""Bounded capture of a sandbox output stream to disk (ORO-1414, Follow-up 1).

A misbehaving agent can emit unbounded stdout/stderr. The validator captures
that output to a host file, so the sandbox's own cgroup limits do not apply and
a single run can fill the host disk (the kevgol incident: ~14 GiB to one
``sandbox_stdout.log``). ``drain_capped`` keeps at most ``max_bytes`` on disk
while still reading the source to EOF, so the child never blocks on a full pipe.
"""

from __future__ import annotations

import os
import stat
import subprocess
import threading
import time
from pathlib import Path
from typing import BinaryIO, Callable

from src.agent.sandbox_executor import read_regular_file

_DEFAULT_CHUNK_SIZE = 64 * 1024
_MARKER_ROOM = 1024  # drain_capped's truncation marker follows the capped bytes


class RunStoppedEarly(RuntimeError):
    """The caller cancelled a running sandbox after a terminal run condition."""


class SandboxResourceExceeded(RunStoppedEarly):
    """The sandbox wrote past its per-evaluation disk budget and was killed."""


def dir_usage_for_uid(path: Path, uid: int) -> tuple[int, int]:
    """Return (allocated bytes, file count) of *uid*-owned files under *path*.

    Counts allocated blocks (``st_blocks``), so a sparse file contributes only
    the disk it actually uses. Only files owned by the sandbox uid are counted,
    so validator-written inputs and capture logs don't inflate the total. A
    directory that disappears mid-walk (cleanup) is skipped, not raised.
    """
    total_bytes = 0
    count = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if entry.is_dir(follow_symlinks=False):
                stack.append(Path(entry.path))
            elif info.st_uid == uid and stat.S_ISREG(info.st_mode):
                total_bytes += info.st_blocks * 512
                count += 1
    return total_bytes, count


def make_eval_dir_guard(
    path: Path, *, uid: int, max_bytes: int, max_files: int, interval: float = 1.0
) -> Callable[[], str | None]:
    """Build a guard that reports a reason once *uid* writes past the budget.

    The walk runs at most once per *interval* seconds, so polling the guard
    frequently stays cheap on a normal run.
    """
    state = {"next_check": 0.0}

    def guard() -> str | None:
        now = time.monotonic()
        if now < state["next_check"]:
            return None
        state["next_check"] = now + interval
        used_bytes, files = dir_usage_for_uid(path, uid)
        if used_bytes > max_bytes:
            return f"sandbox wrote {used_bytes} bytes, over the {max_bytes} limit"
        if files > max_files:
            return f"sandbox created {files} files, over the {max_files} limit"
        return None

    return guard


def drain_capped(
    src: BinaryIO,
    dst: BinaryIO,
    max_bytes: int,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> int:
    """Copy ``src`` to ``dst``, writing at most ``max_bytes`` of content.

    Reads ``src`` to EOF regardless of the cap (draining) so the producer is
    never blocked by a full pipe. If the source exceeds ``max_bytes``, a
    truncation marker noting how many bytes were dropped is appended to ``dst``.

    Returns the total number of bytes read from ``src``.
    """
    total = 0
    written = 0
    while True:
        chunk = src.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if written < max_bytes:
            to_write = min(len(chunk), max_bytes - written)
            dst.write(chunk[:to_write])
            written += to_write

    if total > max_bytes:
        dropped = total - max_bytes
        dst.write(
            f"\n...[output truncated: capped at {max_bytes} bytes, "
            f"dropped {dropped} bytes]\n".encode()
        )
    return total


def read_text_lossy(path: Path, max_bytes: int) -> str:
    """Read a log captured by ``run_capped`` with the same ``max_bytes``.

    An agent can emit arbitrary bytes, so a plain ``Path.read_text()`` can raise
    ``UnicodeDecodeError`` and lose the very stderr/stdout we want for debugging.
    Invalid bytes are replaced instead of raising. A log that is not a plain
    file within the cap is reported rather than read.
    """
    try:
        content = read_regular_file(path, max_bytes + _MARKER_ROOM)
    except (OSError, ValueError) as exc:
        return f"[unreadable sandbox log: {type(exc).__name__}]"
    return content.decode(errors="replace")


def _open_capture(path: str, flags: int) -> int:
    # Capture only into a plain file that only the validator can read or write.
    return os.open(path, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)


def run_capped(
    cmd: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    max_bytes: int,
    timeout: float,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    stop_event: threading.Event | None = None,
    size_guard: Callable[[], str | None] | None = None,
) -> int:
    """Run ``cmd``, capturing stdout/stderr to files each capped at ``max_bytes``.

    Drop-in for ``subprocess.run(cmd, stdout=f, stderr=f, timeout=...)`` that
    bounds how much reaches disk. Reader threads drain both pipes concurrently
    (so the child never blocks on a full pipe and stdout/stderr can't deadlock).
    Raises ``subprocess.TimeoutExpired`` and kills the child on timeout, matching
    ``subprocess.run``. Returns the child's exit code.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Once Popen succeeds the child is running; any failure before we normally
    # return (e.g. opening the cap files raises ENOSPC — the very disk-full case
    # this guards against) must still kill it, or we leak an undrained, unwaited
    # process whose pipe eventually fills and blocks.
    try:
        with (
            open(stdout_path, "wb", opener=_open_capture) as out_file,
            open(stderr_path, "wb", opener=_open_capture) as err_file,
        ):
            assert proc.stdout is not None and proc.stderr is not None
            readers = [
                threading.Thread(
                    target=drain_capped,
                    args=(pipe, dst, max_bytes),
                    kwargs={"chunk_size": chunk_size},
                    daemon=True,
                )
                for pipe, dst in ((proc.stdout, out_file), (proc.stderr, err_file))
            ]
            for t in readers:
                t.start()
            try:
                if stop_event is None and size_guard is None:
                    proc.wait(timeout=timeout)
                else:

                    def _stop(exc: BaseException) -> None:
                        proc.kill()
                        proc.wait()
                        raise exc

                    deadline = time.monotonic() + timeout
                    while proc.poll() is None:
                        if stop_event is not None and stop_event.is_set():
                            _stop(
                                RunStoppedEarly(
                                    "sandbox stopped after terminal run condition"
                                )
                            )
                        reason = size_guard() if size_guard is not None else None
                        if reason is not None:
                            _stop(SandboxResourceExceeded(reason))
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(cmd, timeout)
                        if stop_event is not None:
                            stop_event.wait(min(0.25, remaining))
                        else:
                            time.sleep(min(0.25, remaining))
            except subprocess.TimeoutExpired:
                # Kill BEFORE joining: the readers only reach EOF once the child
                # exits, so joining a live child would deadlock.
                proc.kill()
                proc.wait()
                raise
            finally:
                # The child owns the only write ends of these pipes, so once it
                # exits (or is killed above) the reads hit EOF and the threads
                # finish; join before the dest files close. daemon=True keeps a
                # wedged reader from blocking interpreter shutdown as a backstop.
                for t in readers:
                    t.join()
    except BaseException:
        # Reached if the cap files fail to open before any reader starts (no
        # readers to join), or re-raised from the timeout path above (child
        # already dead, so kill/wait are no-ops). Either way, never leak the
        # child.
        proc.kill()
        proc.wait()
        raise
    return proc.returncode
