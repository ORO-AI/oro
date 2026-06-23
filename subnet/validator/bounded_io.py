"""Bounded capture of a sandbox output stream to disk (ORO-1414, Follow-up 1).

A misbehaving agent can emit unbounded stdout/stderr. The validator captures
that output to a host file, so the sandbox's own cgroup limits do not apply and
a single run can fill the host disk (the kevgol incident: ~14 GiB to one
``sandbox_stdout.log``). ``drain_capped`` keeps at most ``max_bytes`` on disk
while still reading the source to EOF, so the child never blocks on a full pipe.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import BinaryIO

_DEFAULT_CHUNK_SIZE = 64 * 1024


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


def run_capped(
    cmd: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    max_bytes: int,
    timeout: float,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> int:
    """Run ``cmd``, capturing stdout/stderr to files each capped at ``max_bytes``.

    Drop-in for ``subprocess.run(cmd, stdout=f, stderr=f, timeout=...)`` that
    bounds how much reaches disk. Reader threads drain both pipes concurrently
    (so the child never blocks on a full pipe and stdout/stderr can't deadlock).
    Raises ``subprocess.TimeoutExpired`` and kills the child on timeout, matching
    ``subprocess.run``. Returns the child's exit code.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with (
        open(stdout_path, "wb") as out_file,
        open(stderr_path, "wb") as err_file,
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
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
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
    return proc.returncode
