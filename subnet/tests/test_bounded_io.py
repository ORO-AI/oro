"""Tests for bounded sandbox-output capture (ORO-1414, Follow-up 1).

A misbehaving agent can write unbounded stdout; the validator must keep at
most a fixed number of bytes on disk while still draining the stream so the
child process never blocks on a full pipe.

Lives in subnet/tests/ (not subnet/tests/validator/) because the unit is pure
stdlib and must not pull in the validator conftest's heavy import chain.
"""

from __future__ import annotations

import io
import subprocess
import sys
import time

import pytest

from validator.bounded_io import drain_capped, run_capped


def test_under_cap_writes_everything_verbatim():
    src = io.BytesIO(b"hello world")
    dst = io.BytesIO()

    total = drain_capped(src, dst, max_bytes=100)

    assert total == 11
    assert dst.getvalue() == b"hello world"


def test_over_cap_truncates_content_and_appends_marker():
    src = io.BytesIO(b"A" * 1000)
    dst = io.BytesIO()

    total = drain_capped(src, dst, max_bytes=100)

    out = dst.getvalue()
    # full source was drained (so the producer never blocks)...
    assert total == 1000
    # ...but only the cap's worth of content reached disk, plus a marker
    assert out[:100] == b"A" * 100
    assert b"truncated" in out
    assert out.count(b"A") == 100
    assert len(out) < 1000


def test_run_capped_bounds_a_runaway_stdout_to_disk(tmp_path):
    out_path = tmp_path / "stdout.log"
    err_path = tmp_path / "stderr.log"
    # Agent that spews ~5 MB of stdout; cap at 4 KB.
    cmd = [sys.executable, "-c", "import sys; sys.stdout.write('A' * (5 * 1024 * 1024))"]

    returncode = run_capped(
        cmd,
        stdout_path=out_path,
        stderr_path=err_path,
        max_bytes=4096,
        timeout=30,
    )

    assert returncode == 0
    data = out_path.read_bytes()
    # disk write stayed near the cap, not the 5 MB the process produced
    assert len(data) < 4096 + 512
    assert data[:4096] == b"A" * 4096
    assert b"truncated" in data


def test_run_capped_kills_on_timeout(tmp_path):
    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]

    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run_capped(
            cmd,
            stdout_path=tmp_path / "o.log",
            stderr_path=tmp_path / "e.log",
            max_bytes=4096,
            timeout=1,
        )
    # process was killed at the timeout, not waited out for 30s
    assert time.monotonic() - start < 10
