"""Tests for bounded sandbox-output capture (ORO-1414, Follow-up 1).

A misbehaving agent can write unbounded stdout; the validator must keep at
most a fixed number of bytes on disk while still draining the stream so the
child process never blocks on a full pipe.

Lives in subnet/tests/ (not subnet/tests/validator/) because the unit is pure
stdlib and must not pull in the validator conftest's heavy import chain.
"""

from __future__ import annotations

import io
import os
import stat
import subprocess
import sys
import threading
import time

import pytest

from validator.bounded_io import RunStoppedEarly, drain_capped, read_text_lossy, run_capped


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


def test_read_text_lossy_tolerates_non_utf8(tmp_path):
    # An agent can emit arbitrary bytes; reading the captured log must not raise.
    path = tmp_path / "sandbox_stdout.log"
    path.write_bytes(b"before \xff\xfe\x80 after")

    text = read_text_lossy(path, max_bytes=4096)

    assert "before " in text
    assert "after" in text  # decode replaced the invalid bytes instead of raising


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


def test_run_capped_stops_early_on_terminal_condition(tmp_path):
    stop = threading.Event()
    timer = threading.Timer(0.1, stop.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(RunStoppedEarly):
            run_capped(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                stdout_path=tmp_path / "o.log",
                stderr_path=tmp_path / "e.log",
                max_bytes=4096,
                timeout=30,
                stop_event=stop,
            )
    finally:
        timer.join()
    assert time.monotonic() - started < 5


def test_run_capped_reaps_child_when_log_open_fails(tmp_path, monkeypatch):
    # Popen starts the child before the cap files are opened. If open() fails
    # (here: a missing parent dir; in prod: ENOSPC on a full disk — the case
    # this guards), run_capped must still reap the child, not leak it running.
    import validator.bounded_io as bio

    captured: dict = {}
    real_popen = subprocess.Popen

    def spy_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        captured["proc"] = proc
        return proc

    monkeypatch.setattr(bio.subprocess, "Popen", spy_popen)

    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    with pytest.raises(FileNotFoundError):
        run_capped(
            cmd,
            stdout_path=tmp_path / "missing_dir" / "o.log",  # open() raises
            stderr_path=tmp_path / "e.log",
            max_bytes=4096,
            timeout=30,
        )

    proc = captured["proc"]
    proc.wait(timeout=5)  # must not hang; the child was killed on the way out
    assert proc.poll() is not None


def test_read_text_lossy_reports_a_non_regular_log(tmp_path):
    other = tmp_path / "other.log"
    other.write_text("other content")
    log = tmp_path / "sandbox_stderr.log"
    log.symlink_to(other)

    text = read_text_lossy(log, max_bytes=4096)

    assert "other content" not in text
    assert "unreadable sandbox log" in text


def test_run_capped_requires_plain_capture_files(tmp_path):
    target = tmp_path / "target"
    target.write_text("keep")
    (tmp_path / "e.log").symlink_to(target)

    with pytest.raises(OSError):
        run_capped(
            [sys.executable, "-c", "print('x')"],
            stdout_path=tmp_path / "o.log",
            stderr_path=tmp_path / "e.log",
            max_bytes=4096,
            timeout=30,
        )

    assert target.read_text() == "keep"


def test_run_capped_creates_validator_only_capture_files(tmp_path):
    previous = os.umask(0)
    try:
        run_capped(
            [sys.executable, "-c", "print('x')"],
            stdout_path=tmp_path / "o.log",
            stderr_path=tmp_path / "e.log",
            max_bytes=4096,
            timeout=30,
        )
    finally:
        os.umask(previous)

    for name in ("o.log", "e.log"):
        assert stat.S_IMODE((tmp_path / name).stat().st_mode) == 0o600


def _write_owned(path, name, size, uid):
    import os as _os

    f = path / name
    f.write_bytes(b"\0" * size)
    # chown may require privilege; skip the test when it isn't available.
    try:
        _os.chown(f, uid, -1)
    except (PermissionError, OSError):
        pass
    return f


def test_dir_usage_counts_only_the_sandbox_uid(tmp_path):
    from validator.bounded_io import dir_usage_for_uid

    me = os.getuid()
    (tmp_path / "mine.jsonl").write_bytes(b"x" * 8192)
    other = tmp_path / "root_owned.log"
    other.write_bytes(b"y" * 8192)
    # Count for a uid that owns nothing here: zero.
    assert dir_usage_for_uid(tmp_path, me + 12345) == (0, 0)
    used, count = dir_usage_for_uid(tmp_path, me)
    assert count == 2  # both files are owned by the test uid
    assert used >= 16384


def test_dir_usage_ignores_sparse_holes(tmp_path):
    from validator.bounded_io import dir_usage_for_uid

    sparse = tmp_path / "sparse.bin"
    with open(sparse, "wb") as f:
        f.truncate(1 << 30)  # 1 GiB logical, ~0 allocated blocks
    used, count = dir_usage_for_uid(tmp_path, os.getuid())
    assert count == 1
    assert used < 1 << 20  # allocated blocks, not the 1 GiB logical size


def test_guard_trips_on_bytes_then_file_count(tmp_path):
    from validator.bounded_io import make_eval_dir_guard

    uid = os.getuid()
    guard = make_eval_dir_guard(
        tmp_path, uid=uid, max_bytes=10_000, max_files=3, interval=0.0
    )
    assert guard() is None
    (tmp_path / "big").write_bytes(b"x" * 20_000)
    assert "bytes" in (guard() or "")

    import shutil

    shutil.rmtree(tmp_path)
    tmp_path.mkdir()
    guard = make_eval_dir_guard(
        tmp_path, uid=uid, max_bytes=1 << 30, max_files=2, interval=0.0
    )
    for i in range(4):
        (tmp_path / f"f{i}").write_bytes(b"x")
    assert "files" in (guard() or "")


def test_guard_rate_limits_its_walk(tmp_path):
    from validator.bounded_io import make_eval_dir_guard

    guard = make_eval_dir_guard(
        tmp_path, uid=os.getuid(), max_bytes=1, max_files=1, interval=3600.0
    )
    (tmp_path / "over").write_bytes(b"xx")
    first = guard()  # first call walks and trips
    assert first is not None
    (tmp_path / "over").unlink()
    # within the interval, the guard doesn't walk again, so no new verdict
    assert guard() is None


def test_run_capped_kills_a_sandbox_over_its_disk_budget(tmp_path):
    from validator.bounded_io import SandboxResourceExceeded

    target = tmp_path / "eval"
    target.mkdir()
    trip = {"n": 0}

    def guard():
        trip["n"] += 1
        return "over budget" if trip["n"] >= 2 else None

    started = time.monotonic()
    with pytest.raises(SandboxResourceExceeded, match="over budget"):
        run_capped(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout_path=tmp_path / "o.log",
            stderr_path=tmp_path / "e.log",
            max_bytes=4096,
            timeout=30,
            size_guard=guard,
        )
    assert time.monotonic() - started < 10  # killed promptly, not after the 30s timeout
