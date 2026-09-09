from __future__ import annotations

import threading

import pytest

from validator.generated_progress_reporter import GeneratedProgressReporter


def _result(task_id: str) -> dict:
    return {"task_id": task_id}


def test_flush_sends_only_unacknowledged_deltas() -> None:
    batches: list[list[dict]] = []
    reporter = GeneratedProgressReporter(
        registry=None,  # type: ignore[arg-type]
        emit_batch=lambda batch: batches.append(batch),
    )
    first, second = _result("one"), _result("two")

    reporter.flush([first])
    reporter.flush([first, second])
    reporter.flush([first, second])

    assert batches == [[first], [second]]


def test_failed_flush_remains_pending_for_retry() -> None:
    attempts = 0

    def emit(_batch: list[dict]) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary failure")

    reporter = GeneratedProgressReporter(
        registry=None,  # type: ignore[arg-type]
        emit_batch=emit,
    )
    result = _result("one")

    with pytest.raises(RuntimeError, match="temporary failure"):
        reporter.flush([result])
    reporter.flush([result])

    assert attempts == 2


def test_reporter_reads_registry_off_thread() -> None:
    emitted = threading.Event()

    class Registry:
        @staticmethod
        def terminal_results() -> list[dict]:
            return [_result("one")]

    reporter = GeneratedProgressReporter(
        Registry(),  # type: ignore[arg-type]
        emit_batch=lambda _batch: emitted.set(),
        report_interval=0,
        poll_interval=0.001,
    )

    reporter.start()
    assert emitted.wait(1)
    reporter.stop()
    reporter.stop()
