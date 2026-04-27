"""Tail the sandbox envelope stream and yield typed records.

Stateless wrt scoring/reasoning/backend reporting — those are the orchestrator's
job. This class owns:
  - file position tracking
  - envelope JSON parsing
  - truncation handling
  - skipping malformed/unrecognized lines
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from bittensor.utils.btlogging import logging

from src.agent.sandbox_status import SandboxProblemStatus


@dataclass
class ErrorInfo:
    """Structured error info parsed from the envelope's `error` field."""

    type: str
    message: str


@dataclass
class ProblemRecord:
    """One per-problem outcome, parsed from a single envelope line."""

    problem_id: str
    status: SandboxProblemStatus
    execution_time: float
    inference_failure_count: int
    inference_total: int
    error: Optional[ErrorInfo]
    dialogue: Optional[List[Dict[str, Any]]]


class OutputWatcher:
    """Tails a sandbox `output.jsonl` stream and yields parsed records.

    Each call to :meth:`read_new` returns an iterator over envelopes appended
    since the previous call. The watcher tolerates:

    - Missing file (returns no records)
    - File truncation (resets position to 0)
    - Malformed JSON lines (logged and skipped)
    - Unrecognized status values (logged and skipped)
    """

    def __init__(self, output_file: Path):
        self.output_file = Path(output_file)
        self._file_position = 0

    def reset(self) -> None:
        """Reset file position. Called when re-using the watcher across runs."""
        self._file_position = 0

    def read_new(self) -> Iterator[ProblemRecord]:
        """Yield records appended since the last call. Generator."""
        if not self.output_file.exists():
            return

        try:
            file_size = self.output_file.stat().st_size
        except OSError as e:
            logging.warning(f"OutputWatcher: stat failed: {e}")
            return

        if file_size < self._file_position:
            logging.info("OutputWatcher: file truncated, resetting position")
            self._file_position = 0

        try:
            with open(self.output_file) as f:
                f.seek(self._file_position)
                new_lines = f.readlines()
                self._file_position = f.tell()
        except OSError as e:
            logging.warning(f"OutputWatcher: read failed: {e}")
            self._file_position = 0
            return

        for line in new_lines:
            line = line.strip()
            if not line:
                continue
            record = self._parse(line)
            if record is not None:
                yield record

    @staticmethod
    def _parse(line: str) -> Optional[ProblemRecord]:
        try:
            line = line.replace("\x00", "")
            envelope = json.loads(line)
        except json.JSONDecodeError as e:
            logging.warning(f"Skipping malformed envelope line: {e}")
            return None
        if not isinstance(envelope, dict):
            logging.warning(
                f"Skipping non-dict envelope line: {type(envelope).__name__}"
            )
            return None

        problem_id = envelope.get("problem_id")
        raw_status = envelope.get("status")
        if not problem_id or not raw_status:
            logging.warning("Skipping envelope without problem_id or status")
            return None

        try:
            status = SandboxProblemStatus(raw_status)
        except ValueError:
            logging.warning(
                f"Skipping envelope with unrecognized status '{raw_status}'"
            )
            return None

        error_obj = envelope.get("error")
        error_info: Optional[ErrorInfo] = None
        if isinstance(error_obj, dict):
            error_info = ErrorInfo(
                type=str(error_obj.get("type") or "UnknownError"),
                message=str(error_obj.get("message") or ""),
            )

        dialogue = envelope.get("dialogue")
        if not isinstance(dialogue, list):
            dialogue = None

        return ProblemRecord(
            problem_id=str(problem_id),
            status=status,
            execution_time=float(envelope.get("execution_time") or 0.0),
            inference_failure_count=int(envelope.get("inference_failure_count") or 0),
            inference_total=int(envelope.get("inference_total") or 0),
            error=error_info,
            dialogue=dialogue,
        )
