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
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from bittensor.utils.btlogging import logging

from src.agent.sandbox_executor import MAX_SANDBOX_OUTPUT_BYTES, open_regular_file
from src.agent.sandbox_status import SandboxProblemStatus
from src.agent.types import Dialogue


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
    dialogue: Optional[Dialogue]


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
            with open_regular_file(self.output_file, MAX_SANDBOX_OUTPUT_BYTES) as f:
                file_size = os.fstat(f.fileno()).st_size
                if file_size < self._file_position:
                    logging.info("OutputWatcher: file truncated, resetting position")
                    self._file_position = 0
                f.seek(self._file_position)
                new_bytes = f.read(file_size - self._file_position)
                # Leave a partly written last line for the next poll.
                new_bytes = new_bytes[: new_bytes.rfind(b"\n") + 1]
                self._file_position += len(new_bytes)
        except (OSError, ValueError) as e:
            logging.warning(f"OutputWatcher: read failed, retry next poll: {e}")
            return

        new_lines = new_bytes.decode("utf-8", errors="replace").splitlines()
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
            return OutputWatcher._parse_unsafe(line)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logging.warning(f"Skipping malformed envelope line: {e}")
            return None

    @staticmethod
    def _parse_unsafe(line: str) -> Optional[ProblemRecord]:
        # Sandbox kill mid-write can leave NUL bytes in the line.
        line = line.replace("\x00", "")
        envelope = json.loads(line)
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
