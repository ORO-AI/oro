"""Split sandbox output.jsonl by problem for per-problem trajectory upload."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import UUID


def split_output_by_problem(
    output_file: Path,
    problem_ids: list[UUID],
) -> dict[str, bytes]:
    """Read sandbox output.jsonl and group per-problem upload payloads.

    Sandbox writes one envelope dict per line (ORO-907 IPC):
    ``{"problem_id": "...", "status": "...", "dialogue": [steps], ...}``.
    The Frontend's ``Trajectory`` type is ``TrajectoryStep[]``, so we
    extract ``dialogue`` and use *that* JSON array as the upload payload —
    not the whole envelope. The envelope's status / timing / error fields
    are not lost: they flow back via the progress-report path
    (ProgressReporter → Backend → eval-run detail endpoint).

    Returns ``{problem_id_str: bytes_to_upload}`` (un-gzipped — caller
    handles compression). If parsing produces zero entries we fall back
    to uploading the entire file under ``problem_ids[0]`` so a corrupt
    run still has *some* artifact attached for forensics.
    """
    problem_lines: dict[str, bytes] = {}

    for raw_line in output_file.read_bytes().splitlines():
        if not raw_line.strip():
            continue
        try:
            envelope = json.loads(raw_line)
        except json.JSONDecodeError:
            logging.debug(f"Skipping unparseable JSONL line in {output_file}")
            continue

        if not (isinstance(envelope, dict) and envelope.get("problem_id")):
            logging.debug(
                f"Skipping line missing problem_id envelope in {output_file}"
            )
            continue

        pid = str(envelope["problem_id"])
        dialogue = envelope.get("dialogue") or []
        problem_lines[pid] = json.dumps(dialogue).encode("utf-8")

    if not problem_lines and problem_ids:
        logging.warning(
            "Could not split output by problem_id, uploading as single file"
        )
        problem_lines[str(problem_ids[0])] = output_file.read_bytes()

    return problem_lines
