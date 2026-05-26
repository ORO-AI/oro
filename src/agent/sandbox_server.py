"""HTTP wrapper around `execute_single_problem`.

Exposes the sandbox executor as `POST /run` for callers that want to submit
work over loopback instead of via the one-shot CLI. Running the agent inside
this container preserves the same process isolation the CLI relies on.

Run:

    python -m src.agent.sandbox_server --host 0.0.0.0 --port 7000

Endpoints:

* `GET /health` — liveness probe.
* `POST /run` — body `{problem, agent_code, timeout}`; writes `agent_code`
  to a tempfile, runs `execute_single_problem`, returns the canonical
  envelope (same shape as the JSONL writer).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent.sandbox_executor import build_result_envelope, execute_single_problem

app = FastAPI()

_MAX_AGENT_CODE_BYTES = 2 * 1024 * 1024


class RunRequest(BaseModel):
    problem: dict[str, Any]
    agent_code: str = Field(..., max_length=_MAX_AGENT_CODE_BYTES)
    timeout: float = Field(300.0, ge=1.0, le=900.0)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run")
def run(req: RunRequest) -> dict[str, Any]:
    if not req.agent_code.strip():
        raise HTTPException(status_code=400, detail="agent_code is empty")

    with NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(req.agent_code)
        agent_path = fh.name

    try:
        result = execute_single_problem(
            problem=req.problem, timeout=req.timeout, agent_file=agent_path
        )
    finally:
        Path(agent_path).unlink(missing_ok=True)

    return build_result_envelope(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7000)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
