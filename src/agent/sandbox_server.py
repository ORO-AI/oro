"""HTTP wrapper around `execute_single_problem`.

Exposes the sandbox executor over loopback so callers can submit work
without using the one-shot CLI. Running agents inside this container
preserves the same process isolation the CLI relies on.

Run:

    python -m src.agent.sandbox_server --host 0.0.0.0 --port 7000

Endpoints:

* `GET /health` — liveness probe.
* `POST /run` — body `{problem, agent_code, timeout}`; runs one
  `(problem, agent)` pair, returns the canonical envelope.
* `POST /run_matrix` — body `{problems, agents, pairs?, timeout,
  max_workers?}`. Evaluates a cross-product of problems × agents (or
  an explicit list of `(problem_idx, agent_idx)` cells) concurrently
  inside this container. Lets a caller pay for each agent_code once
  on the wire even when reusing it across many problems.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent.sandbox_executor import build_result_envelope, execute_single_problem

app = FastAPI()

_MAX_AGENT_CODE_BYTES = 2 * 1024 * 1024
_DEFAULT_BATCH_WORKERS = 15


class RunRequest(BaseModel):
    problem: dict[str, Any]
    agent_code: str = Field(..., max_length=_MAX_AGENT_CODE_BYTES)
    timeout: float = Field(300.0, ge=1.0, le=900.0)


class RunMatrixRequest(BaseModel):
    problems: list[dict[str, Any]] = Field(..., min_length=1)
    agents: list[str] = Field(..., min_length=1)
    # Optional sparse selection. Each entry = (problem_idx, agent_idx).
    # Default (None) = full cross-product problems × agents.
    pairs: Optional[list[tuple[int, int]]] = None
    timeout: float = Field(300.0, ge=1.0, le=900.0)
    max_workers: Optional[int] = Field(None, ge=1)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run")
def run(req: RunRequest) -> dict[str, Any]:
    if not req.agent_code.strip():
        raise HTTPException(status_code=400, detail="agent_code is empty")

    with NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(req.agent_code)
        agent_path = fh.name

    try:
        result = execute_single_problem(
            problem=req.problem, timeout=req.timeout, agent_file=agent_path
        )
    finally:
        Path(agent_path).unlink(missing_ok=True)

    return build_result_envelope(result)


@app.post("/run_matrix")
def run_matrix(req: RunMatrixRequest) -> dict[str, list[dict[str, Any]]]:
    """Evaluate problems × agents (or selected cells) in parallel."""
    for ai, code in enumerate(req.agents):
        if not code.strip():
            raise HTTPException(status_code=400, detail=f"agents[{ai}] is empty")

    n_p, n_a = len(req.problems), len(req.agents)
    if req.pairs is None:
        cells = list(product(range(n_p), range(n_a)))
    else:
        for k, (pi, ai) in enumerate(req.pairs):
            if not (0 <= pi < n_p) or not (0 <= ai < n_a):
                raise HTTPException(
                    status_code=400, detail=f"pairs[{k}] out of range: ({pi}, {ai})"
                )
        cells = req.pairs

    workers = req.max_workers or min(len(cells), _DEFAULT_BATCH_WORKERS)

    with TemporaryDirectory(prefix="sandbox_batch_") as batch_dir:
        agent_paths: list[str] = []
        for ai, code in enumerate(req.agents):
            path = Path(batch_dir) / f"agent_{ai}.py"
            path.write_text(code, encoding="utf-8")
            agent_paths.append(str(path))

        def _eval(cell: tuple[int, int]) -> Any:
            pi, ai = cell
            return execute_single_problem(
                req.problems[pi], req.timeout, agent_paths[ai]
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            envelopes = list(pool.map(_eval, cells))

    return {
        "results": [
            {"problem_idx": pi, "agent_idx": ai, "envelope": build_result_envelope(env)}
            for (pi, ai), env in zip(cells, envelopes)
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7000)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
