"""Smoke tests for sandbox_server FastAPI app."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.agent.sandbox_executor import ExecutionResult
from src.agent.sandbox_status import SandboxProblemStatus
from src.agent.sandbox_server import app

client = TestClient(app)


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_run_rejects_empty_agent_code() -> None:
    r = client.post("/run", json={"problem": {"query": "x"}, "agent_code": "   "})
    assert r.status_code == 400


def test_run_invokes_executor_and_returns_envelope() -> None:
    fake_result = ExecutionResult(
        query="test",
        success=True,
        result=[{"action": "search", "args": {"q": "x"}}],
        execution_time=1.5,
        problem_id="p-1",
        status=SandboxProblemStatus.SUCCESS,
    )
    with patch("src.agent.sandbox_server.execute_single_problem", return_value=fake_result) as m:
        r = client.post(
            "/run",
            json={
                "problem": {"query": "test"},
                "agent_code": "def agent_main(*a, **k): return []\n",
                "timeout": 30.0,
            },
        )

    assert r.status_code == 200
    body = r.json()
    assert body["problem_id"] == "p-1"
    assert body["status"] == "SUCCESS"
    assert body["dialogue"][0]["action"] == "search"
    assert m.call_args.kwargs["timeout"] == 30.0


def test_run_rejects_timeout_out_of_range() -> None:
    r = client.post(
        "/run",
        json={"problem": {"query": "x"}, "agent_code": "x = 1\n", "timeout": 0.1},
    )
    assert r.status_code == 422
