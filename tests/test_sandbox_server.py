"""Smoke tests for sandbox_server FastAPI app."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.agent.sandbox_executor import ExecutionResult
from src.agent.sandbox_server import app
from src.agent.sandbox_status import SandboxProblemStatus


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _mk_result(problem_id: str, *, success: bool = True) -> ExecutionResult:
    return ExecutionResult(
        query=f"q-{problem_id}",
        success=success,
        result=[{"action": "noop"}] if success else None,
        execution_time=0.5,
        problem_id=problem_id,
        status=SandboxProblemStatus.SUCCESS if success else SandboxProblemStatus.FAILED,
    )


def _patched_executor(results_by_pid: dict[str, ExecutionResult], capture=None):
    """Patch execute_single_problem to return canned results keyed by problem_id."""

    def fake(problem, timeout, agent_file):
        if capture is not None:
            capture(problem, timeout, agent_file)
        return results_by_pid[problem["problem_id"]]

    return patch("src.agent.sandbox_server.execute_single_problem", side_effect=fake)


# ---------------------------------------------------------------------------
# /health + /run
# ---------------------------------------------------------------------------


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_run_rejects_empty_agent_code(client: TestClient) -> None:
    r = client.post("/run", json={"problem": {"query": "x"}, "agent_code": "   "})
    assert r.status_code == 400


def test_run_invokes_executor_and_returns_envelope(client: TestClient) -> None:
    fake = ExecutionResult(
        query="test",
        success=True,
        result=[{"action": "search", "args": {"q": "x"}}],
        execution_time=1.5,
        problem_id="p-1",
        status=SandboxProblemStatus.SUCCESS,
    )
    with patch(
        "src.agent.sandbox_server.execute_single_problem", return_value=fake
    ) as m:
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


def test_run_rejects_timeout_out_of_range(client: TestClient) -> None:
    r = client.post(
        "/run",
        json={"problem": {"query": "x"}, "agent_code": "x = 1\n", "timeout": 0.1},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /run_matrix
# ---------------------------------------------------------------------------


def _matrix_body(*, n_problems: int, n_agents: int, **kwargs) -> dict:
    return {
        "problems": [
            {"query": f"q-{i}", "problem_id": f"p{i}"} for i in range(n_problems)
        ],
        "agents": [f"# agent {a}\nx = {a}\n" for a in range(n_agents)],
        "timeout": 30.0,
        **kwargs,
    }


def _result_for_problem(problem_id: str, *, success: bool = True) -> ExecutionResult:
    return _mk_result(problem_id, success=success)


def test_run_matrix_returns_cross_product_in_order(client: TestClient) -> None:
    body = _matrix_body(n_problems=3, n_agents=2)
    fakes = {f"p{i}": _result_for_problem(f"p{i}") for i in range(3)}

    with _patched_executor(fakes):
        r = client.post("/run_matrix", json=body)

    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 6
    assert [(c["problem_idx"], c["agent_idx"]) for c in results] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    ]
    assert all(c["envelope"]["status"] == "SUCCESS" for c in results)


def test_run_matrix_honors_sparse_pairs(client: TestClient) -> None:
    body = _matrix_body(n_problems=3, n_agents=3, pairs=[[0, 1], [2, 0], [1, 2]])
    fakes = {f"p{i}": _result_for_problem(f"p{i}") for i in range(3)}

    with _patched_executor(fakes):
        r = client.post("/run_matrix", json=body)

    assert r.status_code == 200
    results = r.json()["results"]
    assert [(c["problem_idx"], c["agent_idx"]) for c in results] == [
        (0, 1),
        (2, 0),
        (1, 2),
    ]


def test_run_matrix_writes_each_agent_once(client: TestClient) -> None:
    """Agents list defines the on-disk files. 4×3 matrix → 3 agent files."""
    body = _matrix_body(n_problems=4, n_agents=3)
    fakes = {f"p{i}": _result_for_problem(f"p{i}") for i in range(4)}
    seen_paths: set[str] = set()

    with _patched_executor(fakes, capture=lambda p, t, f: seen_paths.add(f)):
        r = client.post("/run_matrix", json=body)

    assert r.status_code == 200
    assert len(seen_paths) == 3


def test_run_matrix_rejects_empty_agent(client: TestClient) -> None:
    body = _matrix_body(n_problems=1, n_agents=2)
    body["agents"][1] = "   "
    r = client.post("/run_matrix", json=body)
    assert r.status_code == 400
    assert "agents[1]" in r.json()["detail"]


@pytest.mark.parametrize("pair", [(-1, 0), (0, -1), (1, 0), (0, 1)])
def test_run_matrix_rejects_out_of_range_pair(client: TestClient, pair) -> None:
    body = _matrix_body(n_problems=1, n_agents=1, pairs=[list(pair)])
    r = client.post("/run_matrix", json=body)
    assert r.status_code == 400
    assert "out of range" in r.json()["detail"]


def test_run_matrix_e2e_with_real_executor(client: TestClient) -> None:
    """No mocks: actually fan out forkserver children and confirm wiring."""
    agent_code = (
        "def agent_main(problem):\n"
        "    return [{'think': 'noop', 'response': 'done', 'extra_info': {}}]\n"
    )
    body = {
        "problems": [
            {"query": "q-0", "problem_id": "p0"},
            {"query": "q-1", "problem_id": "p1"},
        ],
        "agents": [agent_code, agent_code.replace("noop", "alt")],
        "timeout": 10.0,
    }
    r = client.post("/run_matrix", json=body)

    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 4
    assert [(c["problem_idx"], c["agent_idx"]) for c in results] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    for c in results:
        env = c["envelope"]
        assert env["status"] == "SUCCESS"
        assert env["execution_time"] > 0
        assert env["dialogue"] is not None
    # Cell (0,0) and (1,0) share agent 0 (noop); (0,1) and (1,1) share agent 1 (alt)
    assert results[0]["envelope"]["dialogue"][0]["think"] == "noop"
    assert results[1]["envelope"]["dialogue"][0]["think"] == "alt"
