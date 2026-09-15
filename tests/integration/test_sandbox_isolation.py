"""
Integration tests for sandbox network isolation.
Tests that sandbox containers are properly isolated and can only communicate through the proxy.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

from subnet.sandbox import build_sandbox_command
from tests.integration.conftest import (
    PROXY_CONTAINER,
    SEARCH_SERVER_CONTAINER,
    SESSION_RUNTIME_CONTAINER,
    exec_in_container,
    get_container_health,
    is_container_running,
)


class TestSandboxIsolation:
    """Tests for sandbox network isolation."""

    def test_generated_artifact_mounts_are_readonly(self, tmp_path):
        tmp_path.chmod(0o755)
        output_dir = tmp_path / "sandbox"
        output_dir.mkdir(mode=0o777)
        output_dir.chmod(0o777)
        inputs = {
            "problems.jsonl": json.dumps(
                {"problem_id": "artifact-probe", "query": "Probe artifact mounts"}
            )
            + "\n",
            "environment_sessions.json": '{"sessions": []}\n',
            "summary.json": '{"status": "running"}\n',
            "episode_results.jsonl": '{"trusted": true}\n',
        }
        for name, content in inputs.items():
            path = tmp_path / name
            path.write_text(content)
            path.chmod(0o666)
        command = build_sandbox_command(
            agent_host_path=str(
                Path("tests/integration/hostile_artifact_agent.py").resolve()
            ),
            logs_host_path=str(tmp_path),
            output_host_path=str(output_dir),
            problem_file_arg="/app/logs/problems.jsonl",
            output_path="/app/output/sandbox_output.jsonl",
            image=os.getenv(
                "INTEGRATION_SANDBOX_IMAGE", "ghcr.io/oro-ai/oro/sandbox:stable"
            ),
            network="none",
            max_workers=1,
            timeout=30,
        )
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=60, check=False
        )
        assert result.returncode == 0, result.stderr
        row = json.loads((output_dir / "sandbox_output.jsonl").read_text())
        assert row["status"] == "SUCCESS"
        report = row["dialogue"][0]["artifact_isolation"]
        assert len(report) == 9 and all(report.values()), report
        assert {name: (tmp_path / name).read_text() for name in inputs} == inputs

    def test_required_containers_running(self):
        """Test that required containers are running."""
        required = [SEARCH_SERVER_CONTAINER, PROXY_CONTAINER, SESSION_RUNTIME_CONTAINER]
        missing = [c for c in required if not is_container_running(c)]

        assert not missing, (
            f"Required containers are not running: {missing}. "
            "Please start them with: docker compose up -d search-server proxy"
        )

    def test_containers_healthy(self):
        """Test that required containers are healthy."""
        for container in (
            SEARCH_SERVER_CONTAINER,
            PROXY_CONTAINER,
            SESSION_RUNTIME_CONTAINER,
        ):
            health = get_container_health(container)
            if health != "healthy":
                pytest.skip(f"{container} may not be healthy yet (status: {health})")

    def test_internet_connectivity_blocked(self, sandbox_container):
        """Test that sandbox container cannot reach the internet."""
        result = exec_in_container(
            sandbox_container,
            ["curl", "-s", "--max-time", "5", "https://www.google.com"],
            timeout=10,
        )

        assert result.returncode != 0, (
            "Container can reach the internet (should be blocked)"
        )

    @pytest.mark.parametrize(
        "url",
        [
            "http://search-server:5632/health",
            "http://session-runtime:9101/health",
        ],
    )
    def test_direct_service_access_blocked(self, sandbox_container, url):
        """The sandbox cannot bypass the proxy to reach internal services."""
        result = exec_in_container(
            sandbox_container,
            ["curl", "-s", "--max-time", "5", url],
            timeout=10,
        )

        assert result.returncode != 0, f"Container can directly reach {url}"

    def test_proxy_access_allowed(self, sandbox_container):
        """Test that sandbox container can reach proxy."""
        result = exec_in_container(
            sandbox_container,
            ["curl", "-s", "--max-time", "5", "http://proxy:80/health"],
            timeout=10,
        )

        assert result.returncode == 0, f"Container cannot reach proxy: {result.stderr}"
        assert "healthy" in result.stdout.lower() or result.stdout.strip() == "healthy"

    def test_legacy_search_routes_are_gone(self, sandbox_container):
        """The ShoppingBench /search/* routes answer 410 and never reach search-server."""
        result = exec_in_container(
            sandbox_container,
            [
                "curl",
                "-s",
                "--max-time",
                "5",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                "http://proxy:80/search/find_product?q=test&page=1",
            ],
            timeout=10,
        )

        assert result.returncode == 0, f"Container cannot reach proxy: {result.stderr}"
        assert result.stdout.strip() == "410", (
            f"Expected 410 for a legacy search route, got: {result.stdout[:200]}"
        )

    def test_inference_route_is_live(self, sandbox_container):
        """/inference/* is owned by the njs validator: a body without a model is rejected there, not by the default deny."""
        result = exec_in_container(
            sandbox_container,
            [
                "curl",
                "-s",
                "--max-time",
                "5",
                "-w",
                "\n%{http_code}",
                "-H",
                "Content-Type: application/json",
                "-d",
                "{}",
                "http://proxy:80/inference/chat/completions",
            ],
            timeout=10,
        )

        assert result.returncode == 0, result.stderr
        body, status = result.stdout.rsplit("\n", 1)
        assert status == "400", (
            f"Expected the model validator to answer, got {status}: {body[:200]}"
        )
        assert "model" in json.loads(body)["error"]

    def test_session_calls_route_only_through_proxy(self, sandbox_container):
        """A grouped solver turn reaches the real SessionServer only through nginx."""

        result = exec_in_container(
            sandbox_container,
            [
                "curl",
                "-s",
                "--max-time",
                "5",
                "-w",
                "\n%{http_code}",
                "-H",
                "Content-Type: application/json",
                "-d",
                json.dumps(
                    {
                        "session_id": "integration-session",
                        "tool_contract_version": "oro_task_tools_v2",
                        "call_id": "integration-group-1",
                        "idempotency_key": "integration-group-1",
                        "turn": 1,
                        "calls": [
                            {"call_id": "search", "action": {"name": "search"}},
                            {"call_id": "message", "action": {"name": "message"}},
                        ],
                    }
                ),
                "http://proxy:80/environment/call",
            ],
            timeout=10,
        )

        assert result.returncode == 0, result.stderr
        body, status = result.stdout.rsplit("\n", 1)
        assert status == "200"
        response = json.loads(body)
        assert response["solver_turn_count"] == 1
        assert response["action_count"] == 2
        assert [call["call_id"] for call in response["calls"]] == [
            "search",
            "message",
        ]

    def test_environment_preflight_agent_runs_through_proxy(self, tmp_path):
        """The owned policy runs in the real sandbox with no inference token."""

        tmp_path.chmod(0o777)
        (tmp_path / "agent.py").write_text(
            Path("subnet/validator/environment_preflight_agent.py").read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )
        (tmp_path / "problems.jsonl").write_text(
            json.dumps(
                {
                    "problem_id": "positive",
                    "query": "Run the deterministic environment preflight.",
                    "environment": {
                        "binding": {
                            "session_id": "integration-session",
                            "tool_contract_version": "oro_task_tools_v2",
                        },
                        "action_groups": [
                            [
                                {"name": "search", "args": {"query": "shoes"}},
                                {
                                    "name": "message",
                                    "args": {"content": "Continue?"},
                                },
                            ]
                        ],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        command = build_sandbox_command(
            agent_host_path="",
            logs_host_path=str(tmp_path),
            problem_file_arg="/app/logs/problems.jsonl",
            output_path="/app/logs/output.jsonl",
            image=os.getenv(
                "INTEGRATION_SANDBOX_IMAGE", "ghcr.io/oro-ai/oro/sandbox:stable"
            ),
            network=os.getenv("INTEGRATION_SANDBOX_NETWORK", "sandbox-network"),
            agent_container_path="/app/logs/agent.py",
            max_workers=1,
            timeout=30,
        )

        result = subprocess.run(command, capture_output=True, text=True, timeout=60)

        assert result.returncode == 0, result.stderr
        row = json.loads((tmp_path / "output.jsonl").read_text(encoding="utf-8"))
        assert row["status"] == "SUCCESS"
        assert row["inference_total"] == 0
        environment_result = row["dialogue"][0]["environment_result"]
        assert environment_result["action_count"] == 2
        assert environment_result["user_message"]["content"]

    def test_hostile_agent_cannot_reach_validator_private_state(self, tmp_path):
        """A real sandbox sees only the documented session and proxy surface."""

        tmp_path.chmod(0o777)
        (tmp_path / "agent.py").write_text(
            Path("tests/integration/hostile_isolation_agent.py").read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )
        (tmp_path / "problems.jsonl").write_text(
            json.dumps(
                {
                    "problem_id": "hostile-isolation",
                    "query": "Probe the sandbox for validator-private state.",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (tmp_path / "environment_sessions.json").write_text(
            json.dumps(
                {
                    "schema_version": "oro.session_bootstrap.v1",
                    "sessions": [
                        {
                            "session_id": "integration-session",
                            "policy_view": {
                                "query": "Public shopper request",
                                "max_steps": 10,
                                "tool_contract_version": "oro_task_tools_v2",
                                "tools": [],
                                "max_calls_per_turn": 16,
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        command = build_sandbox_command(
            agent_host_path="",
            logs_host_path=str(tmp_path),
            problem_file_arg="/app/logs/problems.jsonl",
            output_path="/app/logs/output.jsonl",
            image=os.getenv(
                "INTEGRATION_SANDBOX_IMAGE", "ghcr.io/oro-ai/oro/sandbox:stable"
            ),
            network=os.getenv("INTEGRATION_SANDBOX_NETWORK", "sandbox-network"),
            agent_container_path="/app/logs/agent.py",
            max_workers=1,
            timeout=30,
        )

        result = subprocess.run(command, capture_output=True, text=True, timeout=60)

        assert result.returncode == 0, result.stderr
        row = json.loads((tmp_path / "output.jsonl").read_text(encoding="utf-8"))
        assert row["status"] == "SUCCESS"
        assert row["inference_total"] == 0
        report = row["dialogue"][0]["isolation_report"]
        assert report and all(report.values()), report

    def test_dns_resolution(self, sandbox_container):
        """Test DNS resolution (optional - may not have nslookup)."""
        result = exec_in_container(sandbox_container, ["nslookup", "proxy"], timeout=5)

        if result.returncode == 0:
            # DNS works
            assert True
        else:
            # nslookup may not be available, which is fine
            pytest.skip("nslookup not available (this is normal)")
