"""
Shared fixtures and utilities for integration tests.
"""

import os
import time
import subprocess
import pytest
import requests
from typing import List, Optional

# Container names
SEARCH_SERVER_CONTAINER = "shoppingbench-search-server"
PROXY_CONTAINER = "shoppingbench-proxy"
SANDBOX_CONTAINER = "shoppingbench-sandbox"

# Service URLs
SEARCH_SERVER_PORT = int(os.getenv("PORT", "5632"))
PROXY_PORT = int(os.getenv("PROXY_PORT", "8080"))


def is_container_running(container_name: str) -> bool:
    """Check if a Docker container is running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return container_name in result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_container_health(container_name: str) -> Optional[str]:
    """Get the health status of a container."""
    try:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                container_name,
                "--format",
                "{{.State.Health.Status}}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def wait_for_service_healthy(
    service_name: str, health_check_url: str, max_wait: int = 120, timeout: int = 5
) -> None:
    """Wait for a service to become healthy by checking its health endpoint."""
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            response = requests.get(health_check_url, timeout=timeout)
            if response.status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)

    # Show logs on failure
    try:
        logs = subprocess.run(
            ["docker", "logs", "--tail", "30", service_name],
            capture_output=True,
            text=True,
        )
        print(f"\n{service_name} logs:\n{logs.stdout}")
        if logs.stderr:
            print(f"\n{service_name} errors:\n{logs.stderr}")
    except subprocess.CalledProcessError:
        pass

    raise RuntimeError(
        f"{service_name} did not become healthy within {max_wait} seconds"
    )


def exec_in_container(
    container_name: str, command: List[str], timeout: int = 5
) -> subprocess.CompletedProcess:
    """Execute a command inside a Docker container."""
    return subprocess.run(
        ["docker", "exec", container_name] + command,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.fixture(scope="session")
def docker_services():
    """Fixture to ensure required Docker services are running."""
    # This is a marker fixture - individual test modules will check for specific services
    yield


@pytest.fixture(scope="session")
def search_server_ready(docker_services):
    """Fixture to ensure search server is running and healthy."""
    if not is_container_running(SEARCH_SERVER_CONTAINER):
        pytest.skip(
            f"Container {SEARCH_SERVER_CONTAINER} is not running. "
            "Please start it with: docker-compose up -d search-server"
        )

    wait_for_service_healthy(
        SEARCH_SERVER_CONTAINER, f"http://localhost:{SEARCH_SERVER_PORT}/health"
    )
    yield


@pytest.fixture(scope="session")
def proxy_ready(docker_services):
    """Fixture to ensure proxy is running and healthy."""
    if not is_container_running(PROXY_CONTAINER):
        pytest.skip(
            f"Container {PROXY_CONTAINER} is not running. "
            "Please start it with: docker-compose up -d proxy"
        )

    wait_for_service_healthy(PROXY_CONTAINER, f"http://localhost:{PROXY_PORT}/health")
    yield


@pytest.fixture(scope="session")
def sandbox_container(docker_services):
    """Fixture to ensure sandbox container is running."""
    if not is_container_running(SANDBOX_CONTAINER):
        pytest.skip(
            f"Container {SANDBOX_CONTAINER} is not running. "
            "Please start it with: docker-compose up -d sandbox"
        )

    yield SANDBOX_CONTAINER
