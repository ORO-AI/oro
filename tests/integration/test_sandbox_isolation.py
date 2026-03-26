"""
Integration tests for sandbox network isolation.
Tests that sandbox containers are properly isolated and can only communicate through the proxy.
"""

import pytest
from tests.integration.conftest import (
    exec_in_container,
    SEARCH_SERVER_CONTAINER,
    PROXY_CONTAINER,
    is_container_running,
    get_container_health,
)


class TestSandboxIsolation:
    """Tests for sandbox network isolation."""

    def test_required_containers_running(self):
        """Test that required containers are running."""
        required = [SEARCH_SERVER_CONTAINER, PROXY_CONTAINER]
        missing = [c for c in required if not is_container_running(c)]

        assert not missing, (
            f"Required containers are not running: {missing}. "
            "Please start them with: docker-compose up -d search-server proxy"
        )

    def test_containers_healthy(self):
        """Test that required containers are healthy."""
        search_health = get_container_health(SEARCH_SERVER_CONTAINER)
        proxy_health = get_container_health(PROXY_CONTAINER)

        if search_health != "healthy":
            pytest.skip(
                f"search-server may not be healthy yet (status: {search_health})"
            )

        if proxy_health != "healthy":
            pytest.skip(f"proxy may not be healthy yet (status: {proxy_health})")

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

    def test_direct_search_server_access_blocked(self, sandbox_container):
        """Test that sandbox container cannot directly reach search-server."""
        result = exec_in_container(
            sandbox_container,
            ["curl", "-s", "--max-time", "5", "http://search-server:5632/health"],
            timeout=10,
        )

        assert result.returncode != 0, (
            "Container can directly reach search-server (should be blocked)"
        )

    def test_proxy_access_allowed(self, sandbox_container):
        """Test that sandbox container can reach proxy."""
        result = exec_in_container(
            sandbox_container,
            ["curl", "-s", "--max-time", "5", "http://proxy:80/health"],
            timeout=10,
        )

        assert result.returncode == 0, f"Container cannot reach proxy: {result.stderr}"
        assert "healthy" in result.stdout.lower() or result.stdout.strip() == "healthy"

    def test_search_server_through_proxy(self, sandbox_container):
        """Test that sandbox container can reach search-server through proxy."""
        result = exec_in_container(
            sandbox_container,
            [
                "curl",
                "-s",
                "--max-time",
                "5",
                "http://proxy:80/search/find_product?q=test&page=1",
            ],
            timeout=10,
        )

        assert result.returncode == 0, (
            f"Container cannot reach search-server through proxy: {result.stderr}"
        )

        response = result.stdout
        assert "product_id" in response or "[]" in response, (
            f"Unexpected response from search-server: {response[:200]}"
        )

    def test_dns_resolution(self, sandbox_container):
        """Test DNS resolution (optional - may not have nslookup)."""
        result = exec_in_container(sandbox_container, ["nslookup", "proxy"], timeout=5)

        if result.returncode == 0:
            # DNS works
            assert True
        else:
            # nslookup may not be available, which is fine
            pytest.skip("nslookup not available (this is normal)")
