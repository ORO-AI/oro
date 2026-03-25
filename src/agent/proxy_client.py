"""
Simple proxy client for making HTTP requests to ShoppingBench services via the proxy.

This module provides a minimal HTTP client that handles:
- URL building
- Retry logic
- Error handling
- GET and POST requests

All requests go through the proxy service for network isolation.
"""

import os
import logging
import threading
import time
from typing import Dict, Optional, Callable
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2


class InferenceStats:
    """Thread-safe counter for inference call outcomes."""

    def __init__(self):
        self._lock = threading.Lock()
        self._success = 0
        self._failed = 0

    def record_success(self):
        with self._lock:
            self._success += 1

    def record_failure(self):
        with self._lock:
            self._failed += 1

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "inference_success": self._success,
                "inference_failed": self._failed,
                "inference_total": self._success + self._failed,
            }


class ProxyClient:
    """
    Simple client for making HTTP requests to ShoppingBench services via the proxy.

    Handles URL building, retry logic, and error handling for both GET and POST requests.
    """

    def __init__(
        self,
        proxy_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the proxy client.

        Args:
            proxy_url: Base URL for the proxy (defaults to SANDBOX_PROXY_URL env var)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            api_key: API key for inference requests (defaults to CHUTES_ACCESS_TOKEN env var).
                When set, inference POST requests include an Authorization header.
        """
        self.proxy_url = proxy_url or os.getenv("SANDBOX_PROXY_URL", "http://proxy:80")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.api_key = api_key or os.getenv("CHUTES_ACCESS_TOKEN")
        self.inference_stats = InferenceStats()

    def _build_url(self, path: str, params: Optional[Dict] = None) -> str:
        """
        Build a complete URL for a proxy endpoint.

        Args:
            path: API path (e.g., "/search/find_product")
            params: Optional query parameters as a dictionary

        Returns:
            Complete URL string
        """
        base_url = self.proxy_url.rstrip("/")
        url = f"{base_url}{path}"

        if params:
            # Filter out None values and encode
            filtered_params = {k: v for k, v in params.items() if v is not None}
            if filtered_params:
                url += "?" + urlencode(filtered_params, doseq=True)

        return url

    def _make_request_with_retries(
        self,
        request_func: Callable[[], requests.Response],
        operation_name: str = "request",
    ) -> Optional[requests.Response]:
        """
        Make an HTTP request with retry logic.

        Args:
            request_func: Function that makes the HTTP request and returns a Response
            operation_name: Name of the operation for logging

        Returns:
            Response object if successful, None otherwise
        """
        for i in range(self.max_retries):
            try:
                response = request_func()
                if response.status_code == 200:
                    return response
                logger.warning(
                    f"{operation_name} returned status {response.status_code}, "
                    f"retry {i + 1}/{self.max_retries}"
                )
            except Exception as e:
                logger.error(
                    f"{operation_name} error, retry {i + 1}/{self.max_retries}: {e}"
                )
            if i < self.max_retries - 1:
                delay = self.retry_delay * (2**i)
                time.sleep(delay)

        logger.error(f"Failed {operation_name} after {self.max_retries} retries")
        return None

    def get(self, path: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make a GET request to the proxy.

        Args:
            path: API path
            params: Query parameters

        Returns:
            JSON response as dict, or None if request failed
        """
        url = self._build_url(path, params)

        def make_request():
            return requests.get(url, timeout=self.timeout)

        response = self._make_request_with_retries(make_request, f"GET {path}")
        if response:
            return response.json()
        return None

    def post(self, path: str, json_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make a POST request to the proxy.

        Inference requests (paths containing "/inference/") automatically include
        an Authorization header when an API key is available (CHUTES_ACCESS_TOKEN env var).

        Args:
            path: API path
            json_data: JSON data to send in request body

        Returns:
            JSON response as dict, or None if request failed
        """
        url = self._build_url(path)
        headers: Dict[str, str] = {}
        if self.api_key and "/inference/" in path:
            headers["Authorization"] = f"Bearer {self.api_key}"

        def make_request():
            response = requests.post(
                url, json=json_data, headers=headers, timeout=self.timeout
            )
            # Don't retry on 404 (e.g., model not found)
            if response.status_code == 404:
                logger.error(f"Resource not found: {path}")
            return response

        response = self._make_request_with_retries(make_request, f"POST {path}")
        # Track inference call outcomes
        if "/inference/" in path:
            if response and response.status_code == 200:
                self.inference_stats.record_success()
            else:
                self.inference_stats.record_failure()
        if response and response.status_code == 200:
            return response.json()
        return None


# Global instance for convenience
_default_client: Optional[ProxyClient] = None


def get_proxy_client() -> ProxyClient:
    """
    Get or create the default proxy client instance.

    Returns:
        Default ProxyClient instance
    """
    global _default_client
    if _default_client is None:
        _default_client = ProxyClient()
    return _default_client


def set_proxy_client(client: ProxyClient) -> None:
    """
    Set the default proxy client instance.

    Useful for testing or custom configuration.

    Args:
        client: ProxyClient instance to use as default
    """
    global _default_client
    _default_client = client
