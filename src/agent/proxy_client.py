"""HTTP proxy client for ShoppingBench services."""

import json
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
DEFAULT_RATE_LIMIT_RETRY_DELAY = 5


class RequestLog:
    """Thread-safe log of all proxy HTTP calls.

    Appends one JSONL entry per request so data survives process kills.
    """

    def __init__(self, log_file: Optional[str] = None):
        self._lock = threading.Lock()
        self._log_file = log_file

    def record(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        status_code: Optional[int] = None,
        response_body: object = None,
        duration_ms: float = 0.0,
    ) -> None:
        if not self._log_file:
            return
        entry = {
            "method": method,
            "path": path,
            "timestamp": int(time.time() * 1000),
            "duration_ms": round(duration_ms, 1),
            "status_code": status_code,
        }
        if params:
            entry["params"] = {k: v for k, v in params.items() if v is not None}
        if json_data:
            if "/inference/" in path:
                # Omit large message arrays from inference requests
                entry["json_data"] = {k: v for k, v in json_data.items() if k != "messages"}
            else:
                entry["json_data"] = json_data
        if response_body is not None:
            body_str = json.dumps(response_body, default=str)
            if len(body_str) > 2000:
                entry["response_truncated"] = True
                entry["response_length"] = len(body_str)
                if "/search/find_product" in path and isinstance(response_body, list):
                    # Preserve product IDs so the reasoning judge can verify the
                    # agent's selections came from actual search results.
                    entry["result_product_ids"] = [
                        str(item["product_id"])
                        for item in response_body
                        if isinstance(item, dict) and "product_id" in item
                    ]
            else:
                entry["response"] = response_body
        with self._lock:
            try:
                with open(self._log_file, "a") as f:
                    f.write(json.dumps(entry, default=str) + "\n")
            except OSError:
                pass


class InferenceStats:
    """Thread-safe counter for inference call outcomes.

    Writes stats to a JSONL file after every call so that data is
    available even if the process is killed (e.g., Docker timeout).
    """

    def __init__(self, stats_file: str | None = None):
        self._lock = threading.Lock()
        self._success = 0
        self._failed = 0
        self._stats_file = stats_file

    def record_success(self):
        with self._lock:
            self._success += 1
            self._flush()

    def record_failure(self):
        with self._lock:
            self._failed += 1
            self._flush()

    def _flush(self) -> None:
        """Write current stats to the JSONL file (must hold _lock)."""
        if not self._stats_file:
            logger.debug("InferenceStats: no stats file configured, skipping flush")
            return
        try:
            problem_data = os.environ.get("PROBLEM_DATA", "{}")
            problem = json.loads(problem_data)
            problem_id = problem.get("problem_id") or problem.get("id", "unknown")
            entry = {
                "problem_id": str(problem_id),
                "inference_success": self._success,
                "inference_failed": self._failed,
                "inference_total": self._success + self._failed,
            }
            with open(self._stats_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except (OSError, json.JSONDecodeError):
            pass  # Best-effort; don't crash the agent


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
        rate_limit_retry_delay: float = DEFAULT_RATE_LIMIT_RETRY_DELAY,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the proxy client.

        Args:
            proxy_url: Base URL for the proxy (defaults to SANDBOX_PROXY_URL env var)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds (doubled each attempt)
            rate_limit_retry_delay: Base delay for 429 retries in seconds (doubled each
                attempt). Longer than retry_delay since rate limits need more time to clear.
            api_key: API key for inference requests (defaults to CHUTES_ACCESS_TOKEN env var).
                When set, inference POST requests include an Authorization header.
        """
        self.proxy_url = proxy_url or os.getenv("SANDBOX_PROXY_URL", "http://proxy:80")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit_retry_delay = rate_limit_retry_delay
        self.api_key = api_key or os.getenv("CHUTES_ACCESS_TOKEN")
        stats_file = os.environ.get(
            "INFERENCE_STATS_FILE", "/app/logs/inference_stats.jsonl"
        )
        self.inference_stats = InferenceStats(stats_file)
        request_log_file = os.environ.get("REQUEST_LOG_FILE")
        self.request_log = RequestLog(request_log_file)

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

        Rate-limited (429) responses use a separate retry counter with longer
        backoff (5s base) so transient capacity issues don't exhaust the normal
        retry budget.

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
                if response.status_code == 429:
                    logger.warning(
                        f"{operation_name} rate limited (429), "
                        f"retry {i + 1}/{self.max_retries}"
                    )
                else:
                    logger.warning(
                        f"{operation_name} returned status {response.status_code}, "
                        f"retry {i + 1}/{self.max_retries}"
                    )
            except requests.RequestException as e:
                logger.error(
                    f"{operation_name} error, retry {i + 1}/{self.max_retries}: {e}"
                )
                response = None

            if i < self.max_retries - 1:
                is_rate_limited = response is not None and response.status_code == 429
                base_delay = self.rate_limit_retry_delay if is_rate_limited else self.retry_delay
                delay = min(base_delay * (2 ** i), 10)
                time.sleep(delay)

        logger.error(f"Failed {operation_name} after {self.max_retries} retries")
        return None

    def get(self, path: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make a GET request to the proxy."""
        url = self._build_url(path, params)

        def make_request():
            return requests.get(url, timeout=self.timeout)

        t0 = time.monotonic()
        response = self._make_request_with_retries(make_request, f"GET {path}")
        duration_ms = (time.monotonic() - t0) * 1000
        result = response.json() if response else None

        self.request_log.record(
            method="GET",
            path=path,
            params=params,
            status_code=response.status_code if response else None,
            response_body=result,
            duration_ms=duration_ms,
        )
        return result

    def post(self, path: str, json_data: Optional[Dict] = None) -> Optional[Dict]:
        """Make a POST request to the proxy."""
        url = self._build_url(path)
        headers: Dict[str, str] = {}
        if self.api_key and "/inference/" in path:
            headers["Authorization"] = f"Bearer {self.api_key}"

        def make_request():
            response = requests.post(
                url, json=json_data, headers=headers, timeout=self.timeout
            )
            if response.status_code == 404:
                logger.error(f"Resource not found: {path}")
            return response

        t0 = time.monotonic()
        response = self._make_request_with_retries(make_request, f"POST {path}")
        duration_ms = (time.monotonic() - t0) * 1000

        if "/inference/" in path:
            if response and response.status_code == 200:
                self.inference_stats.record_success()
            else:
                self.inference_stats.record_failure()

        result = None
        if response and response.status_code == 200:
            result = response.json()

        self.request_log.record(
            method="POST",
            path=path,
            json_data=json_data,
            status_code=response.status_code if response else None,
            response_body=result,
            duration_ms=duration_ms,
        )
        return result
