"""HTTP client for Backend API communication using oro-sdk.

This module wraps the oro-sdk to provide a simplified interface for the validator.
All types returned are SDK types directly - no local conversion layer.

Error Handling:
    All API errors are raised as BackendError with the SDK error model attached.
    Callers can inspect error.sdk_error to get typed error information, or use
    error.status_code for HTTP status-based handling.
"""

import logging
from typing import Any, Callable, Optional
from uuid import UUID

import httpx
import requests
from bittensor_wallet import Wallet
from oro_sdk import BittensorAuthClient, Client
from oro_sdk.api.public import get_top_agent
from oro_sdk.api.validator import (
    claim_work,
    complete_run,
    get_run_problems,
    heartbeat,
    presign_upload,
    update_progress,
)
from oro_sdk.models.claim_work_response import ClaimWorkResponse
from oro_sdk.models.complete_run_request import CompleteRunRequest
from oro_sdk.models.complete_run_response import CompleteRunResponse
from oro_sdk.models.at_capacity_error import AtCapacityError
from oro_sdk.models.eval_run_not_found_error import EvalRunNotFoundError
from oro_sdk.models.invalid_problem_id_error import InvalidProblemIdError
from oro_sdk.models.lease_expired_error import LeaseExpiredError
from oro_sdk.models.missing_score_error import MissingScoreError
from oro_sdk.models.not_run_owner_error import NotRunOwnerError
from oro_sdk.models.run_already_complete_error import RunAlreadyCompleteError
from oro_sdk.models.heartbeat_request import HeartbeatRequest as SdkHeartbeatRequest
from oro_sdk.models.heartbeat_response import HeartbeatResponse
from oro_sdk.models.presign_upload_request import PresignUploadRequest
from oro_sdk.models.presign_upload_response import PresignUploadResponse
from oro_sdk.models.problem_progress_update import ProblemProgressUpdate
from oro_sdk.models.progress_update_request import ProgressUpdateRequest
from oro_sdk.models.terminal_status import TerminalStatus
from oro_sdk.models.top_agent_response import TopAgentResponse
from oro_sdk.types import UNSET, Unset, Response
from oro_sdk import errors as sdk_errors


_HEARTBEAT_METRIC_KEYS = ("cpu_pct", "ram_pct", "disk_pct", "docker_container_count")


def _build_heartbeat_body(
    service_versions: Optional[dict[str, str]],
    resource_metrics: Optional[dict[str, Any]],
) -> Optional[SdkHeartbeatRequest]:
    """Build an SdkHeartbeatRequest from optional inputs, or None if both empty.

    Centralised so claim_work and heartbeat construct the body the same way.
    """
    body_kwargs: dict[str, Any] = {}
    if service_versions is not None:
        body_kwargs["service_versions"] = service_versions
    if resource_metrics:
        for key in _HEARTBEAT_METRIC_KEYS:
            value = resource_metrics.get(key)
            if value is not None:
                body_kwargs[key] = value
    if not body_kwargs:
        return None
    return SdkHeartbeatRequest(**body_kwargs)


class BackendError(Exception):
    """Backend API error with typed error details.

    Attributes:
        message: Human-readable error message.
        sdk_error: The SDK error model (typed if available).
        status_code: HTTP status code from the response.
        is_transient: True if this is a transient error that may be retried.
    """

    def __init__(
        self,
        message: str,
        sdk_error: Any = None,
        status_code: Optional[int] = None,
    ):
        super().__init__(message)
        self.message = message
        self.sdk_error = sdk_error
        self.status_code = status_code

    @property
    def is_transient(self) -> bool:
        """Return True if this error is transient and may be retried."""
        if self.status_code is not None and self.status_code >= 500:
            return True
        # Connection/timeout errors don't have status codes but are transient
        if self.status_code is None and self.sdk_error is None:
            return True
        return False

    @property
    def is_auth_error(self) -> bool:
        return self.status_code in (401, 403)

    @property
    def is_banned(self) -> bool:
        """Return True if this is a ban response (403 with 'banned' in message)."""
        return self.status_code == 403 and "banned" in self.message.lower()

    @property
    def is_conflict(self) -> bool:
        return self.status_code == 409

    @property
    def is_not_found(self) -> bool:
        """Return True if this is a not found error (404)."""
        return self.status_code == 404

    @property
    def error_code(self) -> Optional[str]:
        """Return the error_code from the SDK error if available."""
        if self.sdk_error is not None and hasattr(self.sdk_error, "error_code"):
            return self.sdk_error.error_code
        return None

    def is_error(self, error_type: type) -> bool:
        """Check if sdk_error is an instance of a specific SDK error type.

        Args:
            error_type: An SDK error model class (e.g., LeaseExpiredError).

        Returns:
            True if sdk_error is an instance of the given type.
        """
        return isinstance(self.sdk_error, error_type)

    @property
    def is_lease_expired(self) -> bool:
        """Return True if this is a lease expired error."""
        return self.is_error(LeaseExpiredError)

    @property
    def is_at_capacity(self) -> bool:
        """Return True if validator is at capacity."""
        return self.is_error(AtCapacityError)

    @property
    def is_not_run_owner(self) -> bool:
        return self.is_error(NotRunOwnerError)

    @property
    def is_run_already_complete(self) -> bool:
        return self.is_error(RunAlreadyCompleteError)

    @property
    def is_eval_run_not_found(self) -> bool:
        return self.is_error(EvalRunNotFoundError)

    @property
    def is_invalid_problem_id(self) -> bool:
        return self.is_error(InvalidProblemIdError)

    @property
    def is_missing_score(self) -> bool:
        return self.is_error(MissingScoreError)

    def __str__(self) -> str:
        return self.message


class BackendClient:
    """HTTP client for ORO Backend API using oro-sdk.

    This class wraps the oro-sdk and returns SDK types directly.
    All validator endpoints require authentication via the BittensorAuthClient
    which signs requests using the wallet's hotkey.
    """

    def __init__(self, base_url: str, wallet: Wallet, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.wallet = wallet
        self.timeout = timeout

        # Create authenticated client for validator endpoints
        self._auth_client = BittensorAuthClient(
            base_url=self.base_url,
            wallet=wallet,
            timeout=httpx.Timeout(timeout),
            raise_on_unexpected_status=False,
        )

        # Create public client for unauthenticated endpoints
        self._public_client = Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            raise_on_unexpected_status=False,
        )

    def _handle_response(
        self,
        response: Response[Any],
        operation: str,
        allow_204: bool = False,
    ) -> Any:
        """Handle API response, raising BackendError on failure.

        Args:
            response: The Response object from an SDK call.
            operation: Name of the operation for error messages.
            allow_204: If True, returns None for 204 status instead of raising.

        Returns:
            The parsed response on success.

        Raises:
            BackendError: For any non-success response.
        """
        status = response.status_code

        # Success cases
        if status == 200:
            # Check if we got an error response despite 200 status (shouldn't happen)
            # All error models have error_code attribute, success models don't
            if hasattr(response.parsed, "error_code"):
                raise BackendError(
                    f"{operation}: {response.parsed.detail}",
                    sdk_error=response.parsed,
                    status_code=status,
                )
            return response.parsed

        if status == 204:
            if allow_204:
                return None
            raise BackendError(
                f"{operation}: Unexpected 204 response",
                status_code=status,
            )

        # Error cases - extract detail from parsed error, fall back to body
        detail = None
        if response.parsed is not None and hasattr(response.parsed, "detail"):
            detail = response.parsed.detail
        elif hasattr(response, "content") and response.content:
            body = response.content[:500].decode("utf-8", errors="replace")
            detail = f"HTTP {status}: {body}"
        if not detail:
            detail = f"HTTP {status}"

        raise BackendError(
            f"{operation}: {detail}",
            sdk_error=response.parsed,
            status_code=status,
        )

    def _call_api(
        self,
        api_func: Callable[..., Response[Any]],
        operation: str,
        allow_204: bool = False,
        **kwargs,
    ) -> Any:
        """Call an SDK API function with unified error handling.

        Args:
            api_func: The SDK sync_detailed function to call.
            operation: Name of the operation for error messages.
            allow_204: If True, returns None for 204 status.
            **kwargs: Arguments to pass to the API function.

        Returns:
            The parsed response on success.

        Raises:
            BackendError: For any API error (transient or permanent).
        """
        try:
            response = api_func(**kwargs)
            return self._handle_response(response, operation, allow_204)

        except httpx.TimeoutException:
            raise BackendError(f"{operation}: Request timed out")
        except httpx.ConnectError as e:
            raise BackendError(f"{operation}: Connection error: {e}")
        except sdk_errors.UnexpectedStatus as e:
            body = ""
            if hasattr(e, "content") and e.content:
                body = f": {e.content[:500].decode('utf-8', errors='replace')}"
            raise BackendError(
                f"{operation}: Unexpected status {e.status_code}{body}",
                status_code=e.status_code,
            )
        except (KeyError, AttributeError, TypeError) as e:
            # SDK response parsing failed. Happens when the response body doesn't
            # match the expected schema (e.g., a proxy/WAF strips fields). The
            # original KeyError('loc') becomes "'loc'" via str(e) higher up,
            # which is unactionable — surface the parsing failure explicitly.
            logging.exception(
                "%s: SDK response parsing failed (%s). Check for proxy/WAF "
                "interference or SDK/Backend version mismatch.",
                operation, type(e).__name__,
            )
            raise BackendError(
                f"{operation}: Response parsing failed ({type(e).__name__}: {e}). "
                f"This usually indicates the Backend response was modified in "
                f"transit (proxy/WAF) or the validator SDK is out of sync."
            )

    def claim_work(
        self,
        service_versions: Optional[dict[str, str]] = None,
        resource_metrics: Optional[dict[str, Any]] = None,
    ) -> Optional[ClaimWorkResponse]:
        """Claim the next available work item.

        Args:
            service_versions: Optional Docker image digests for validator stack services.
            resource_metrics: Optional host metrics dict — keys cpu_pct, ram_pct,
                disk_pct, docker_container_count. Each is independently optional.

        Returns:
            ClaimWorkResponse if work is available, None if no work (204).

        Raises:
            BackendError: If the request fails.
                - is_transient=True for 5xx/timeout/connection errors
                - is_conflict=True if at capacity (409)
                - is_auth_error=True if authentication fails (401/403)
        """
        kwargs: dict[str, Any] = {
            "client": self._auth_client,
        }

        body = _build_heartbeat_body(service_versions, resource_metrics)
        if body is not None:
            kwargs["body"] = body

        return self._call_api(
            claim_work.sync_detailed,
            "claim_work",
            allow_204=True,
            **kwargs,
        )

    def heartbeat(
        self,
        eval_run_id: UUID,
        service_versions: Optional[dict[str, str]] = None,
        resource_metrics: Optional[dict[str, Any]] = None,
    ) -> HeartbeatResponse:
        """Send heartbeat to maintain lease.

        Args:
            eval_run_id: The evaluation run ID.
            service_versions: Optional Docker image digests for validator stack services.
            resource_metrics: Optional host metrics dict — same keys as claim_work.

        Returns:
            HeartbeatResponse with updated lease expiration.

        Raises:
            BackendError: If the request fails.
                - error_code="LEASE_EXPIRED" if lease has expired
                - error_code="NOT_RUN_OWNER" if not the owner
                - is_not_found=True if run doesn't exist (404)
        """
        kwargs: dict[str, Any] = {
            "eval_run_id": eval_run_id,
            "client": self._auth_client,
        }

        body = _build_heartbeat_body(service_versions, resource_metrics)
        if body is not None:
            kwargs["body"] = body

        return self._call_api(
            heartbeat.sync_detailed,
            "heartbeat",
            **kwargs,
        )

    def report_progress(
        self, eval_run_id: UUID, problems: list[ProblemProgressUpdate]
    ) -> None:
        """Report per-problem progress.

        Args:
            eval_run_id: The evaluation run ID.
            problems: List of problem progress updates.

        Raises:
            BackendError: If the request fails.
        """
        request_body = ProgressUpdateRequest(problems=problems)
        self._call_api(
            update_progress.sync_detailed,
            "report_progress",
            eval_run_id=eval_run_id,
            client=self._auth_client,
            body=request_body,
        )

    def get_presigned_upload_url(
        self,
        content_length: int,
        eval_run_id: UUID,
        problem_id: UUID,
        content_type: str = "application/gzip",
    ) -> PresignUploadResponse:
        """Get presigned URL for log upload.

        Args:
            content_length: Size of the content in bytes.
            eval_run_id: Evaluation run ID.
            problem_id: Optional problem ID. Uses nil UUID if not provided.
            content_type: MIME type (default: "application/gzip").

        Returns:
            PresignUploadResponse with upload URL and S3 key.

        Raises:
            BackendError: If the request fails.
        """
        request_body = PresignUploadRequest(
            content_length=content_length,
            eval_run_id=eval_run_id,
            problem_id=problem_id,
            content_type=content_type,
        )

        return self._call_api(
            presign_upload.sync_detailed,
            "get_presigned_upload_url",
            client=self._auth_client,
            body=request_body,
        )

    def complete_run(
        self,
        eval_run_id: UUID,
        status: TerminalStatus,
        score: Optional[float] = None,
        score_components: Optional[dict] = None,
        results_s3_key: str = "",
        failure_reason: Optional[str] = None,
        sandbox_metadata: Optional[dict] = None,
    ) -> CompleteRunResponse:
        """Complete an evaluation run.

        Args:
            eval_run_id: The evaluation run ID.
            status: Terminal status.
            score: Validator-computed score (required for SUCCESS, must be None otherwise).
            score_components: Breakdown of score components (required for SUCCESS).
            results_s3_key: S3 key where logs were uploaded.
            failure_reason: Reason for failure (sent for non-SUCCESS statuses).
            sandbox_metadata: Optional dict with sandbox execution metadata
                (exit_code, duration_seconds, stderr_tail).

        Returns:
            CompleteRunResponse with final status and eligibility.

        Raises:
            BackendError: If the request fails.
                - error_code="RUN_ALREADY_COMPLETE" if already complete
                - error_code="NOT_RUN_OWNER" if not the owner
        """
        # Build request based on status
        if status == TerminalStatus.SUCCESS:
            from oro_sdk.models.complete_run_request_score_components_type_0 import (
                CompleteRunRequestScoreComponentsType0,
            )

            sdk_score_components = CompleteRunRequestScoreComponentsType0.from_dict(
                score_components or {}
            )
            request_body = CompleteRunRequest(
                terminal_status=status,
                validator_score=score,
                score_components=sdk_score_components,
                results_s3_key=results_s3_key if results_s3_key else UNSET,
                sandbox_metadata=sandbox_metadata if sandbox_metadata else UNSET,
            )
        else:
            # For FAILED/TIMED_OUT, include failure reason
            request_body = CompleteRunRequest(
                terminal_status=status,
                results_s3_key=results_s3_key if results_s3_key else UNSET,
                failure_reason=failure_reason if failure_reason else UNSET,
                sandbox_metadata=sandbox_metadata if sandbox_metadata else UNSET,
            )

        return self._call_api(
            complete_run.sync_detailed,
            "complete_run",
            eval_run_id=eval_run_id,
            client=self._auth_client,
            body=request_body,
        )

    def get_top_miner(self) -> TopAgentResponse:
        """Get the current top miner for emissions.

        Returns:
            TopAgentResponse with top miner info.

        Raises:
            BackendError: If the request fails.
        """
        return self._call_api(
            get_top_agent.sync_detailed,
            "get_top_miner",
            client=self._public_client,
        )

    def upload_to_s3(self, presign: PresignUploadResponse, data: bytes) -> None:
        """Upload data to S3 using presigned URL.

        Args:
            presign: PresignUploadResponse from get_presigned_upload_url.
            data: Bytes to upload.

        Raises:
            BackendError: If S3 upload fails.
        """
        method = presign.method if presign.method is not UNSET else "PUT"
        headers = {"Content-Type": "application/gzip"}
        response = requests.request(
            method,
            presign.upload_url,
            headers=headers,
            data=data,
            timeout=60,
        )
        if response.status_code >= 400:
            raise BackendError(
                f"S3 upload failed: {response.status_code}",
                status_code=response.status_code,
            )

    def get_run_problems(self, eval_run_id: UUID) -> list[dict]:
        """Get problems for a specific evaluation run via SDK.

        For RACE runs, returns race-specific problems from the hidden bank.
        For QUALIFYING runs, returns the suite problems.

        Args:
            eval_run_id: The evaluation run ID.

        Returns:
            List of problem dicts with full metadata.

        Raises:
            BackendError: If the request fails.
        """
        response = self._call_api(
            get_run_problems.sync_detailed,
            "get_run_problems",
            eval_run_id=eval_run_id,
            client=self._auth_client,
        )
        problems = []
        for p in response.problems:
            metadata = p.metadata.to_dict() if p.metadata and not isinstance(p.metadata, Unset) else {}
            metadata["problem_id"] = str(p.problem_id)
            problems.append(metadata)
        return problems

