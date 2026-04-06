"""Local persistence for failed API requests (completions and progress reports)."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

from oro_sdk.models.terminal_status import TerminalStatus

from .backend_client import BackendClient, BackendError
from .models import CompletionRequest


class _TransientRetry(Exception):
    """Internal signal that a transient error occurred and entry should be retried."""


class LocalRetryQueue:
    """Persists failed completion requests for later retry.

    Stores pending completions in a JSON file. On process_pending(),
    attempts to complete each one. Successful completions are removed,
    failed ones remain for next attempt with incremented retry count.
    """

    DEFAULT_MAX_RETRIES = 10

    def __init__(
        self,
        backend_client: BackendClient,
        storage_path: Optional[Path] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.backend_client = backend_client
        self.storage_path = (
            storage_path or Path.home() / ".validator" / "retry_queue.json"
        )
        self.max_retries = max_retries
        self._ensure_storage_exists()

    def _ensure_storage_exists(self) -> None:
        """Create storage file if it doesn't exist."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            self._save({"pending": []})

    def _load(self) -> dict:
        """Load queue from storage."""
        try:
            with open(self.storage_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"pending": []}

    def _save(self, data: dict) -> None:
        """Save queue to storage."""
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def add(self, completion: CompletionRequest) -> None:
        """Add a failed completion request to the retry queue."""
        data = self._load()
        entry = completion.to_dict()
        entry["type"] = "completion"
        entry["added_at"] = datetime.now().isoformat()
        entry["retry_count"] = 0
        data["pending"].append(entry)
        self._save(data)
        logging.info(f"Added {completion.eval_run_id} to retry queue")


    def get_pending_count(self) -> int:
        """Get number of pending retries."""
        data = self._load()
        return len(data["pending"])

    def process_pending(self) -> None:
        """Attempt to process all pending completions."""
        data = self._load()
        remaining = []

        for entry in data["pending"]:
            entry_type = entry.get("type", "completion")

            try:
                if entry_type == "completion":
                    self._process_completion(entry)
                else:
                    logging.warning(f"Unknown entry type '{entry_type}', dropping")
            except _TransientRetry:
                entry["retry_count"] += 1
                if entry["retry_count"] >= self.max_retries:
                    logging.error(
                        f"Max retries ({self.max_retries}) exceeded for "
                        f"{entry_type} {entry.get('eval_run_id')}, dropping"
                    )
                else:
                    remaining.append(entry)

        data["pending"] = remaining
        self._save(data)

    def _process_completion(self, entry: dict) -> None:
        """Process a completion retry entry. Raises _TransientRetry on transient error."""
        kwargs = {
            "eval_run_id": UUID(entry["eval_run_id"]),
            "status": TerminalStatus(entry["terminal_status"]),
        }
        if entry.get("validator_score") is not None:
            kwargs["score"] = entry["validator_score"]
            kwargs["score_components"] = entry.get("score_components", {})
        if entry.get("results_s3_key"):
            kwargs["results_s3_key"] = entry["results_s3_key"]
        if entry.get("failure_reason"):
            kwargs["failure_reason"] = entry["failure_reason"]

        try:
            self.backend_client.complete_run(**kwargs)
            logging.info(f"Retry succeeded for completion {entry['eval_run_id']}")
        except BackendError as e:
            if e.is_run_already_complete:
                logging.info(f"Run {entry['eval_run_id']} already complete, removing")
            elif e.is_not_run_owner:
                logging.warning(f"Lost ownership of {entry['eval_run_id']}, removing")
            elif e.is_eval_run_not_found:
                logging.warning(f"Run {entry['eval_run_id']} not found, removing")
            elif e.is_transient:
                logging.warning(
                    f"Retry {entry['retry_count'] + 1}/{self.max_retries} failed "
                    f"for completion {entry['eval_run_id']}: {e}"
                )
                raise _TransientRetry()
            else:
                logging.error(
                    f"Non-retryable error for completion {entry['eval_run_id']}, dropping: {e}"
                )
        except Exception as e:
            logging.error(
                f"Unexpected error for completion {entry['eval_run_id']}, dropping: {e}"
            )

