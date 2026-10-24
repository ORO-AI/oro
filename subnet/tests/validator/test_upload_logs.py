"""Guard against reintroducing the removed per-problem progress-retry path.

ORO-702 removed the progress retry queue ("no disk retry queue for progress
reports"); ``LocalRetryQueue`` must never grow an ``add_progress`` method again.
"""

from validator.retry_queue import LocalRetryQueue


def test_retry_queue_has_no_add_progress():
    """Guard against reintroducing the orphaned progress-retry path (ORO-702)."""
    assert not hasattr(LocalRetryQueue, "add_progress")
