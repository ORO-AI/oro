"""Watchtower update poller + sandbox image refresh.

Triggered between evaluation cycles. All errors swallowed — never crashes the
main loop. After Watchtower restarts services, waits for proxy /health (which
transitively covers search-server) before returning.
"""

from __future__ import annotations

import os
import subprocess
import time

import requests
from bittensor.utils.btlogging import logging

from subnet.sandbox import SANDBOX_IMAGE


WATCHTOWER_URL = os.environ.get("ORO_WATCHTOWER_URL", "http://watchtower:8080")
WATCHTOWER_TOKEN = os.environ.get("WATCHTOWER_TOKEN", "oro-watchtower-token")
AUTO_UPDATE_ENABLED = os.environ.get("ORO_AUTO_UPDATE", "true").lower() in (
    "true",
    "1",
    "yes",
)


def check_for_updates() -> bool:
    """Trigger Watchtower + pull sandbox image. Returns when proxy is healthy.

    Returns True if the update cycle ran (caller should re-collect service
    versions), False when auto-update is disabled.
    """
    if not AUTO_UPDATE_ENABLED:
        return False

    try:
        logging.info("Triggering Watchtower update check...")
        resp = requests.get(
            f"{WATCHTOWER_URL}/v1/update",
            headers={"Authorization": f"Bearer {WATCHTOWER_TOKEN}"},
            timeout=300,
        )
        if resp.ok:
            logging.info(
                f"Watchtower update check completed (status {resp.status_code})"
            )
        else:
            logging.warning(f"Watchtower update check returned {resp.status_code}")
    except requests.exceptions.ConnectionError:
        logging.debug("Watchtower not reachable, skipping update check")
    except Exception as e:
        logging.warning(f"Watchtower update check failed: {e}")

    for _ in range(30):
        try:
            if requests.get("http://proxy:80/health", timeout=5).ok:
                break
        except Exception:
            pass
        time.sleep(10)

    try:
        result = subprocess.run(
            ["docker", "pull", SANDBOX_IMAGE],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            logging.warning(f"Sandbox image pull failed: {result.stderr.strip()}")
    except (subprocess.SubprocessError, OSError, FileNotFoundError) as e:
        logging.warning(f"Sandbox image pull failed: {e}")

    return True
