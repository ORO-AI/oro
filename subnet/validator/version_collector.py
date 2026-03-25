"""Collect Docker image digests for validator stack services.

Uses `docker inspect` to get image digests (sha256) for running containers
and pulled images. Returns short-form digests for compact storage.
"""

import logging
import os
import socket
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

# Container names to inspect (running companion services)
SERVICE_CONTAINERS = [
    "shoppingbench-search-server",
    "shoppingbench-proxy",
]

# Sandbox image to inspect (pulled but not always running)
SANDBOX_IMAGE = os.environ.get(
    "SANDBOX_IMAGE", "ghcr.io/oro-ai/oro/sandbox:latest"
)

# Short name mapping for output keys
CONTAINER_KEY_MAP = {
    "shoppingbench-search-server": "search-server",
    "shoppingbench-proxy": "proxy",
}


def _run_docker_inspect(target: str, format_str: str) -> Optional[str]:
    """Run docker inspect and return trimmed output, or None on failure."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", format_str, target],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def _shorten_digest(digest: str) -> str:
    """Shorten a sha256:abc123... digest to sha256:abc123def4 (first 10 hex chars)."""
    if digest.startswith("sha256:") and len(digest) > 17:
        return digest[:17]
    return digest


def _get_container_image_digest(container_name: str) -> Optional[str]:
    """Get the image digest for a running container.

    Containers don't have RepoDigests directly — we get the image name
    from the container, then inspect the image for its registry digest.
    """
    # Get the image name from the container
    image = _run_docker_inspect(container_name, "{{.Config.Image}}")
    if image:
        digest = _get_image_digest(image)
        if digest:
            return digest

    # Fallback: use the container's image ID directly
    image_id = _run_docker_inspect(container_name, "{{.Image}}")
    if image_id:
        return _shorten_digest(image_id)

    return None


def _get_image_digest(image_name: str) -> Optional[str]:
    """Get the digest for a pulled image (not necessarily running)."""
    digest = _run_docker_inspect(
        image_name,
        "{{index .RepoDigests 0}}",
    )
    if digest and "@" in digest:
        return _shorten_digest(digest.split("@", 1)[1])

    image_id = _run_docker_inspect(image_name, "{{.Id}}")
    if image_id:
        return _shorten_digest(image_id)

    return None


def _get_validator_digest() -> Optional[str]:
    """Get the image digest of the validator's own container.

    Inside Docker, the hostname equals the container ID.
    """
    container_id = socket.gethostname()
    if not container_id:
        return None

    # Get the image used by this container, then its digest
    image = _run_docker_inspect(container_id, "{{.Config.Image}}")
    if image:
        return _get_image_digest(image)

    return None


def collect_service_versions() -> dict[str, str]:
    """Collect Docker image digests for all validator stack services.

    Returns:
        Dict mapping service name to short-form digest string.
        Example: {"search-server": "sha256:abc123def4", "proxy": "sha256:..."}
        Only includes services that could be inspected.
    """
    versions: dict[str, str] = {}

    # Companion services (running containers)
    for container in SERVICE_CONTAINERS:
        key = CONTAINER_KEY_MAP.get(container, container)
        digest = _get_container_image_digest(container)
        if digest:
            versions[key] = digest

    # Validator itself
    validator_digest = _get_validator_digest()
    if validator_digest:
        versions["validator"] = validator_digest

    # Sandbox image (pulled but not always running)
    sandbox_digest = _get_image_digest(SANDBOX_IMAGE)
    if sandbox_digest:
        versions["sandbox"] = sandbox_digest

    logger.info(f"Collected service versions: {len(versions)} services")
    logger.debug(f"Service versions: {versions}")

    return versions
