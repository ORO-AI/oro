"""Small URL helpers shared across validator modules."""


def rewrite_localhost_url(url: str) -> str:
    """Rewrite localhost URLs to host.docker.internal for Docker connectivity."""
    if url.startswith("http://localhost:"):
        return url.replace("http://localhost:", "http://host.docker.internal:", 1)
    return url
