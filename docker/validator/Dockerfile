FROM python:3.10-slim

# Install Docker CLI (for spawning sandbox containers via host socket)
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL https://download.docker.com/linux/static/stable/$(uname -m)/docker-27.5.1.tgz \
       | tar xz --strip-components=1 -C /usr/local/bin docker/docker \
    && apt-get purge -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Install uv for deterministic dependency resolution
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv

WORKDIR /app

# Install all dependencies from lockfile — fully deterministic builds.
# The lockfile pins every transitive dependency so fresh builds always
# get identical packages regardless of upstream releases.
COPY docker/validator/pyproject.toml docker/validator/uv.lock /app/
RUN uv sync --frozen --no-dev --no-install-project

# Pre-cache SentenceTransformer model (avoids runtime download)
ARG HF_TOKEN=""
RUN HF_HUB_ENABLE_HF_TRANSFER=0 HF_TOKEN=${HF_TOKEN} .venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')"

# Copy validator code
COPY subnet/ /app/subnet/

# Link workspace logs dir to the volume-mounted /app/logs so temp problem
# files written by the validator are visible on the host (needed for
# Docker-in-Docker sandbox volume mounts).
RUN mkdir -p /app/logs && ln -sf /app/logs /app/subnet/logs

# Copy evaluation/scoring code (used by test_runner and ProgressReporter)
COPY src/agent/ /app/src/agent/
COPY src/__init__.py /app/src/

# Copy data files (test problems for local testing)
COPY data/ /app/data/

ENV PYTHONPATH=/app

ENTRYPOINT [".venv/bin/python", "-m", "subnet.validator.main"]
