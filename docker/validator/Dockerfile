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

# Pre-cache the canonical sentence model (avoids runtime download).
ARG HF_TOKEN=""
RUN HF_HUB_ENABLE_HF_TRANSFER=0 HF_TOKEN=${HF_TOKEN} .venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

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

# bittensor 10.3+ ships CLI parsing OFF by default
# (`bittensor.core.config.no_parse_cli` reads `BT_NO_PARSE_CLI_ARGS` and
# defaults it to "true"), which makes `Config(parser)` early-return
# without applying our `--wallet.name`/`--wallet.hotkey`/`--netuid`/
# `--subtensor.network` command-line arguments. Without those the
# validator falls back to class defaults `netuid=None, wallet=default,
# network=finney` and crashes at metagraph sync inside the runtime
# panic `NeuronInfoRuntimeApi::get_neurons_lite(None)`.
#
# Baked into the image (not just docker-compose.yml) so operators
# whose compose file hasn't been re-pulled since PR #229 don't crash
# on Watchtower auto-update.
ENV BT_NO_PARSE_CLI_ARGS=false

ENTRYPOINT [".venv/bin/python", "-m", "subnet.validator.main"]
