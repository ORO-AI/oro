# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Install Docker CLI (for spawning sandbox containers via host socket)
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL https://download.docker.com/linux/static/stable/$(uname -m)/docker-27.5.1.tgz \
       | tar xz --strip-components=1 -C /usr/local/bin docker/docker \
    && apt-get purge -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Install uv for deterministic dependency resolution
COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /usr/local/bin/uv

WORKDIR /app

# Install all dependencies from the shared lockfile. The mutually exclusive
# profiles select the hosted v6 runtime or miner-local v5 runtime without
# duplicating the rest of the dependency graph.
COPY docker/validator/pyproject.toml docker/validator/uv.lock /app/
ARG RUNTIME_PROFILE=validator
RUN uv pip install --system --only-binary :all: --no-deps 'pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70' 'setuptools>=84.1.0' && \
    uv sync --frozen --no-install-project --no-default-groups --group ${RUNTIME_PROFILE}

# Pre-cache the canonical sentence model (avoids runtime download).
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN,required=false \
    HF_HUB_ENABLE_HF_TRANSFER=0 .venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Copy validator code
COPY subnet/ /app/subnet/

# Link workspace logs dir to the volume-mounted /app/logs so temp problem
# files written by the validator are visible on the host (needed for
# Docker-in-Docker sandbox volume mounts).
RUN mkdir -p /app/logs && ln -sf /app/logs /app/subnet/logs

# Copy evaluation/scoring code used by the validator.
COPY src/agent/ /app/src/agent/
COPY src/__init__.py /app/src/

# Copy data files (test problems for local testing)
COPY data/ /app/data/
COPY trajectory-viewer/ /app/trajectory-viewer/

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
