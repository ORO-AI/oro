FROM python:3.10-slim

# Install Docker CLI (for spawning sandbox containers via host socket)
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL https://download.docker.com/linux/static/stable/$(uname -m)/docker-27.5.1.tgz \
       | tar xz --strip-components=1 -C /usr/local/bin docker/docker \
    && apt-get purge -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first (avoids pulling ~2GB CUDA version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Layer 1: Heavy, stable dependencies (bittensor, sentence-transformers, etc.)
# This layer rarely changes and stays cached across most builds.
COPY docker/validator/requirements-base.txt /app/requirements-base.txt
RUN pip install --no-cache-dir -r /app/requirements-base.txt

# Layer 2: Pre-cache SentenceTransformer model (avoids runtime download)
ARG HF_TOKEN=""
RUN HF_HUB_ENABLE_HF_TRANSFER=0 HF_TOKEN=${HF_TOKEN} python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')"

# Layer 3: Fast-changing dependencies (oro-sdk updates frequently)
COPY docker/validator/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

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

ENTRYPOINT ["python", "-m", "subnet.validator.main"]
