FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# requests: agent HTTP calls
# ujson: ProblemScorer per-problem scoring
# fastapi/uvicorn/pydantic: sandbox_server.py HTTP mode
RUN python -m pip install --no-cache-dir --no-deps 'pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70'
RUN python -m pip install --no-cache-dir requests 'certifi>=2026.10.19' ujson fastapi 'uvicorn[standard]' pydantic

# Copy agent code and source structure
COPY src /app/src

# Create non-root sandbox user (UID 1000) and writable logs directory
RUN groupadd -g 1000 sandbox && \
    useradd -u 1000 -g sandbox -s /bin/sh sandbox && \
    mkdir -p /app/logs && \
    chown -R 1000:1000 /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV SANDBOX_PROXY_URL=http://proxy:80

# Default command runs the sandbox executor
# Can be overridden in docker-compose or when running
CMD ["python", "-m", "src.agent.run_sandbox", "--problem-file", "data/synthesize_product_test.jsonl"]

