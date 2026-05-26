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
RUN pip install --no-cache-dir requests ujson fastapi 'uvicorn[standard]' pydantic

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

