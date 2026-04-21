#!/bin/bash
set -e

# Wait for index files to be available
if [ ! -d "/app/indexes" ] || [ -z "$(ls -A /app/indexes)" ]; then
    echo "ERROR: Index directory is empty or does not exist!"
    exit 1
fi

echo "Index directory found with $(ls -1 /app/indexes | wc -l) files"

# Verify Java is available
if ! command -v java &> /dev/null; then
    echo "ERROR: Java not found in PATH" >&2
    exit 1
fi
echo "Java version: $(java -version 2>&1 | head -1)" >&2

# JAVA_OPTS default is set in the Dockerfile ENV.
# Override at runtime via docker-compose or -e JAVA_OPTS="..." if needed.

echo "JVM Options: $JAVA_OPTS" >&2

# Start the server (Python will handle signals for graceful shutdown)
exec python /app/src/search_engine/server.py


