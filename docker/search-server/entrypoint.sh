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

# Set JVM memory options for pyserini/Lucene (default: 8GB heap, adjust as needed)
# Use JAVA_OPTS environment variable if set, otherwise use defaults
export JAVA_OPTS="${JAVA_OPTS:--Xmx8g -Xms4g}"

echo "JVM Options: $JAVA_OPTS" >&2

# Start the server (Python will handle signals for graceful shutdown)
exec python /app/src/search_engine/server.py


