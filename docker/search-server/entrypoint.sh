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

# Worker processes. The per-request cost is Lucene search + per-doc raw fetch,
# both via JNI, and pyjnius holds the GIL across JNI calls -- so a single
# process serializes requests under concurrency once its GIL-bound throughput
# ceiling is hit, leaving cores idle (badly on hosts with CPU steal). Multiple
# worker PROCESSES each get their own GIL + JVM and use the idle cores.
#
# Benefit requires SPARE cores: if workers exceed available cores the extra
# JVMs oversubscribe and it regresses. Default to half the cores (floor 2),
# leaving headroom for the sandboxes sharing the box; override with
# SEARCH_WORKERS. Set SEARCH_WORKERS=1 to fall back to a single process.
CORES="$(nproc)"
DEFAULT_WORKERS=$(( CORES / 2 ))
[ "$DEFAULT_WORKERS" -lt 2 ] && DEFAULT_WORKERS=2
WORKERS="${SEARCH_WORKERS:-$DEFAULT_WORKERS}"
THREADS="${SEARCH_THREADS:-4}"
PORT="${PORT:-5632}"
echo "Starting search-server: ${WORKERS} gunicorn worker(s) x ${THREADS} threads on :${PORT} (cores=${CORES})" >&2
echo "JVM options (per worker): ${_JAVA_OPTIONS}" >&2

# gthread (not sync): the WORKERS processes give the real parallelism (each its
# own GIL + JVM, using otherwise-idle cores), while a few THREADS per worker
# keep the worker from being single-request-blocked -- so trivial requests like
# /health always find a slot even while searches are in flight (a sync worker
# would queue /health behind a multi-second search and flap the healthcheck
# unhealthy under the target burst). Threads also overlap the JNI Lucene work,
# which releases the GIL.
#
# No --preload: the embedded JVM does not survive fork(); each worker must
# initialize its own LuceneSearcher/JVM after forking. The mmap'd index is
# shared across workers via the OS page cache (~1x index RAM). No access log on
# the hot search path.
exec gunicorn \
    --workers "${WORKERS}" \
    --worker-class gthread \
    --threads "${THREADS}" \
    --bind "0.0.0.0:${PORT}" \
    --chdir /app \
    --timeout 120 \
    --graceful-timeout 30 \
    --error-logfile - \
    src.search_engine.server:app


