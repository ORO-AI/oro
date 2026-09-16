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
# JVMs oversubscribe and it regresses (measured: WORKERS=cores/2 hits ~2100
# rps clean at c=96 vs WORKERS=cores at ~1400 rps). Default to half the cores
# (floor 2), leaving headroom for the sandboxes sharing the box; override
# with SEARCH_WORKERS. Set SEARCH_WORKERS=1 to fall back to a single process.
CORES="$(nproc)"
DEFAULT_WORKERS=$(( CORES / 2 ))
[ "$DEFAULT_WORKERS" -lt 2 ] && DEFAULT_WORKERS=2
WORKERS="${SEARCH_WORKERS:-$DEFAULT_WORKERS}"
PORT="${PORT:-5632}"
echo "Starting search-server: ${WORKERS} sync gunicorn worker(s) on :${PORT} (cores=${CORES})" >&2
echo "JVM options (per worker): ${_JAVA_OPTIONS}" >&2
if [ -n "${SEARCH_THREADS:-}" ]; then
    echo "warning: SEARCH_THREADS=${SEARCH_THREADS} ignored — server now uses sync workers (see ORO-2244)" >&2
fi

# --worker-class sync (was gthread --threads N): pyserini's LuceneSearcher +
# the pyjnius bridge are NOT thread-safe. Sharing one embedded JVM across
# multiple gthread threads in a single worker races the pyjnius method-ID +
# class-lookup caches -- observed symptoms include
# `java.lang.NullPointerException`, `IncompatibleClassChangeError: TotalHits
# does not implement List`, `NoSuchMethodError SortField.search(...)`, and
# `OutOfMemoryError: String length out of range` under concurrent
# /internal/catalog/{search,filter} load (ORO-2244).
#
# Sync workers give exactly one Python thread per JVM, matching
# pyserini/pyjnius's implicit single-thread assumption. WORKERS processes
# still provide the actual parallelism -- each with its own JVM. Sync
# workers do serialize /health behind a slow search within a single worker,
# but as long as WORKERS > 1 the gunicorn accept loop picks an idle worker
# for the healthcheck so probes still succeed under target load. Perf-tested
# vs gthread threads=4:
#   c=48, r=1000: sync clean p99 63ms / 1541 rps (no errors) vs
#                 gthread threads=4 p99 588ms / 1745 rps (23% errors + resets)
#   c=96, r=2000: sync clean p99 80ms / 2158 rps (no errors)
# Cold sequential path costs +~0.5ms vs gthread threads=4.
#
# No --preload: the embedded JVM does not survive fork(); each worker must
# initialize its own LuceneSearcher/JVM after forking. The mmap'd index is
# shared across workers via the OS page cache (~1x index RAM). No access log
# on the hot search path.
exec gunicorn \
    --workers "${WORKERS}" \
    --worker-class sync \
    --bind "0.0.0.0:${PORT}" \
    --chdir /app \
    --timeout 120 \
    --graceful-timeout 30 \
    --error-logfile - \
    src.search_engine.server:app
