"""Prometheus metric definitions for the validator process.

All metrics live on the default registry — the same one
`start_http_server` exposes from main.py.
"""

from prometheus_client import Counter, Gauge, Histogram

ACTIVE_RUNS = Gauge(
    "validator_active_runs",
    "Number of evaluation runs currently being executed by this validator",
)

HEARTBEAT_TOTAL = Counter(
    "validator_heartbeat_total",
    "Heartbeat send attempts, by outcome",
    labelnames=("result",),  # success | failure
)

CLAIM_WORK_SECONDS = Histogram(
    "validator_claim_work_seconds",
    "Latency of POST /v1/validator/work/claim",
)

CLAIM_WORK_TOTAL = Counter(
    "validator_claim_work_total",
    "Outcome of claim_work polls",
    labelnames=("result",),  # success | empty | error
)

SANDBOX_ACTIVE = Gauge(
    "validator_sandbox_active",
    "Sandbox containers currently running on this host",
)

# Buckets tuned to the sandbox timeout regime; default histogram buckets
# bottom out below 10s and lose resolution above that.
SANDBOX_DURATION_SECONDS = Histogram(
    "validator_sandbox_duration_seconds",
    "Wall-clock duration of a sandbox subprocess",
    buckets=(30, 60, 120, 180, 300, 600, 1200, 1800, 3600),
)
