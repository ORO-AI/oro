"""Prometheus metric definitions for the validator process.

Centralizes the metric objects so call sites just `from .metrics import M`
and increment / observe. Default registry — same one start_http_server
exposes from main.py.
"""

from prometheus_client import Counter, Gauge, Histogram

# In-flight evaluation runs claimed by this validator. Required by the
# auto-scale lifecycle hook: an instance is safe to terminate when this
# gauge has been 0 for the cooldown window.
ACTIVE_RUNS = Gauge(
    "validator_active_runs",
    "Number of evaluation runs currently being executed by this validator",
)

# Heartbeat outcome counter. Labels:
#   result = success | failure
HEARTBEAT_TOTAL = Counter(
    "validator_heartbeat_total",
    "Heartbeat send attempts, by outcome",
    labelnames=("result",),
)

# claim_work latency. Default histogram buckets work fine for HTTP-scale
# latencies (5ms..10s).
CLAIM_WORK_SECONDS = Histogram(
    "validator_claim_work_seconds",
    "Latency of POST /v1/validator/work/claim",
)

# claim_work outcome counter. Labels:
#   result = success | empty | error
CLAIM_WORK_TOTAL = Counter(
    "validator_claim_work_total",
    "Outcome of claim_work polls",
    labelnames=("result",),
)

# Number of sandbox containers currently spawned by this validator.
# Mirrors ACTIVE_RUNS today (1 sandbox per run) but kept separate so the
# semantics stay clean if we ever pipeline sandboxes per run.
SANDBOX_ACTIVE = Gauge(
    "validator_sandbox_active",
    "Sandbox containers currently running on this host",
)

# Wall-clock per sandbox subprocess. Sandbox runtime variance is the
# main signal we care about for race-completion projection.
SANDBOX_DURATION_SECONDS = Histogram(
    "validator_sandbox_duration_seconds",
    "Wall-clock duration of a sandbox subprocess",
    buckets=(30, 60, 120, 180, 300, 600, 1200, 1800, 3600),
)
