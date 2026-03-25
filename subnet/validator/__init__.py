"""Validator package for ORO Backend API integration.

This package contains the Bittensor validator implementation that:
- Claims work from the Backend API
- Executes agent evaluations in Docker sandbox
- Reports progress and results back to the Backend
- Sets on-chain weights based on leaderboard

Modules:
    main: Validator class (main entry point)
    backend_client: BackendClient and BackendError
    heartbeat_manager: HeartbeatManager for lease maintenance
    progress_reporter: ProgressReporter for per-problem updates
    retry_queue: LocalRetryQueue for failed completions
    weight_setter: WeightSetterThread for on-chain weights
    backoff: ExponentialBackoff utility
    models: Local domain models
"""

# Lazy imports to avoid triggering bittensor at import time
# Use: from validator.backend_client import BackendClient
# Or:  from validator.main import Validator
