"""Errors shared by the environment session registry and its HTTP bridge, and the validator's
caller mark. Dependency-free: the session service's slim test image imports it."""

import secrets

# Marks the validator's own proxy calls (model readings and decisions), which the
# agent sandbox may not make. Minted per process and never given to the sandbox: the
# agent holds the same provider key, so the key alone cannot tell the two apart.
VALIDATOR_CALLER_HEADER = "X-ORO-Validator"
VALIDATOR_CALLER_SECRET = secrets.token_urlsafe(32)


class HarnessError(RuntimeError):
    """A validator-runtime failure that must never become miner reward."""


class HarnessTimeoutError(HarnessError):
    """A tool call timed out and its session was quarantined."""


class HarnessExecutionError(HarnessError):
    """The environment runtime failed and its session was quarantined."""


class AgentInferenceBudgetError(HarnessExecutionError):
    """The miner-funded per-run inference key has exhausted its credits."""


class InvalidSessionError(HarnessError):
    """A call does not match an active, healthy session."""


__all__ = [
    "AgentInferenceBudgetError",
    "HarnessError",
    "HarnessExecutionError",
    "HarnessTimeoutError",
    "InvalidSessionError",
]
