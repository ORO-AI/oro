"""Errors shared by the environment session registry and its HTTP bridge."""


class HarnessError(RuntimeError):
    """A validator-runtime failure that must never become miner reward."""


class HarnessTimeoutError(HarnessError):
    """A tool call timed out and its session was quarantined."""


class HarnessExecutionError(HarnessError):
    """The environment runtime failed and its session was quarantined."""


class InvalidSessionError(HarnessError):
    """A call does not match an active, healthy session."""


__all__ = [
    "HarnessError",
    "HarnessExecutionError",
    "HarnessTimeoutError",
    "InvalidSessionError",
]
