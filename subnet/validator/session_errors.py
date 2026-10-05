"""Errors shared by the environment session registry and its HTTP bridge."""


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


class HarnessResponseError(HarnessError):
    """The environment runtime or simulator returned a malformed response.

    Raised only by session_registry's own named response-shape guards
    (e.g. a step result or simulator decision that isn't the object shape
    the contract requires) -- never a bare ``TypeError``, so the HTTP
    bridge can map exactly this failure mode to ``environment_error``
    without also catching an unrelated internal ``TypeError`` from
    elsewhere in the registry and misreporting it the same way.
    """


__all__ = [
    "AgentInferenceBudgetError",
    "HarnessError",
    "HarnessExecutionError",
    "HarnessResponseError",
    "HarnessTimeoutError",
    "InvalidSessionError",
]
