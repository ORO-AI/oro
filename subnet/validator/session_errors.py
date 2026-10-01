"""Errors shared by the environment session registry and its HTTP bridge."""


class HarnessError(RuntimeError):
    """A validator-runtime failure that must never become miner reward."""


class HarnessTimeoutError(HarnessError):
    """A tool call timed out and its session was quarantined."""


class HarnessExecutionError(HarnessError):
    """The environment runtime failed and its session was quarantined."""


class AgentFaultError(HarnessExecutionError):
    """The agent's own input failed its session, which scores as an agent error."""


class AgentInferenceBudgetError(AgentFaultError):
    """The miner-funded per-run inference key has exhausted its credits."""


class InvalidSessionError(HarnessError):
    """A call does not match an active, healthy session."""


__all__ = [
    "AgentFaultError",
    "AgentInferenceBudgetError",
    "HarnessError",
    "HarnessExecutionError",
    "HarnessTimeoutError",
    "InvalidSessionError",
]
