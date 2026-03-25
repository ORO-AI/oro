"""Test fixture: agent that raises an exception."""


def agent_main(problem):
    raise RuntimeError("agent crashed on purpose")
