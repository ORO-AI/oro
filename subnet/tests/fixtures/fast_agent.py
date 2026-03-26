"""Test fixture: agent that returns immediately."""


def agent_main(problem):
    return {"answer": "hello", "query": problem.get("query", "")}
