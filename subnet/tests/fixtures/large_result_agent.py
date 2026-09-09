"""Agent fixture whose result exceeds a multiprocessing pipe's capacity."""


def agent_main(problem):
    return {"answer": "x" * 2_000_000}
