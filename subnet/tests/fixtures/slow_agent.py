"""Test fixture: agent that sleeps forever (simulates a hung miner agent)."""

import time


def agent_main(problem):
    time.sleep(9999)
    return {"answer": "never reached"}
