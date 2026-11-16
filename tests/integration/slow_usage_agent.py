"""Make one (failing, network-less) inference call, then outlive the run."""

import time

from src.agent.proxy_client import ProxyClient


def agent_main(problem):
    ProxyClient(timeout=2, max_retries=0).post(
        "/inference/chat/completions", {"model": "probe", "messages": []}
    )
    time.sleep(120)
    return []
