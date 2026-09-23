"""Fixture that records inference usage from inside the agent process."""

from src.agent.proxy_client import ProxyClient


_client = ProxyClient()


def agent_main(_problem):
    _client.inference_stats.record_success(
        {
            "cost": 0.25,
            "prompt_tokens": 10,
            "completion_tokens": 2,
        }
    )
    return {"answer": "ok"}
