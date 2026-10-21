"""Reference environment agent: inference request shape and text-answer retry."""

import copy

import pytest

from src.agent import environment_agent

TOOL = {
    "type": "function",
    "function": {
        "name": "message_shopper",
        "description": "Send a message to the shopper.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
}
BINDING = {"session_id": "s-1", "token": "t-1"}


def _problem() -> dict:
    return {
        "environment": {
            "binding": dict(BINDING),
            "policy_view": {
                "query": "Help the shopper.",
                "opening": None,
                "tools": [TOOL],
                "max_steps": 5,
                "finished": False,
            },
        }
    }


def _text(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _tool_call(text: str) -> dict:
    call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "message_shopper", "arguments": f'{{"text": "{text}"}}'},
    }
    return {
        "choices": [
            {"message": {"role": "assistant", "content": None, "tool_calls": [call]}}
        ]
    }


@pytest.fixture
def fake_post(monkeypatch):
    requests: list[tuple[str, dict]] = []
    inference_replies: list[dict] = []
    environment_replies: list[dict] = []

    def post(path, json_data=None, **_):
        requests.append((path, copy.deepcopy(json_data)))
        if path == "/inference/chat/completions":
            return inference_replies.pop(0)
        assert path == "/environment/call"
        return environment_replies.pop(0)

    monkeypatch.setattr(environment_agent._proxy, "post", post)
    return requests, inference_replies, environment_replies


def test_inference_uses_auto_tool_choice_and_retries_text(fake_post):
    requests, inference_replies, environment_replies = fake_post
    inference_replies += [_text("Let me think."), _tool_call("hi")]
    environment_replies.append({"observation": {"reply": "thanks"}, "finished": True})

    dialogue = environment_agent.agent_main(_problem())

    inference = [
        body for path, body in requests if path == "/inference/chat/completions"
    ]
    environment = [body for path, body in requests if path == "/environment/call"]
    assert len(inference) == 2
    assert all(body["tool_choice"] == "auto" for body in inference)
    assert "required" not in {body["tool_choice"] for body in inference}
    assert inference[1]["messages"][-1]["role"] == "user"
    assert environment == [
        {
            **BINDING,
            "step": 1,
            "action": {"name": "message_shopper", "args": {"text": "hi"}},
        }
    ]
    assert dialogue[0] == {"role": "assistant", "content": "Let me think."}
    assert dialogue[1]["environment_result"]["finished"] is True


def test_repeated_text_answers_raise_without_environment_call(fake_post):
    requests, inference_replies, _ = fake_post
    inference_replies += [_text("no tool") for _ in range(5)]

    with pytest.raises(RuntimeError, match="no tool call"):
        environment_agent.agent_main(_problem())

    paths = [path for path, _ in requests]
    assert "/environment/call" not in paths
    assert paths.count("/inference/chat/completions") == 3
