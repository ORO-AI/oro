import hashlib
import base64
try:
    import ujson as json
except ImportError:
    import json

from pydantic import BaseModel


USER_ROLES = ["user"]
ASSISTANT_ROLES = ["think", "tool_call", "obs", "response"]
OUTPUT_ROLES = ["think", "tool_call", "response"]


def generate_tool_call_id(name: str, parameters: dict, length: int = 8) -> str:
    # NOTE: Duplicated from src.agent.agent_interface.generate_tool_call_id
    # because this module is imported via bare `from util.message import ...`
    # inside the sandbox (sys.path hack), so cross-package imports are fragile.
    tool_call_str = f"{name}\n{parameters}"
    hash_bytes = hashlib.md5(tool_call_str.encode("utf-8"), usedforsecurity=False).digest()

    base64_str = base64.urlsafe_b64encode(hash_bytes).decode("utf-8")
    clean_str = base64_str.replace("=", "").replace("+", "").replace("/", "")

    return clean_str[:length]


class Message(BaseModel):
    user: str = ""
    think: str = ""
    tool_call: list[dict] = []
    obs: list[dict] = []
    response: str = ""

    def to_dict(self, roles: list[str] = []) -> dict:
        if not roles:
            roles = USER_ROLES + ASSISTANT_ROLES

        result = dict()
        for role in roles:
            if hasattr(self, role) and getattr(self, role):
                result[role] = getattr(self, role)
        return result

    def to_string(self, roles: list[str] = []) -> str:
        if not roles:
            roles = USER_ROLES + ASSISTANT_ROLES

        current = []
        for role in roles:
            if hasattr(self, role) and getattr(self, role):
                content = getattr(self, role)
                if isinstance(content, (dict, list)):
                    content = json.dumps(content)
                elif isinstance(content, str):
                    pass
                else:
                    raise Exception(
                        f"Invalid content type: {type(content)}, content: {content}"
                    )
                current.append(f"<{role}>{content}</{role}>")
        return "\n".join(current)

    @classmethod
    def from_dict(clf, message: dict):
        return clf(**message)
