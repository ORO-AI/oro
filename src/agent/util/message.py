try:
    import ujson as json
except ImportError:
    import json

from pydantic import BaseModel


USER_ROLES = ["user"]
ASSISTANT_ROLES = ["think", "tool_call", "obs", "response"]
OUTPUT_ROLES = ["think", "tool_call", "response"]


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
