"""Tool registration, execution, and dialogue step creation for ShoppingBench agents."""

import json
import hashlib
import base64
import time
from typing import List, Dict, Any, Callable, TypedDict

class ToolCallResult(TypedDict):
    """
    Complete result from executing a tool call.

    Contains all information needed to create dialogue steps.
    Field names match the evaluation framework format.

    - name: Name of the tool that was called
    - parameters: Parameters that were passed to the tool
    - tool_call_id: Unique ID for this tool call
    - result: The result from tool execution (not included in output)
    """

    name: str
    parameters: Dict[str, Any]
    tool_call_id: str
    result: Any


# Global tool registry: maps tool names to callable functions
_TOOL_REGISTRY: Dict[str, Callable] = {}


def register_tool(name: str, func: Callable) -> None:
    """Register a callable as a named tool."""
    _TOOL_REGISTRY[name] = func


def Tool(name: str = None):
    """
    Decorator to register a tool function.

    If no name is provided, the function's name is used as the tool name.

    Args:
        name: Optional tool name. If not provided, uses the function's __name__

    Example:
        # Use function name as tool name
        @Tool
        def my_custom_tool(param1: str, param2: int) -> str:
            return f"Result: {param1} {param2}"

        # Override with custom name
        @Tool("custom_name")
        def my_function(query: str) -> List[Dict]:
            return []
    """

    def decorator(func: Callable) -> Callable:
        # Use provided name or default to function name
        tool_name = name if name is not None else func.__name__
        register_tool(tool_name, func)
        return func

    # Handle both @Tool and @Tool("name") usage
    if callable(name):
        # Called as @Tool without parentheses
        func = name
        tool_name = func.__name__
        register_tool(tool_name, func)
        return func
    else:
        # Called as @Tool() or @Tool("name")
        return decorator


def get_tool(name: str) -> Callable:
    """Get a registered tool by name. Raises KeyError if not found."""
    if name not in _TOOL_REGISTRY:
        available = ", ".join(sorted(_TOOL_REGISTRY.keys()))
        raise ValueError(
            f"Tool '{name}' is not registered. "
            f"Available tools: {available if available else '(none)'}"
        )

    return _TOOL_REGISTRY[name]


def generate_tool_call_id(name: str, parameters: dict, length: int = 8) -> str:
    """Generate a deterministic ID from tool name + params."""
    tool_call_str = f"{name}\n{parameters}"
    hash_bytes = hashlib.md5(tool_call_str.encode("utf-8"), usedforsecurity=False).digest()
    base64_str = base64.urlsafe_b64encode(hash_bytes).decode("utf-8")
    clean_str = base64_str.replace("=", "").replace("+", "").replace("/", "")
    return clean_str[:length]


def format_content(think: str, tool_calls: List[dict], response: str) -> str:
    """Format content string with proper tags for format scoring."""
    parts = []
    if think:
        parts.append(f"<think>{think}</think>")
    if tool_calls:
        # Format tool_calls as JSON array (without tool_call_id in content, but it's in message dict)
        tool_calls_for_content = [
            {"name": tc["name"], "parameters": tc["parameters"]} for tc in tool_calls
        ]
        tool_calls_json = json.dumps(tool_calls_for_content)
        parts.append(f"<tool_call>{tool_calls_json}</tool_call>")
    if response:
        parts.append(f"<response>{response}</response>")
    return "\n".join(parts)


def create_dialogue_step(
    think: str, tool_results: List[ToolCallResult], response: str, query: str, step: int
) -> dict:
    """Create a dialogue step dict for the evaluation framework."""
    # Generate content using formatted tags
    content = format_content(think, tool_results, response)

    # Create message dict (same structure as Message.to_dict())
    message_dict = {}
    if think:
        message_dict["think"] = think
    if tool_results:
        message_dict["tool_call"] = tool_results
    if response:
        message_dict["response"] = response

    # Create step structure matching react_loop() output
    step_dict = {
        "completion": {
            "reasoning_content": "",
            "content": content,
            "message": message_dict,
        },
        "extra_info": {
            "step": step,
            "query": query,
            "timestamp": int(time.time() * 1000),
        },
    }

    return step_dict


def execute_tool_call(tool_name: str, parameters: dict) -> ToolCallResult:
    """Execute a registered tool and return the result with metadata."""
    # Generate tool_call_id
    tool_call_id = generate_tool_call_id(tool_name, parameters)

    # Get and execute tool from registry
    tool_func = get_tool(tool_name)
    result = tool_func(**parameters)

    return {
        "name": tool_name,
        "parameters": parameters,
        "tool_call_id": tool_call_id,
        "result": result,
    }
