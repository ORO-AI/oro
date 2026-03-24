"""
Agent interface definitions and helper functions for ShoppingBench sandbox execution.

This module provides:
- Type definitions for agent interfaces
- Helper functions for creating expected output
- Tool registry for dynamic tool registration

Usage in agent files:
    # Import helpers (use absolute import for compatibility)
    from src.agent.agent_interface import (
        Tool,
        execute_tool_call,
        create_dialogue_step,
    )

    # Register a custom tool (name defaults to function name)
    @Tool
    def my_custom_tool(param1: str) -> str:
        return f"Result: {param1}"

    # Or override the name
    @Tool("custom_name")
    def my_function(param1: str) -> str:
        return f"Result: {param1}"

    # Use in agent_main
    tool_result = execute_tool_call("my_custom_tool", {"param1": "value"})
    step = create_dialogue_step(
        think="...",
        tool_results=[tool_result],
        response="...",
        query="...",
        step=1
    )
"""

import json
import hashlib
import base64
import time
from typing import List, Dict, Any, Callable, TypedDict

# Agent must return a list of steps in the format expected by the evaluation framework.
AgentOutput = List[Dict]

# Tools can return any serializable type
ToolResult = Any


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
    """
    Register a tool function in the global registry.

    Args:
        name: Tool name (e.g., "find_product", "my_custom_tool")
        func: Callable function that implements the tool

    Example:
        # Direct registration
        register_tool("my_tool", my_tool_function)
    """
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
    """
    Get a registered tool by name.

    Args:
        name: Tool name

    Returns:
        The registered tool function

    Raises:
        ValueError: If tool is not registered
    """
    if name not in _TOOL_REGISTRY:
        available = ", ".join(sorted(_TOOL_REGISTRY.keys()))
        raise ValueError(
            f"Tool '{name}' is not registered. "
            f"Available tools: {available if available else '(none)'}"
        )

    return _TOOL_REGISTRY[name]


def list_tools() -> List[str]:
    """
    List all registered tool names.

    Returns:
        List of registered tool names
    """
    return list(_TOOL_REGISTRY.keys())


def generate_tool_call_id(name: str, parameters: dict, length: int = 8) -> str:
    """
    Generate a tool call ID from tool name and parameters.

    Args:
        name: Tool name
        parameters: Tool parameters dict
        length: Length of the generated ID (default: 8)

    Returns:
        Tool call ID string
    """
    tool_call_str = f"{name}\n{parameters}"
    hash_bytes = hashlib.md5(tool_call_str.encode("utf-8"), usedforsecurity=False).digest()
    base64_str = base64.urlsafe_b64encode(hash_bytes).decode("utf-8")
    clean_str = base64_str.replace("=", "").replace("+", "").replace("/", "")
    return clean_str[:length]


def format_content(think: str, tool_calls: List[dict], response: str) -> str:
    """
    Format content string with proper tags for format scoring.

    This matches the expected format from the evaluation framework:
    - Uses <think> tag in content string (message dict uses "think" key)
    - Uses <tool_call> with JSON array
    - Uses <response> tag

    Args:
        think: Reasoning/thinking text
        tool_calls: List of tool call dicts (with name, parameters, tool_call_id)
        response: Response text to user

    Returns:
        Formatted content string with XML-like tags
    """
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
    """
    Create a dialogue step in the format expected by the evaluation framework.

    Formats content properly for format scoring.

    Args:
        think: Reasoning/thinking text
        tool_results: List of ToolCallResult objects from execute_tool_call()
        response: Response text to user
        query: The original query
        step: Step number in the dialogue

    Returns:
        Dialogue step dict with completion structure matching evaluation framework
    """
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
    """
    Execute a tool call and return all information in a single result object.

    This helper function:
    - Generates tool_call_id using generate_tool_call_id()
    - Looks up the tool in the registry
    - Executes the tool with the provided parameters
    - Returns everything in a single ToolCallResult object

    Args:
        tool_name: Name of the tool to execute (must be registered)
        parameters: Parameters to pass to the tool

    Returns:
        ToolCallResult dict containing:
        - name: Name of the tool
        - parameters: Parameters that were passed
        - tool_call_id: Unique ID for this tool call
        - result: The result from tool execution (not included in output)

    Raises:
        ValueError: If tool is not registered

    Example:
        tool_result = execute_tool_call("find_product", {"q": "laptop"})
        # Access: tool_result["name"], tool_result["parameters"],
        #         tool_result["tool_call_id"], tool_result["result"]
    """
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
