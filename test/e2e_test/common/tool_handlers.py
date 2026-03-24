"""Tool registration and handling utilities for E2E tests.

Provides helpers for registering stub tools with tracking and response logic.
"""

from typing import Any, Callable, Dict, List


def register_tracking_tool(
    stub: Any,
    tool_name: str,
    calls_list: List[Dict[str, Any]],
    response: Any,
) -> None:
    """Register a stub tool that tracks calls and returns a fixed response.

    Args:
        stub: The stub server object
        tool_name: Name of the tool to register
        calls_list: List to append call arguments to
        response: Response to return from the tool
    """
    stub.register_tool(tool_name, lambda args: (calls_list.append(args) or response))


def register_conditional_tool(
    stub: Any,
    tool_name: str,
    calls_list: List[Dict[str, Any]],
    handler: Callable[[Dict[str, Any]], Any],
) -> None:
    """Register a stub tool with a custom handler that tracks calls.

    Args:
        stub: The stub server object
        tool_name: Name of the tool to register
        calls_list: List to append call arguments to
        handler: Function to handle the tool call and return response
    """
    def tracking_handler(args: Dict[str, Any]) -> Any:
        calls_list.append(args)
        return handler(args)

    stub.register_tool(tool_name, tracking_handler)


def make_tracking_lambda(
    calls_list: List[Dict[str, Any]],
    response: Any,
) -> Callable[[Dict[str, Any]], Any]:
    """Create a lambda function that tracks calls and returns a response.

    This is a convenience function for creating simple tracking tools.

    Args:
        calls_list: List to append call arguments to
        response: Response to return from the tool

    Returns:
        A lambda function suitable for stub.register_tool()
    """
    return lambda args: (calls_list.append(args) or response)


def make_conditional_lambda(
    calls_list: List[Dict[str, Any]],
    handler: Callable[[Dict[str, Any]], Any],
) -> Callable[[Dict[str, Any]], Any]:
    """Create a lambda function that tracks calls and uses a custom handler.

    Args:
        calls_list: List to append call arguments to
        handler: Function to handle the tool call

    Returns:
        A lambda function suitable for stub.register_tool()
    """
    return lambda args: (calls_list.append(args) or handler(args))
