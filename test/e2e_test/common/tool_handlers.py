"""Tool registration and handling utilities for E2E tests.

Provides helpers for registering stub tools with tracking and response logic.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List
from unittest.mock import MagicMock


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


def run_dev_handler_sync(event: dict, registry: Any) -> None:
    """Run dev_handler and block until the background crew completes.

    Used in E2E tests to synchronously execute the dev Slack handler,
    which normally dispatches work asynchronously. This ensures the
    handler completes before assertions are made.

    Args:
        event: Slack event dict with text, channel, thread_ts, ts
        registry: Agent registry containing the dev agent
    """
    from src.slack_app.handlers.dev import dev_handler

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        dev_handler(
            text=event.get("text", ""),
            event=event,
            say=MagicMock(),
            registry=registry,
            executor=executor,
        )
        executor.shutdown(wait=True, cancel_futures=False)
    finally:
        executor.shutdown(wait=False)
