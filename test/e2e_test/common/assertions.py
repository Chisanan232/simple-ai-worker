"""Assertion helpers for E2E tests.

Provides utilities for conditional assertions based on test mode
and common assertion patterns used across E2E tests.
"""

from typing import Any, List


def assert_stub_calls_in_stub_mode(
    e2e_settings: Any,
    condition: bool,
    message: str = "",
) -> None:
    """Assert a condition only in stub mode with real LLM.

    Args:
        e2e_settings: E2E settings object
        condition: The condition to assert
        message: Optional assertion message
    """
    if not e2e_settings.USE_TESTCONTAINERS and not e2e_settings.USE_FAKE_LLM:
        assert condition, message


def assert_stub_was_called(
    e2e_settings: Any,
    stub: Any,
    tool_name: str,
    message: str = "",
) -> None:
    """Assert that a stub tool was called, only in stub mode with real LLM.

    Args:
        e2e_settings: E2E settings object
        stub: The stub server object
        tool_name: Name of the tool to check
        message: Optional assertion message
    """
    if not e2e_settings.USE_TESTCONTAINERS and not e2e_settings.USE_FAKE_LLM:
        full_message = message or f"Expected stub to be called with tool '{tool_name}'"
        assert stub.was_called(tool_name), full_message


def assert_stub_calls_count(
    e2e_settings: Any,
    calls: List[Any],
    min_count: int = 0,
    max_count: int | None = None,
    message: str = "",
) -> None:
    """Assert the count of stub tool calls, only in stub mode with real LLM.

    Args:
        e2e_settings: E2E settings object
        calls: List of recorded calls
        min_count: Minimum expected call count
        max_count: Maximum expected call count (optional)
        message: Optional assertion message
    """
    if not e2e_settings.USE_TESTCONTAINERS and not e2e_settings.USE_FAKE_LLM:
        call_count = len(calls)
        if max_count is None:
            assert call_count >= min_count, message or f"Expected at least {min_count} calls, got {call_count}"
        else:
            assert min_count <= call_count <= max_count, message or (
                f"Expected {min_count}-{max_count} calls, got {call_count}"
            )


def assert_no_calls_in_stub_mode(
    e2e_settings: Any,
    calls: List[Any],
    tool_name: str,
    message: str = "",
) -> None:
    """Assert that no calls were made to a tool, only in stub mode with real LLM.

    Args:
        e2e_settings: E2E settings object
        calls: List of recorded calls
        tool_name: Name of the tool
        message: Optional assertion message
    """
    if not e2e_settings.USE_TESTCONTAINERS and not e2e_settings.USE_FAKE_LLM:
        assert len(calls) == 0, message or f"Expected no calls to '{tool_name}', got {len(calls)}"
