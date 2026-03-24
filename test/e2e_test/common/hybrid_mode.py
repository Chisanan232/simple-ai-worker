"""Hybrid mode utilities for E2E tests.

Provides helpers for handling both testcontainers and stub modes,
including URL selection and conditional tool registration logic.
"""

from typing import Any, Dict


def get_service_url(
    service_name: str,
    e2e_settings: Any,
    mcp_urls: Dict[str, str],
    stub: Any,
) -> str:
    """Get the appropriate URL for a service based on the test mode.

    Args:
        service_name: Name of the service (e.g., 'jira', 'clickup', 'github')
        e2e_settings: E2E settings object with USE_TESTCONTAINERS flag
        mcp_urls: Dictionary mapping service names to testcontainer URLs
        stub: Stub server object with url attribute

    Returns:
        The appropriate URL for the service based on test mode
    """
    if e2e_settings.USE_TESTCONTAINERS:
        return mcp_urls[service_name]
    else:
        return stub.url


def should_register_stub_tools(e2e_settings: Any) -> bool:
    """Check if stub tools should be registered.

    Tools are only registered in stub mode (not testcontainers mode).

    Args:
        e2e_settings: E2E settings object with USE_TESTCONTAINERS flag

    Returns:
        True if tools should be registered, False otherwise
    """
    return not e2e_settings.USE_TESTCONTAINERS


def should_assert_stub_calls(e2e_settings: Any) -> bool:
    """Check if stub call assertions should be performed.

    Assertions are only performed in stub mode with real LLM
    (not in testcontainers mode and not using fake LLM).

    Args:
        e2e_settings: E2E settings object with USE_TESTCONTAINERS and USE_FAKE_LLM flags

    Returns:
        True if stub assertions should be performed, False otherwise
    """
    return not e2e_settings.USE_TESTCONTAINERS and not e2e_settings.USE_FAKE_LLM
