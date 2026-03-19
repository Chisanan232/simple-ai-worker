"""Planner Agent E2E test fixtures — Layer 3."""
from __future__ import annotations

from typing import Any, Generator, Optional

import pytest

from test.e2e_test.conftest import (
    E2ESettings,
    FakeLLM,
    build_planner_agent_against_stubs,
    build_e2e_registry,
)


@pytest.fixture
def planner_agent(mcp_urls: dict, e2e_settings: E2ESettings) -> Any:
    """Build a real planner CrewAI Agent pointing at stub or live MCP servers."""
    return build_planner_agent_against_stubs(
        url=mcp_urls["jira"],
        e2e_settings=e2e_settings,
    )


@pytest.fixture
def planner_registry(planner_agent: Any) -> Any:
    """Wrap planner_agent in an AgentRegistry."""
    return build_e2e_registry(planner_agent, "planner")


@pytest.fixture
def create_task_and_notify_tool_order(fake_llm_session: Optional[FakeLLM]) -> Generator[None, None, None]:
    """FakeLLM: call ``create_task`` → ``send_message`` → ``reply_to_thread``.

    Use this fixture in planner tests that assert a ticket is created AND
    a Dev Lead hand-off message is sent AND a conclusion reply is posted
    (accept / full-lifecycle scenarios: PI-04, PI-05).
    """
    if fake_llm_session is not None:
        fake_llm_session.set_tool_order("create_task", "send_message", "reply_to_thread")
    yield


@pytest.fixture
def create_task_only_tool_order(fake_llm_session: Optional[FakeLLM]) -> Generator[None, None, None]:
    """FakeLLM: call ``create_task`` → ``reply_to_thread``.

    Use this fixture in planner reject tests where a REJECTED ticket is
    created and a conclusion reply is posted but no Dev Lead hand-off
    message is sent (PI-03).
    """
    if fake_llm_session is not None:
        fake_llm_session.set_tool_order("create_task", "reply_to_thread")
    yield


@pytest.fixture
def reply_to_thread_tool_order(fake_llm_session: Optional[FakeLLM]) -> Generator[None, None, None]:
    """FakeLLM: call ``reply_to_thread`` only.

    Use this fixture in planner tests where the planner should respond in-thread
    without creating tickets (e.g. PI-01 survey, PI-02 survey plan).
    """
    if fake_llm_session is not None:
        fake_llm_session.set_tool_order("reply_to_thread")
    yield


