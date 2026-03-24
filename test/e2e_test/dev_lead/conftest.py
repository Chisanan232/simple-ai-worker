"""Dev Lead Agent E2E test fixtures — Layer 3."""

from __future__ import annotations

from test.e2e_test.conftest import (
    E2ESettings,
    FakeLLM,
    build_dev_lead_agent_against_stubs,
    build_e2e_registry,
)
from typing import Any, Generator, Optional

import pytest


@pytest.fixture
def dev_lead_agent(mcp_urls: dict, e2e_settings: E2ESettings) -> Any:
    """Build a real dev_lead CrewAI Agent pointing at stub or live MCP servers."""
    return build_dev_lead_agent_against_stubs(
        url=mcp_urls["jira"],
        e2e_settings=e2e_settings,
    )


@pytest.fixture
def dev_lead_registry(dev_lead_agent: Any) -> Any:
    """Wrap dev_lead_agent in an AgentRegistry."""
    return build_e2e_registry(dev_lead_agent, "dev_lead")


@pytest.fixture
def get_task_tool_order(fake_llm_session: Optional[FakeLLM]) -> Generator[None, None, None]:
    """FakeLLM: call ``get_task`` only.

    Use this fixture in dev-lead tests that assert the agent fetches an
    existing task when a task ID is present in the message (E2E-DL-02).
    """
    if fake_llm_session is not None:
        fake_llm_session.set_tool_order("get_task")
    yield


@pytest.fixture
def breakdown_tool_order(fake_llm_session: Optional[FakeLLM]) -> Generator[None, None, None]:
    """FakeLLM: call ``get_task`` → ``create_task`` × 2 → ``add_comment`` → ``reply_to_thread``.

    Use this fixture in dev-lead sub-task breakdown tests that assert at
    least two sub-tasks are created, a comment is added on the parent task,
    and the agent replies in Slack (E2E-DL-03, DL-06).
    """
    if fake_llm_session is not None:
        fake_llm_session.set_tool_order("get_task", "create_task", "create_task", "add_comment", "reply_to_thread")
    yield


@pytest.fixture
def planning_tool_order(fake_llm_session: Optional[FakeLLM]) -> Generator[None, None, None]:
    """FakeLLM: call ``get_task`` then ``add_comment`` (Dev Agent planning tests).

    Use this fixture in plan_and_notify tests that assert ``add_comment`` was
    called (E2E-DL-04, E2E-DL-05).
    """
    if fake_llm_session is not None:
        fake_llm_session.set_tool_order("get_task", "add_comment")
    yield


@pytest.fixture
def reply_only_tool_order(fake_llm_session: Optional[FakeLLM]) -> Generator[None, None, None]:
    """FakeLLM: call ``reply_to_thread`` only.

    Use this fixture in dev-lead tests where the agent should reply in-thread
    with clarifying questions rather than creating tasks (E2E-DL-01).
    """
    if fake_llm_session is not None:
        fake_llm_session.set_tool_order("reply_to_thread")
    yield
