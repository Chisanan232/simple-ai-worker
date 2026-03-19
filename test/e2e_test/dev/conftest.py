"""Dev Agent E2E test fixtures — Layer 3."""
from __future__ import annotations

from typing import Any, Generator, Optional

import pytest

from test.e2e_test.conftest import (
    E2ESettings,
    FakeLLM,
    build_dev_agent_against_stubs,
    build_e2e_registry,
)


@pytest.fixture
def dev_agent(mcp_urls: dict, e2e_settings: E2ESettings) -> Any:
    """Build a real dev_agent CrewAI Agent pointing at stub or live MCP servers."""
    return build_dev_agent_against_stubs(
        jira_url=mcp_urls["jira"],
        clickup_url=mcp_urls["clickup"],
        github_url=mcp_urls["github"],
        slack_url=mcp_urls["slack"],
        e2e_settings=e2e_settings,
    )


@pytest.fixture
def dev_registry(dev_agent: Any) -> Any:
    """Wrap dev_agent in an AgentRegistry."""
    return build_e2e_registry(dev_agent, "dev_agent")


@pytest.fixture
def planning_tool_order(fake_llm_session: Optional[FakeLLM]) -> Generator[None, None, None]:
    """Configure the session FakeLLM to call ``get_task`` then ``add_comment``.

    Use this fixture in plan_and_notify tests that assert ``add_comment`` was
    called.  The ``_fake_llm_reset`` autouse fixture in the parent conftest
    will restore the original tool order after the test.
    """
    if fake_llm_session is not None:
        fake_llm_session.set_tool_order("get_task", "add_comment")
    yield


@pytest.fixture
def thread_summary_tool_order(fake_llm_session: Optional[FakeLLM]) -> Generator[None, None, None]:
    """FakeLLM: call ``get_messages`` then ``add_comment``.

    Use this fixture in Slack thread-summary tests that assert both
    ``stub.was_called("get_messages")`` and ``stub.was_called("add_comment")``.
    """
    if fake_llm_session is not None:
        fake_llm_session.set_tool_order("get_messages", "add_comment")
    yield


@pytest.fixture
def permalink_tool_order(fake_llm_session: Optional[FakeLLM]) -> Generator[None, None, None]:
    """FakeLLM: call ``get_thread_permalink`` then ``add_comment``.

    Use this fixture in tests that assert the Slack permalink appears inside
    the ticket comment body.
    """
    if fake_llm_session is not None:
        fake_llm_session.set_tool_order("get_thread_permalink", "add_comment")
    yield


@pytest.fixture
def reply_only_tool_order(fake_llm_session: Optional[FakeLLM]) -> Generator[None, None, None]:
    """FakeLLM: call ``reply_to_thread`` only.

    Use this fixture in tests that assert the agent asks for a ticket ID
    via Slack (BR-6 guardrail) rather than calling ``add_comment``.
    """
    if fake_llm_session is not None:
        fake_llm_session.set_tool_order("reply_to_thread")
    yield


@pytest.fixture
def merge_tool_order(fake_llm_session: Optional[FakeLLM]) -> Generator[None, None, None]:
    """FakeLLM: call ``get_pull_request_reviews`` then ``merge_pull_request``.

    Use this fixture in PR auto-merge tests that assert
    ``merge_pull_request`` was called.
    """
    if fake_llm_session is not None:
        fake_llm_session.set_tool_order("get_pull_request_reviews", "merge_pull_request")
    yield


@pytest.fixture
def review_reply_tool_order(fake_llm_session: Optional[FakeLLM]) -> Generator[None, None, None]:
    """FakeLLM: call ``get_pull_request_comments`` then ``reply_to_review_comment``.

    Use this fixture in PR review-comment tests that assert the agent
    replies to unresolved inline comments.
    """
    if fake_llm_session is not None:
        fake_llm_session.set_tool_order("get_pull_request_comments", "reply_to_review_comment")
    yield


@pytest.fixture
def user_merged_tool_order(fake_llm_session: Optional[FakeLLM]) -> Generator[None, None, None]:
    """FakeLLM: call ``get_pull_request`` then ``update_task`` (or ``transition_issue``).

    Use this in tests where the PR was already merged by the human before the
    watcher fires (E2E-14, E2E-21).

    Tool sequence per crew:
    - Status check crew (turn 0): ``get_pull_request`` → returns merged=True JSON
      → status check crew finishes, ``_choose_final_answer`` returns is_merged=True.
    - Merge crew (turn 0): ``get_pull_request`` (harmless re-check), (turn 1):
      ``update_task`` → transitions ticket → merge crew finishes with default response.

    Both crews reset to turn 0 (fresh conversation), so the two-tool sequence
    covers both the status-check and the subsequent merge/transition crew.
    """
    if fake_llm_session is not None:
        fake_llm_session.set_tool_order("get_pull_request", "update_task")
    yield


@pytest.fixture
def dev_workflow_tool_order(fake_llm_session: Optional[FakeLLM]) -> Generator[None, None, None]:
    """FakeLLM: call ``get_task`` → ``update_task`` → ``create_pull_request``.

    Use this fixture in dev-agent scan-and-dispatch tests that assert the
    agent fetches a ticket, transitions it, and opens a PR (E2E-07).
    """
    if fake_llm_session is not None:
        fake_llm_session.set_tool_order("get_task", "update_task", "create_pull_request")
    yield


@pytest.fixture
def no_approval_tool_order(fake_llm_session: Optional[FakeLLM]) -> Generator[None, None, None]:
    """FakeLLM: call ``get_pull_request_reviews`` for 0-approval BR-2 guard tests.

    Use this in tests where the PR has 0 approvals so the watcher should
    skip the merge (E2E-12).  The reviews tool returns [] → approval_count=0.
    """
    if fake_llm_session is not None:
        fake_llm_session.set_tool_order("get_pull_request_reviews")
    yield


@pytest.fixture
def pr_timeout_tool_order(fake_llm_session: Optional[FakeLLM]) -> Generator[None, None, None]:
    """FakeLLM: call ``get_pull_request_reviews`` for timeout-not-elapsed tests.

    Use this in tests where the PR has approvals but timeout hasn't elapsed
    (E2E-13).  The reviews result shows 1 approval but age check prevents merge.
    """
    if fake_llm_session is not None:
        fake_llm_session.set_tool_order("get_pull_request_reviews")
    yield


