"""
E2E test configuration and shared fixtures — Layer 2.

Architecture (Two-Tier E2E Strategy)
--------------------------------------
E2E tests use **real LLM calls** (OpenAI / Anthropic) against either:

  Tier 1 — Stub Mode (default):
    MCPStubServer (pytest-httpserver) — no real services needed.
    Requires only an LLM API key in ``test/e2e_test/.env.e2e``.

  Tier 2 — Live Mode:
    Real MCP servers started via ``test/e2e_test/docker-compose.e2e.yml``.
    Requires all credentials in ``test/e2e_test/.env.e2e``.

The ``mcp_urls`` fixture selects the tier automatically based on whether
``E2E_MCP_*_URL`` values are populated in ``test/e2e_test/.env.e2e``.

Run commands (uv-managed project)
-----------------------------------
  uv run pytest -m "e2e" test/e2e_test/
  uv run pytest -m "e2e" test/e2e_test/dev/
  uv run pytest -m "e2e" -k "clickup" test/e2e_test/
  uv run pytest -m "e2e" -k "jira"    test/e2e_test/
"""

from __future__ import annotations

import os
from typing import Any, Generator

import pytest
from pytest_httpserver import HTTPServer

from test.e2e_test.common.e2e_settings import E2ESettings, get_e2e_settings
from test.e2e_test.common.mcp_stub import MCPStubServer, RecordingStub  # noqa: F401
from src.ticket.workflow import WorkflowConfig

# Load E2E settings once at module level — before any monkeypatching.
_e2e_settings: E2ESettings = get_e2e_settings()
_LLM_KEY_AVAILABLE: bool = _e2e_settings.has_llm_key

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

skip_without_llm = pytest.mark.skipif(
    not _LLM_KEY_AVAILABLE,
    reason=(
        "E2E tests require OPENAI_API_KEY or ANTHROPIC_API_KEY "
        "in test/e2e_test/.env.e2e"
    ),
)

skip_without_live_services = pytest.mark.skipif(
    not _e2e_settings.is_live_mode,
    reason=(
        "Live E2E tests require E2E_MCP_*_URL values set "
        "in test/e2e_test/.env.e2e"
    ),
)

# ---------------------------------------------------------------------------
# Standard 8-operation workflow config for all E2E tests
# ---------------------------------------------------------------------------

E2E_WORKFLOW_CONFIG: dict = {
    "scan_for_work":       {"status_value": "ACCEPTED",    "human_only": True},
    "skip_rejected":       {"status_value": "REJECTED"},
    "open_for_dev":        {"status_value": "OPEN",        "human_only": False},
    "in_planning":         {"status_value": "IN PLANNING", "human_only": True},
    "start_development":   {"status_value": "IN PROGRESS"},
    "open_for_review":     {"status_value": "IN REVIEW"},
    "mark_complete":       {"status_value": "COMPLETE"},
    "update_with_context": {"status_value": ""},
}


# ---------------------------------------------------------------------------
# Autouse fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _e2e_require_llm_key(request: pytest.FixtureRequest) -> None:
    """Skip any e2e-marked test at runtime if no LLM key was available at startup."""
    if request.node.get_closest_marker("e2e") and not _LLM_KEY_AVAILABLE:
        pytest.skip(
            "E2E tests require OPENAI_API_KEY or ANTHROPIC_API_KEY "
            "in test/e2e_test/.env.e2e"
        )


@pytest.fixture(autouse=True)
def e2e_state_reset() -> Generator[None, None, None]:
    """Clear all shared in-memory scheduler state before and after each test."""

    def _clear() -> None:
        try:
            import src.scheduler.jobs.scan_tickets as m
            m._in_progress_tickets.clear()
            m._open_prs.clear()
            m._prs_under_review.clear()
        except (ImportError, AttributeError):
            pass

        try:
            import src.scheduler.jobs.pr_review_comment_handler as m
            m._in_progress_comment_fixes.clear()
        except (ImportError, AttributeError):
            pass

        try:
            import src.scheduler.jobs.plan_and_notify as m
            m._in_planning_tickets.clear()
            m._plan_comment_watermarks.clear()
        except (ImportError, AttributeError):
            pass

    _clear()
    yield
    _clear()


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def e2e_settings() -> E2ESettings:
    """Return the cached E2ESettings singleton."""
    return _e2e_settings


@pytest.fixture(scope="session")
def e2e_workflow_config() -> WorkflowConfig:
    """Standard 8-operation WorkflowConfig for all E2E tests."""
    return WorkflowConfig(E2E_WORKFLOW_CONFIG)


# ---------------------------------------------------------------------------
# Function-scoped: two-tier URL selector
# ---------------------------------------------------------------------------


@pytest.fixture
def mcp_urls(e2e_settings: E2ESettings, httpserver: HTTPServer) -> dict[str, str]:
    """Return MCP server URLs for stub mode or live mode.

    Stub mode  (is_live_mode=False): all four URLs point at the same httpserver.
    Live mode  (is_live_mode=True):  URLs from E2E_MCP_*_URL in .env.e2e.
    """
    if e2e_settings.is_live_mode:
        return {
            "jira":    e2e_settings.MCP_JIRA_URL or "",
            "clickup": e2e_settings.MCP_CLICKUP_URL or "",
            "github":  e2e_settings.MCP_GITHUB_URL or "",
            "slack":   e2e_settings.MCP_SLACK_URL or "",
        }
    stub_url = httpserver.url_for("/mcp")
    return {"jira": stub_url, "clickup": stub_url, "github": stub_url, "slack": stub_url}


# ---------------------------------------------------------------------------
# Internal LLM key injection helpers
# ---------------------------------------------------------------------------


def _inject_llm_key(settings: E2ESettings) -> dict[str, str]:
    """Temporarily inject the LLM API key into os.environ for CrewAI validators."""
    llm_key = settings.llm_key_value
    llm_env_var = settings.llm_key_env_var
    injected: dict[str, str] = {}
    if llm_key and llm_env_var not in os.environ:
        os.environ[llm_env_var] = llm_key
        injected[llm_env_var] = llm_key
    return injected


def _restore_env(injected: dict[str, str]) -> None:
    for key in injected:
        os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# Centralised agent builders — all three roles
# ---------------------------------------------------------------------------


def build_dev_agent_against_stubs(
    jira_url: str,
    clickup_url: str,
    github_url: str,
    slack_url: str,
    e2e_settings: E2ESettings,
) -> Any:
    """Build a real dev_agent CrewAI Agent pointing at stub or live MCP servers."""
    from src.agents.factory import AgentFactory
    from src.config.agent_config import AgentTeamConfig
    from src.config.settings import AppSettings

    injected = _inject_llm_key(e2e_settings)
    try:
        settings_kwargs: dict = dict(
            MCP_JIRA_URL=jira_url, MCP_CLICKUP_URL=clickup_url,
            MCP_GITHUB_URL=github_url, MCP_SLACK_URL=slack_url,
        )
        llm_key = e2e_settings.llm_key_value
        if llm_key:
            settings_kwargs[e2e_settings.llm_key_env_var] = llm_key

        settings = AppSettings(**settings_kwargs)
        raw_config = {
            "process": "sequential",
            "mcp_servers": {
                "jira": {
                    "type": "http", "url": jira_url,
                    "tool_filter": [
                        "search_issues", "get_issue", "transition_issue",
                        "add_comment", "update_issue",
                    ],
                    "cache_tools_list": False,
                },
                "clickup": {
                    "type": "http", "url": clickup_url,
                    "tool_filter": [
                        "search_tasks", "get_task", "update_task", "add_comment",
                    ],
                    "cache_tools_list": False,
                },
                "github": {
                    "type": "http", "url": github_url,
                    "tool_filter": [
                        "create_pull_request", "get_pull_request", "merge_pull_request",
                        "get_pull_request_reviews", "get_pull_request_comments",
                        "reply_to_review_comment",
                    ],
                    "cache_tools_list": False,
                },
                "slack": {
                    "type": "http", "url": slack_url,
                    "tool_filter": [
                        "send_message", "reply_to_thread", "get_messages",
                        "get_thread_permalink",
                    ],
                    "cache_tools_list": False,
                },
            },
            "agents": [{
                "id": "dev_agent",
                "role": "Software Developer",
                "goal": (
                    "Scan for ready development tickets, implement them, open GitHub pull "
                    "requests, and update ticket status for human review."
                ),
                "backstory": (
                    "You are a skilled software developer who thrives in an "
                    "autonomous, pull-based workflow."
                ),
                "llm": {
                    "provider": e2e_settings.LLM_PROVIDER,
                    "model": e2e_settings.LLM_MODEL,
                },
                "mcps": ["jira", "clickup", "github", "slack"],
                "apps": [], "allow_delegation": False, "verbose": False,
            }],
        }
        team_config = AgentTeamConfig.model_validate(raw_config)
        return AgentFactory.build(
            team_config.agents[0], settings, mcp_servers=team_config.mcp_servers
        )
    finally:
        _restore_env(injected)


def build_planner_agent_against_stubs(url: str, e2e_settings: E2ESettings) -> Any:
    """Build a real planner CrewAI Agent pointing at stub or live MCP servers."""
    from src.agents.factory import AgentFactory
    from src.config.agent_config import AgentTeamConfig
    from src.config.settings import AppSettings

    injected = _inject_llm_key(e2e_settings)
    try:
        settings_kwargs: dict = dict(
            MCP_JIRA_URL=url, MCP_CLICKUP_URL=url,
            MCP_GITHUB_URL=url, MCP_SLACK_URL=url,
        )
        llm_key = e2e_settings.llm_key_value
        if llm_key:
            settings_kwargs[e2e_settings.llm_key_env_var] = llm_key

        settings = AppSettings(**settings_kwargs)
        raw_config = {
            "process": "sequential",
            "mcp_servers": {
                "jira": {
                    "type": "http", "url": url,
                    "tool_filter": [
                        "create_issue", "search_issues", "update_issue",
                        "transition_issue", "add_comment",
                    ],
                    "cache_tools_list": False,
                },
                "clickup": {
                    "type": "http", "url": url,
                    "tool_filter": [
                        "create_task", "search_tasks", "update_task", "add_comment",
                    ],
                    "cache_tools_list": False,
                },
                "slack": {
                    "type": "http", "url": url,
                    "tool_filter": ["send_message", "reply_to_thread", "get_messages"],
                    "cache_tools_list": False,
                },
            },
            "agents": [{
                "id": "planner",
                "role": "Product Planner",
                "goal": (
                    "Survey and discuss product ideas with human stakeholders; "
                    "create detailed planning documents; manage accept/reject outcomes."
                ),
                "backstory": (
                    "You are a seasoned product planner with 10+ years of experience "
                    "in agile software development, market research, and business modelling."
                ),
                "llm": {
                    "provider": e2e_settings.LLM_PROVIDER,
                    "model": e2e_settings.LLM_MODEL,
                },
                "mcps": ["jira", "clickup", "slack"],
                "apps": [], "allow_delegation": False, "verbose": False,
            }],
        }
        team_config = AgentTeamConfig.model_validate(raw_config)
        return AgentFactory.build(
            team_config.agents[0], settings, mcp_servers=team_config.mcp_servers
        )
    finally:
        _restore_env(injected)


def build_dev_lead_agent_against_stubs(url: str, e2e_settings: E2ESettings) -> Any:
    """Build a real dev_lead CrewAI Agent pointing at stub or live MCP servers."""
    from src.agents.factory import AgentFactory
    from src.config.agent_config import AgentTeamConfig
    from src.config.settings import AppSettings

    injected = _inject_llm_key(e2e_settings)
    try:
        settings_kwargs: dict = dict(
            MCP_JIRA_URL=url, MCP_CLICKUP_URL=url,
            MCP_GITHUB_URL=url, MCP_SLACK_URL=url,
        )
        llm_key = e2e_settings.llm_key_value
        if llm_key:
            settings_kwargs[e2e_settings.llm_key_env_var] = llm_key

        settings = AppSettings(**settings_kwargs)
        raw_config = {
            "process": "sequential",
            "mcp_servers": {
                "jira": {
                    "type": "http", "url": url,
                    "tool_filter": [
                        "get_issue", "search_issues", "create_issue",
                        "update_issue", "link_issues", "transition_issue", "add_comment",
                    ],
                    "cache_tools_list": False,
                },
                "clickup": {
                    "type": "http", "url": url,
                    "tool_filter": [
                        "get_task", "search_tasks", "create_task",
                        "update_task", "add_comment",
                    ],
                    "cache_tools_list": False,
                },
                "slack": {
                    "type": "http", "url": url,
                    "tool_filter": ["send_message", "reply_to_thread", "get_messages"],
                    "cache_tools_list": False,
                },
            },
            "agents": [{
                "id": "dev_lead",
                "role": "Technical Dev Lead",
                "goal": (
                    "Assess feasibility of requirements, ask clarifying questions, "
                    "break down epics into sub-tasks with dependencies."
                ),
                "backstory": (
                    "You are a senior software engineer and tech lead specialising "
                    "in system design, task decomposition, and agile estimation."
                ),
                "llm": {
                    "provider": e2e_settings.LLM_PROVIDER,
                    "model": e2e_settings.LLM_MODEL,
                },
                "mcps": ["jira", "clickup", "slack"],
                "apps": [], "allow_delegation": False, "verbose": False,
            }],
        }
        team_config = AgentTeamConfig.model_validate(raw_config)
        return AgentFactory.build(
            team_config.agents[0], settings, mcp_servers=team_config.mcp_servers
        )
    finally:
        _restore_env(injected)


def build_e2e_registry(agent: Any, role_id: str = "dev_agent") -> Any:
    """Wrap *agent* in an AgentRegistry under *role_id*."""
    from src.agents.registry import AgentRegistry

    registry = AgentRegistry()
    registry.register(role_id, agent)
    return registry

