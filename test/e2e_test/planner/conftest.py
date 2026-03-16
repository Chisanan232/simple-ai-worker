"""Planner Agent E2E test fixtures — Layer 3."""
from __future__ import annotations

from typing import Any

import pytest

from test.e2e_test.conftest import (
    E2ESettings,
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

