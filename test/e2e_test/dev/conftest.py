"""Dev Agent E2E test fixtures — Layer 3."""
from __future__ import annotations

from typing import Any

import pytest

from test.e2e_test.conftest import (
    E2ESettings,
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

