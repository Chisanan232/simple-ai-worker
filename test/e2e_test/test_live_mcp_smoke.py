"""
Live mode infrastructure smoke tests (E2E-LIVE-01 through E2E-LIVE-06).

These tests verify that the real MCP server containers started by
``docker compose -f test/e2e_test/docker-compose.e2e.yml`` are reachable
and respond to the MCP ``tools/list`` call.

**All tests are skipped in stub mode** (when ``E2E_MCP_*_URL`` values are
not set in ``test/e2e_test/.env.e2e``).  Run them after starting the
Docker Compose stack:

    docker compose \\
        -f test/e2e_test/docker-compose.e2e.yml \\
        --env-file test/e2e_test/.env.e2e \\
        up -d --wait

    uv run pytest test/e2e_test/test_live_mcp_smoke.py -v
"""

from __future__ import annotations

import requests

from test.e2e_test.conftest import skip_without_live_services
from test.e2e_test.common.e2e_settings import get_e2e_settings

_e2e_settings = get_e2e_settings()


def _tools_list_request(url: str) -> dict:
    """Send an MCP ``tools/list`` JSON-RPC 2.0 request and return the parsed response."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {},
    }
    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


# ===========================================================================
# E2E-LIVE-01: JIRA MCP server is reachable
# ===========================================================================

@skip_without_live_services
class TestJIRAMCPServer:
    def test_e2e_live_01_jira_mcp_reachable(self) -> None:
        """E2E-LIVE-01: JIRA MCP server responds to tools/list."""
        url = _e2e_settings.MCP_JIRA_URL
        assert url, "E2E_MCP_JIRA_URL not set — live mode not configured"

        result = _tools_list_request(url)

        assert "result" in result or "tools" in result, (
            f"Expected MCP tools/list response. Got: {result}"
        )
        # Should expose JIRA tools
        tools = result.get("result", {}).get("tools", result.get("tools", []))
        tool_names = [t.get("name", "") for t in tools]
        assert any("issue" in name.lower() or "jira" in name.lower()
                   for name in tool_names), (
            f"Expected at least one JIRA tool. Got tool names: {tool_names}"
        )

    def test_e2e_live_02_jira_mcp_search_issues_tool_present(self) -> None:
        """E2E-LIVE-02: JIRA MCP exposes search_issues tool."""
        url = _e2e_settings.MCP_JIRA_URL
        assert url, "E2E_MCP_JIRA_URL not set"

        result = _tools_list_request(url)
        tools = result.get("result", {}).get("tools", result.get("tools", []))
        tool_names = [t.get("name", "") for t in tools]

        assert "search_issues" in tool_names, (
            f"Expected search_issues tool in JIRA MCP. Got: {tool_names}"
        )


# ===========================================================================
# E2E-LIVE-03: ClickUp MCP server is reachable
# ===========================================================================

@skip_without_live_services
class TestClickUpMCPServer:
    def test_e2e_live_03_clickup_mcp_reachable(self) -> None:
        """E2E-LIVE-03: ClickUp MCP server responds to tools/list."""
        url = _e2e_settings.MCP_CLICKUP_URL
        assert url, "E2E_MCP_CLICKUP_URL not set — live mode not configured"

        result = _tools_list_request(url)

        assert "result" in result or "tools" in result, (
            f"Expected MCP tools/list response. Got: {result}"
        )
        tools = result.get("result", {}).get("tools", result.get("tools", []))
        tool_names = [t.get("name", "") for t in tools]
        assert any("task" in name.lower() or "clickup" in name.lower()
                   for name in tool_names), (
            f"Expected at least one ClickUp tool. Got: {tool_names}"
        )

    def test_e2e_live_04_clickup_mcp_search_tasks_tool_present(self) -> None:
        """E2E-LIVE-04: ClickUp MCP exposes search_tasks tool."""
        url = _e2e_settings.MCP_CLICKUP_URL
        assert url, "E2E_MCP_CLICKUP_URL not set"

        result = _tools_list_request(url)
        tools = result.get("result", {}).get("tools", result.get("tools", []))
        tool_names = [t.get("name", "") for t in tools]

        assert "search_tasks" in tool_names, (
            f"Expected search_tasks tool in ClickUp MCP. Got: {tool_names}"
        )


# ===========================================================================
# E2E-LIVE-05: GitHub MCP server is reachable
# ===========================================================================

@skip_without_live_services
class TestGitHubMCPServer:
    def test_e2e_live_05_github_mcp_reachable(self) -> None:
        """E2E-LIVE-05: GitHub MCP server responds to tools/list."""
        url = _e2e_settings.MCP_GITHUB_URL
        assert url, "E2E_MCP_GITHUB_URL not set — live mode not configured"

        result = _tools_list_request(url)

        assert "result" in result or "tools" in result, (
            f"Expected MCP tools/list response. Got: {result}"
        )
        tools = result.get("result", {}).get("tools", result.get("tools", []))
        tool_names = [t.get("name", "") for t in tools]
        assert any("pull_request" in name.lower() or "github" in name.lower()
                   or "pr" in name.lower()
                   for name in tool_names), (
            f"Expected at least one GitHub tool. Got: {tool_names}"
        )


# ===========================================================================
# E2E-LIVE-06: Slack MCP server is reachable
# ===========================================================================

@skip_without_live_services
class TestSlackMCPServer:
    def test_e2e_live_06_slack_mcp_reachable(self) -> None:
        """E2E-LIVE-06: Slack MCP server responds to tools/list."""
        url = _e2e_settings.MCP_SLACK_URL
        assert url, "E2E_MCP_SLACK_URL not set — live mode not configured"

        result = _tools_list_request(url)

        assert "result" in result or "tools" in result, (
            f"Expected MCP tools/list response. Got: {result}"
        )
        tools = result.get("result", {}).get("tools", result.get("tools", []))
        tool_names = [t.get("name", "") for t in tools]
        assert any("message" in name.lower() or "slack" in name.lower()
                   or "thread" in name.lower()
                   for name in tool_names), (
            f"Expected at least one Slack tool. Got: {tool_names}"
        )


