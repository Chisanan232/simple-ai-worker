"""
Unit tests for :class:`src.agents.factory.AgentFactory`.

crewai.Agent and crewai.LLM are both patched so no live API calls occur.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.agents.factory import AgentFactory
from src.config.agent_config import (
    AgentConfig,
    LLMConfig,
    LLMOptions,
    MCPServerDefinition,
    MCPServerRef,
)
from src.config.settings import AppSettings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_config(
    agent_id: str = "planner",
    role: str = "Product Planner",
    goal: str = "Plan the product.",
    backstory: str = "You are a planner.",
    provider: str = "openai",
    model: str = "gpt-4o",
    apps: Optional[List[str]] = None,
    mcps: Optional[list] = None,
    allow_delegation: bool = False,
    verbose: bool = False,
) -> AgentConfig:
    return AgentConfig(
        id=agent_id,
        role=role,
        goal=goal,
        backstory=backstory,
        llm=LLMConfig(
            provider=provider,
            model=model,
            options=LLMOptions(),
        ),
        apps=apps or [],
        mcps=mcps or [],
        allow_delegation=allow_delegation,
        verbose=verbose,
    )


def _make_http_definition(
    url: str = "http://127.0.0.1:8100/mcp",
    headers: Optional[Dict[str, str]] = None,
    tool_filter: Optional[List[str]] = None,
    cache_tools_list: bool = False,
) -> MCPServerDefinition:
    return MCPServerDefinition(
        type="http",
        url=url,
        headers=headers,
        tool_filter=tool_filter,
        cache_tools_list=cache_tools_list,
    )


def _make_sse_definition(
    url: str = "http://127.0.0.1:8104/mcp/sse",
    tool_filter: Optional[List[str]] = None,
) -> MCPServerDefinition:
    return MCPServerDefinition(type="sse", url=url, tool_filter=tool_filter)


def _make_stdio_definition(
    command: str = "python",
    args: Optional[List[str]] = None,
    tool_filter: Optional[List[str]] = None,
) -> MCPServerDefinition:
    return MCPServerDefinition(
        type="stdio",
        command=command,
        args=args or ["server.py"],
        tool_filter=tool_filter,
    )


def _make_settings(openai_key: str = "sk-test") -> AppSettings:
    return AppSettings(OPENAI_API_KEY=openai_key)


# ===========================================================================
# AgentFactory.build
# ===========================================================================


class TestAgentFactoryBuild:
    """Tests for AgentFactory.build() — crewai.Agent and crewai.LLM patched."""

    def test_build_returns_agent_instance(self) -> None:
        """build() must return the crewai.Agent produced by the constructor."""
        cfg = _make_agent_config()
        settings = _make_settings()
        mock_agent = MagicMock()
        mock_llm = MagicMock()

        with (
            patch("src.agents.llm_factory.LLM", return_value=mock_llm),
            patch("src.agents.factory.Agent", return_value=mock_agent) as MockAgent,
        ):
            result = AgentFactory.build(cfg, settings)

        assert result is mock_agent
        MockAgent.assert_called_once()

    def test_build_passes_role(self) -> None:
        """build() must pass agent_config.role to Agent()."""
        cfg = _make_agent_config(role="Dev Lead")
        settings = _make_settings()

        with (
            patch("src.agents.llm_factory.LLM"),
            patch("src.agents.factory.Agent") as MockAgent,
        ):
            AgentFactory.build(cfg, settings)

        kwargs = MockAgent.call_args.kwargs
        assert kwargs["role"] == "Dev Lead"

    def test_build_passes_goal(self) -> None:
        """build() must pass agent_config.goal to Agent()."""
        cfg = _make_agent_config(goal="Break down epics.")
        settings = _make_settings()

        with (
            patch("src.agents.llm_factory.LLM"),
            patch("src.agents.factory.Agent") as MockAgent,
        ):
            AgentFactory.build(cfg, settings)

        kwargs = MockAgent.call_args.kwargs
        assert kwargs["goal"] == "Break down epics."

    def test_build_passes_backstory(self) -> None:
        """build() must pass agent_config.backstory to Agent()."""
        cfg = _make_agent_config(backstory="You are a senior dev.")
        settings = _make_settings()

        with (
            patch("src.agents.llm_factory.LLM"),
            patch("src.agents.factory.Agent") as MockAgent,
        ):
            AgentFactory.build(cfg, settings)

        kwargs = MockAgent.call_args.kwargs
        assert kwargs["backstory"] == "You are a senior dev."

    def test_build_passes_apps_list(self) -> None:
        """build() must pass agent_config.apps directly to Agent()."""
        cfg = _make_agent_config(apps=["jira/create_issue", "slack/reply_to_thread"])
        settings = _make_settings()

        with (
            patch("src.agents.llm_factory.LLM"),
            patch("src.agents.factory.Agent") as MockAgent,
        ):
            AgentFactory.build(cfg, settings)

        kwargs = MockAgent.call_args.kwargs
        assert kwargs["apps"] == ["jira/create_issue", "slack/reply_to_thread"]

    def test_build_passes_empty_apps_as_none(self) -> None:
        """build() converts an empty apps list to None when passed to Agent()."""
        cfg = _make_agent_config(apps=[])
        settings = _make_settings()

        with (
            patch("src.agents.llm_factory.LLM"),
            patch("src.agents.factory.Agent") as MockAgent,
        ):
            AgentFactory.build(cfg, settings)

        kwargs = MockAgent.call_args.kwargs
        # Empty apps list is coerced to None so crewai.Agent gets None, not [].
        assert kwargs["apps"] is None

    def test_build_passes_allow_delegation_false(self) -> None:
        """build() must pass allow_delegation=False to Agent()."""
        cfg = _make_agent_config(allow_delegation=False)
        settings = _make_settings()

        with (
            patch("src.agents.llm_factory.LLM"),
            patch("src.agents.factory.Agent") as MockAgent,
        ):
            AgentFactory.build(cfg, settings)

        kwargs = MockAgent.call_args.kwargs
        assert kwargs["allow_delegation"] is False

    def test_build_passes_allow_delegation_true(self) -> None:
        """build() must pass allow_delegation=True when set."""
        cfg = _make_agent_config(allow_delegation=True)
        settings = _make_settings()

        with (
            patch("src.agents.llm_factory.LLM"),
            patch("src.agents.factory.Agent") as MockAgent,
        ):
            AgentFactory.build(cfg, settings)

        kwargs = MockAgent.call_args.kwargs
        assert kwargs["allow_delegation"] is True

    def test_build_passes_verbose(self) -> None:
        """build() must forward the verbose flag to Agent()."""
        cfg = _make_agent_config(verbose=True)
        settings = _make_settings()

        with (
            patch("src.agents.llm_factory.LLM"),
            patch("src.agents.factory.Agent") as MockAgent,
        ):
            AgentFactory.build(cfg, settings)

        kwargs = MockAgent.call_args.kwargs
        assert kwargs["verbose"] is True

    def test_build_delegates_to_llm_factory(self) -> None:
        """build() must call LLMFactory.build() with the llm config and settings."""
        cfg = _make_agent_config()
        settings = _make_settings()

        with (
            patch("src.agents.factory.LLMFactory") as MockLLMFactory,
            patch("src.agents.factory.Agent"),
        ):
            mock_llm = MagicMock()
            MockLLMFactory.build.return_value = mock_llm
            AgentFactory.build(cfg, settings)

        MockLLMFactory.build.assert_called_once_with(cfg.llm, settings)

    def test_build_passes_llm_to_agent(self) -> None:
        """The LLM returned by LLMFactory.build() must be passed to Agent()."""
        cfg = _make_agent_config()
        settings = _make_settings()
        mock_llm = MagicMock()

        with (
            patch("src.agents.factory.LLMFactory") as MockLLMFactory,
            patch("src.agents.factory.Agent") as MockAgent,
        ):
            MockLLMFactory.build.return_value = mock_llm
            AgentFactory.build(cfg, settings)

        kwargs = MockAgent.call_args.kwargs
        assert kwargs["llm"] is mock_llm

    def test_build_passes_mcps_none_when_no_mcps(self) -> None:
        """build() passes mcps=None to Agent() when agent has no mcps configured."""
        cfg = _make_agent_config(mcps=[])
        settings = _make_settings()

        with (
            patch("src.agents.llm_factory.LLM"),
            patch("src.agents.factory.Agent") as MockAgent,
        ):
            AgentFactory.build(cfg, settings)

        kwargs = MockAgent.call_args.kwargs
        assert kwargs["mcps"] is None

    def test_build_passes_mcps_none_when_no_registry(self) -> None:
        """build() passes mcps=None when mcp_servers dict is not provided."""
        cfg = _make_agent_config(mcps=["jira"])
        settings = _make_settings()

        with (
            patch("src.agents.llm_factory.LLM"),
            patch("src.agents.factory.Agent") as MockAgent,
        ):
            # mcp_servers not passed → no resolution possible
            AgentFactory.build(cfg, settings, mcp_servers=None)

        kwargs = MockAgent.call_args.kwargs
        assert kwargs["mcps"] is None


# ===========================================================================
# AgentFactory._resolve_mcp_configs
# ===========================================================================


class TestResolveMcpConfigs:
    """Tests for AgentFactory._resolve_mcp_configs()."""

    def test_plain_string_ref_resolves_to_http(self) -> None:
        """A plain string ID resolves to MCPServerHTTP for an http definition."""
        from crewai.mcp import MCPServerHTTP

        registry = {"jira": _make_http_definition(url="http://127.0.0.1:8100/mcp")}
        settings = _make_settings()

        result = AgentFactory._resolve_mcp_configs(["jira"], registry, settings)

        assert len(result) == 1
        assert isinstance(result[0], MCPServerHTTP)
        assert result[0].url == "http://127.0.0.1:8100/mcp"

    def test_plain_string_ref_resolves_to_sse(self) -> None:
        """A plain string ID resolves to MCPServerSSE for an sse definition."""
        from crewai.mcp import MCPServerSSE

        registry = {"legacy": _make_sse_definition(url="http://127.0.0.1:8104/mcp/sse")}
        settings = _make_settings()

        result = AgentFactory._resolve_mcp_configs(["legacy"], registry, settings)

        assert len(result) == 1
        assert isinstance(result[0], MCPServerSSE)
        assert result[0].url == "http://127.0.0.1:8104/mcp/sse"

    def test_plain_string_ref_resolves_to_stdio(self) -> None:
        """A plain string ID resolves to MCPServerStdio for a stdio definition."""
        from crewai.mcp import MCPServerStdio

        registry = {"local": _make_stdio_definition(command="python", args=["srv.py"])}
        settings = _make_settings()

        result = AgentFactory._resolve_mcp_configs(["local"], registry, settings)

        assert len(result) == 1
        assert isinstance(result[0], MCPServerStdio)
        assert result[0].command == "python"

    def test_multiple_refs_resolved_in_order(self) -> None:
        """Multiple plain string IDs are resolved in list order."""
        from crewai.mcp import MCPServerHTTP

        registry = {
            "jira": _make_http_definition(url="http://127.0.0.1:8100/mcp"),
            "slack": _make_http_definition(url="http://127.0.0.1:8103/mcp"),
        }
        settings = _make_settings()

        result = AgentFactory._resolve_mcp_configs(["jira", "slack"], registry, settings)

        assert len(result) == 2
        assert isinstance(result[0], MCPServerHTTP)
        assert result[0].url == "http://127.0.0.1:8100/mcp"
        assert result[1].url == "http://127.0.0.1:8103/mcp"

    def test_empty_refs_returns_empty_list(self) -> None:
        """Empty refs list returns an empty list."""
        result = AgentFactory._resolve_mcp_configs([], {}, _make_settings())
        assert result == []

    def test_registry_tool_filter_is_used_for_plain_ref(self) -> None:
        """A plain string ref uses the registry's tool_filter unchanged."""
        from crewai.mcp import MCPServerHTTP

        defn = _make_http_definition(tool_filter=["create_issue", "search_issues"])
        registry = {"jira": defn}
        settings = _make_settings()

        result = AgentFactory._resolve_mcp_configs(["jira"], registry, settings)

        assert len(result) == 1
        assert isinstance(result[0], MCPServerHTTP)
        # tool_filter is a callable (StaticToolFilter); check it was set (not None)
        assert result[0].tool_filter is not None

    def test_override_ref_replaces_tool_filter(self) -> None:
        """An MCPServerRef with tool_filter overrides the registry tool_filter."""
        from crewai.mcp import MCPServerHTTP

        defn = _make_http_definition(tool_filter=["create_issue", "search_issues", "update_issue"])
        registry = {"jira": defn}
        settings = _make_settings()

        ref = MCPServerRef(server="jira", tool_filter=["search_issues"])
        result = AgentFactory._resolve_mcp_configs([ref], registry, settings)

        assert len(result) == 1
        assert isinstance(result[0], MCPServerHTTP)
        # The overridden filter is set (not None)
        assert result[0].tool_filter is not None

    def test_override_ref_with_none_tool_filter_uses_registry_filter(self) -> None:
        """MCPServerRef with tool_filter=None falls back to the registry tool_filter."""
        defn = _make_http_definition(tool_filter=["create_issue"])
        registry = {"jira": defn}
        settings = _make_settings()

        ref = MCPServerRef(server="jira", tool_filter=None)
        result = AgentFactory._resolve_mcp_configs([ref], registry, settings)

        # Registry has a tool_filter → it should be applied
        assert result[0].tool_filter is not None

    def test_plain_ref_no_tool_filter_yields_none_filter(self) -> None:
        """Plain ref to a definition with no tool_filter yields filter=None."""
        defn = _make_http_definition(tool_filter=None)
        registry = {"jira": defn}
        settings = _make_settings()

        result = AgentFactory._resolve_mcp_configs(["jira"], registry, settings)

        assert result[0].tool_filter is None

    def test_cache_tools_list_forwarded(self) -> None:
        """cache_tools_list from the registry definition is forwarded to crewai config."""
        defn = _make_http_definition(cache_tools_list=True)
        registry = {"jira": defn}
        settings = _make_settings()

        result = AgentFactory._resolve_mcp_configs(["jira"], registry, settings)

        assert result[0].cache_tools_list is True


# ===========================================================================
# AgentFactory._definition_to_crewai
# ===========================================================================


class TestDefinitionToCrewai:
    """Tests for AgentFactory._definition_to_crewai()."""

    def test_http_type_returns_mcpserverhttp(self) -> None:
        from crewai.mcp import MCPServerHTTP

        defn = _make_http_definition()
        result = AgentFactory._definition_to_crewai(defn, None, _make_settings())
        assert isinstance(result, MCPServerHTTP)

    def test_sse_type_returns_mcpserversse(self) -> None:
        from crewai.mcp import MCPServerSSE

        defn = _make_sse_definition()
        result = AgentFactory._definition_to_crewai(defn, None, _make_settings())
        assert isinstance(result, MCPServerSSE)

    def test_stdio_type_returns_mcpserverstdio(self) -> None:
        from crewai.mcp import MCPServerStdio

        defn = _make_stdio_definition()
        result = AgentFactory._definition_to_crewai(defn, None, _make_settings())
        assert isinstance(result, MCPServerStdio)

    def test_http_url_forwarded(self) -> None:
        defn = _make_http_definition(url="http://jira.local/mcp")
        result = AgentFactory._definition_to_crewai(defn, None, _make_settings())
        assert result.url == "http://jira.local/mcp"

    def test_http_streamable_default_true(self) -> None:
        defn = _make_http_definition()
        result = AgentFactory._definition_to_crewai(defn, None, _make_settings())
        assert result.streamable is True

    def test_stdio_command_and_args_forwarded(self) -> None:
        defn = _make_stdio_definition(command="npx", args=["-y", "@mcp/server"])
        result = AgentFactory._definition_to_crewai(defn, None, _make_settings())
        assert result.command == "npx"
        assert result.args == ["-y", "@mcp/server"]

    def test_tool_filter_none_when_no_filter(self) -> None:
        defn = _make_http_definition()
        result = AgentFactory._definition_to_crewai(defn, None, _make_settings())
        assert result.tool_filter is None

    def test_tool_filter_set_when_filter_provided(self) -> None:
        defn = _make_http_definition()
        result = AgentFactory._definition_to_crewai(defn, ["create_issue"], _make_settings())
        assert result.tool_filter is not None


# ===========================================================================
# AgentFactory._resolve_headers
# ===========================================================================


class TestResolveHeaders:
    """Tests for AgentFactory._resolve_headers()."""

    def test_none_headers_returns_none(self) -> None:
        result = AgentFactory._resolve_headers(None, _make_settings())
        assert result is None

    def test_empty_dict_returns_empty_dict(self) -> None:
        result = AgentFactory._resolve_headers({}, _make_settings())
        assert result == {}

    def test_plain_header_unchanged(self) -> None:
        headers = {"X-Custom": "plain-value"}
        result = AgentFactory._resolve_headers(headers, _make_settings())
        assert result == {"X-Custom": "plain-value"}

    def test_placeholder_resolved_from_settings_secret_str(self) -> None:
        """${MCP_JIRA_TOKEN} is resolved from AppSettings SecretStr field."""
        settings = AppSettings(OPENAI_API_KEY="sk-test", MCP_JIRA_TOKEN="my-jira-token")
        headers = {"Authorization": "Bearer ${MCP_JIRA_TOKEN}"}

        result = AgentFactory._resolve_headers(headers, settings)

        assert result == {"Authorization": "Bearer my-jira-token"}

    def test_placeholder_resolved_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """${CUSTOM_VAR} falls back to os.environ when not in AppSettings."""
        monkeypatch.setenv("CUSTOM_MCP_VAR", "env-token-value")
        settings = _make_settings()
        headers = {"Authorization": "Bearer ${CUSTOM_MCP_VAR}"}

        result = AgentFactory._resolve_headers(headers, settings)

        assert result == {"Authorization": "Bearer env-token-value"}

    def test_unresolvable_placeholder_left_unchanged(self) -> None:
        """Placeholder that resolves to nothing is left as-is."""
        settings = _make_settings()
        headers = {"Authorization": "Bearer ${NONEXISTENT_TOKEN}"}

        result = AgentFactory._resolve_headers(headers, settings)

        assert result == {"Authorization": "Bearer ${NONEXISTENT_TOKEN}"}

    def test_multiple_headers_all_resolved(self) -> None:
        """All headers in the dict are resolved independently."""
        settings = AppSettings(
            OPENAI_API_KEY="sk-test",
            MCP_JIRA_TOKEN="jira-tok",
            MCP_SLACK_TOKEN="slack-tok",
        )
        headers = {
            "X-Jira-Auth": "Bearer ${MCP_JIRA_TOKEN}",
            "X-Slack-Auth": "Bearer ${MCP_SLACK_TOKEN}",
        }

        result = AgentFactory._resolve_headers(headers, settings)

        assert result == {
            "X-Jira-Auth": "Bearer jira-tok",
            "X-Slack-Auth": "Bearer slack-tok",
        }

    def test_original_headers_dict_not_mutated(self) -> None:
        """_resolve_headers returns a new dict and does not modify the input."""
        settings = AppSettings(OPENAI_API_KEY="sk-test", MCP_JIRA_TOKEN="tok")
        original = {"Authorization": "Bearer ${MCP_JIRA_TOKEN}"}
        original_copy = dict(original)

        AgentFactory._resolve_headers(original, settings)

        assert original == original_copy


# ===========================================================================
# AgentFactory.build — MCP integration
# ===========================================================================


class TestAgentFactoryBuildMcp:
    """Tests for AgentFactory.build() with MCP server configurations."""

    def test_build_resolves_mcps_and_passes_to_agent(self) -> None:
        """build() resolves mcps and passes a non-empty list to crewai.Agent."""
        from crewai.mcp import MCPServerHTTP

        defn = _make_http_definition(url="http://127.0.0.1:8100/mcp")
        mcp_registry = {"jira": defn}
        cfg = _make_agent_config(mcps=["jira"])
        settings = _make_settings()

        with (
            patch("src.agents.llm_factory.LLM"),
            patch("src.agents.factory.Agent") as MockAgent,
        ):
            AgentFactory.build(cfg, settings, mcp_servers=mcp_registry)

        kwargs = MockAgent.call_args.kwargs
        assert kwargs["mcps"] is not None
        assert len(kwargs["mcps"]) == 1
        assert isinstance(kwargs["mcps"][0], MCPServerHTTP)

    def test_build_mcps_none_when_empty_registry(self) -> None:
        """build() passes mcps=None when the mcp_servers dict is empty."""
        cfg = _make_agent_config(mcps=["jira"])
        settings = _make_settings()

        with (
            patch("src.agents.llm_factory.LLM"),
            patch("src.agents.factory.Agent") as MockAgent,
        ):
            AgentFactory.build(cfg, settings, mcp_servers={})

        kwargs = MockAgent.call_args.kwargs
        assert kwargs["mcps"] is None

    def test_build_multiple_mcps_all_passed(self) -> None:
        """build() passes all resolved MCP configs to crewai.Agent."""
        mcp_registry = {
            "jira": _make_http_definition(url="http://127.0.0.1:8100/mcp"),
            "slack": _make_http_definition(url="http://127.0.0.1:8103/mcp"),
            "github": _make_http_definition(url="http://127.0.0.1:8102/mcp"),
        }
        cfg = _make_agent_config(mcps=["jira", "slack", "github"])
        settings = _make_settings()

        with (
            patch("src.agents.llm_factory.LLM"),
            patch("src.agents.factory.Agent") as MockAgent,
        ):
            AgentFactory.build(cfg, settings, mcp_servers=mcp_registry)

        kwargs = MockAgent.call_args.kwargs
        assert len(kwargs["mcps"]) == 3

    def test_build_override_ref_passed_correctly(self) -> None:
        """build() correctly handles MCPServerRef override entries in mcps."""
        from crewai.mcp import MCPServerHTTP

        defn = _make_http_definition(tool_filter=["create_issue", "search_issues", "update_issue"])
        mcp_registry = {"jira": defn}
        ref = MCPServerRef(server="jira", tool_filter=["search_issues"])
        cfg = _make_agent_config(mcps=[ref])
        settings = _make_settings()

        with (
            patch("src.agents.llm_factory.LLM"),
            patch("src.agents.factory.Agent") as MockAgent,
        ):
            AgentFactory.build(cfg, settings, mcp_servers=mcp_registry)

        kwargs = MockAgent.call_args.kwargs
        assert kwargs["mcps"] is not None
        assert len(kwargs["mcps"]) == 1
        assert isinstance(kwargs["mcps"][0], MCPServerHTTP)
        # Overridden filter applied — filter object is set
        assert kwargs["mcps"][0].tool_filter is not None

