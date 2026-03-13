from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.agent_config import (
    AgentConfig,
    AgentTeamConfig,
    LLMConfig,
    LLMOptions,
    MCPServerDefinition,
    MCPServerRef,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_llm() -> dict:
    return {"provider": "openai", "model": "gpt-4o"}


def _minimal_agent(agent_id: str = "planner") -> dict:
    return {
        "id": agent_id,
        "role": "Product Planner",
        "goal": "Plan the product.",
        "backstory": "You are a planner.",
        "llm": _minimal_llm(),
    }


def _minimal_team(*agent_ids: str) -> dict:
    ids = agent_ids if agent_ids else ("planner",)
    return {"agents": [_minimal_agent(aid) for aid in ids]}


# ===========================================================================
# LLMOptions
# ===========================================================================


class TestLLMOptions:
    def test_defaults(self) -> None:
        opts = LLMOptions()
        assert opts.temperature == 0.7
        assert opts.max_tokens == 4096
        assert opts.top_p == 1.0
        assert opts.timeout == 120

    def test_custom_values(self) -> None:
        opts = LLMOptions(temperature=0.3, max_tokens=2048, top_p=0.9, timeout=60)
        assert opts.temperature == 0.3
        assert opts.max_tokens == 2048
        assert opts.top_p == 0.9
        assert opts.timeout == 60

    def test_temperature_lower_bound(self) -> None:
        assert LLMOptions(temperature=0.0).temperature == 0.0

    def test_temperature_upper_bound(self) -> None:
        assert LLMOptions(temperature=2.0).temperature == 2.0

    def test_temperature_above_upper_bound_raises(self) -> None:
        with pytest.raises(ValidationError):
            LLMOptions(temperature=2.1)

    def test_temperature_below_lower_bound_raises(self) -> None:
        with pytest.raises(ValidationError):
            LLMOptions(temperature=-0.1)

    def test_max_tokens_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            LLMOptions(max_tokens=0)

    def test_top_p_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            LLMOptions(top_p=0.0)

    def test_top_p_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            LLMOptions(top_p=1.1)

    def test_timeout_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            LLMOptions(timeout=0)


# ===========================================================================
# LLMConfig
# ===========================================================================


class TestLLMConfig:
    def test_valid_openai(self) -> None:
        cfg = LLMConfig(provider="openai", model="gpt-4o")
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4o"

    def test_valid_anthropic(self) -> None:
        cfg = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-latest")
        assert cfg.provider == "anthropic"

    def test_invalid_provider_raises(self) -> None:
        with pytest.raises(ValidationError):
            LLMConfig(provider="gemini", model="gemini-pro")

    def test_blank_model_raises(self) -> None:
        with pytest.raises(ValidationError):
            LLMConfig(provider="openai", model="   ")

    def test_empty_model_raises(self) -> None:
        with pytest.raises(ValidationError):
            LLMConfig(provider="openai", model="")

    def test_options_default_factory(self) -> None:
        cfg = LLMConfig(provider="openai", model="gpt-4o")
        assert isinstance(cfg.options, LLMOptions)
        assert cfg.options.temperature == 0.7

    def test_options_partial_override(self) -> None:
        cfg = LLMConfig(
            provider="openai",
            model="gpt-4o",
            options={"temperature": 0.1},
        )
        assert cfg.options.temperature == 0.1
        assert cfg.options.max_tokens == 4096


# ===========================================================================
# AgentConfig
# ===========================================================================


class TestAgentConfig:
    def test_minimal_valid_agent(self) -> None:
        agent = AgentConfig.model_validate(_minimal_agent())
        assert agent.id == "planner"
        assert agent.allow_delegation is False
        assert agent.verbose is False
        assert agent.apps == []

    def test_blank_id_raises(self) -> None:
        data = _minimal_agent()
        data["id"] = "   "
        with pytest.raises(ValidationError):
            AgentConfig.model_validate(data)

    def test_empty_id_raises(self) -> None:
        data = _minimal_agent()
        data["id"] = ""
        with pytest.raises(ValidationError):
            AgentConfig.model_validate(data)

    def test_apps_list_populated(self) -> None:
        data = _minimal_agent()
        data["apps"] = ["jira/create_issue", "slack/reply_to_thread"]
        agent = AgentConfig.model_validate(data)
        assert agent.apps == ["jira/create_issue", "slack/reply_to_thread"]

    def test_apps_blank_item_raises(self) -> None:
        data = _minimal_agent()
        data["apps"] = ["jira/create_issue", "  "]
        with pytest.raises(ValidationError):
            AgentConfig.model_validate(data)

    def test_allow_delegation_true(self) -> None:
        data = _minimal_agent()
        data["allow_delegation"] = True
        assert AgentConfig.model_validate(data).allow_delegation is True

    def test_verbose_true(self) -> None:
        data = _minimal_agent()
        data["verbose"] = True
        assert AgentConfig.model_validate(data).verbose is True

    def test_missing_role_raises(self) -> None:
        data = _minimal_agent()
        del data["role"]
        with pytest.raises(ValidationError):
            AgentConfig.model_validate(data)

    def test_missing_llm_raises(self) -> None:
        data = _minimal_agent()
        del data["llm"]
        with pytest.raises(ValidationError):
            AgentConfig.model_validate(data)


# ===========================================================================
# AgentTeamConfig
# ===========================================================================


class TestAgentTeamConfig:
    def test_default_process_is_sequential(self) -> None:
        config = AgentTeamConfig.model_validate(_minimal_team())
        assert config.process == "sequential"

    def test_hierarchical_process(self) -> None:
        data = _minimal_team()
        data["process"] = "hierarchical"
        assert AgentTeamConfig.model_validate(data).process == "hierarchical"

    def test_invalid_process_raises(self) -> None:
        data = _minimal_team()
        data["process"] = "parallel"
        with pytest.raises(ValidationError):
            AgentTeamConfig.model_validate(data)

    def test_single_agent(self) -> None:
        config = AgentTeamConfig.model_validate(_minimal_team("planner"))
        assert len(config.agents) == 1

    def test_multiple_unique_agents(self) -> None:
        config = AgentTeamConfig.model_validate(_minimal_team("planner", "dev_lead", "dev_agent"))
        assert len(config.agents) == 3

    def test_empty_agents_raises(self) -> None:
        with pytest.raises(ValidationError):
            AgentTeamConfig.model_validate({"agents": []})

    def test_missing_agents_raises(self) -> None:
        with pytest.raises(ValidationError):
            AgentTeamConfig.model_validate({"process": "sequential"})

    def test_duplicate_agent_ids_raise(self) -> None:
        data: dict = {"agents": [_minimal_agent("planner"), _minimal_agent("planner")]}
        with pytest.raises(ValidationError, match="duplicate"):
            AgentTeamConfig.model_validate(data)

    def test_duplicate_id_error_names_the_offender(self) -> None:
        data: dict = {"agents": [_minimal_agent("dev_agent"), _minimal_agent("dev_agent")]}
        with pytest.raises(ValidationError) as exc_info:
            AgentTeamConfig.model_validate(data)
        assert "dev_agent" in str(exc_info.value)

    def test_agents_order_preserved(self) -> None:
        config = AgentTeamConfig.model_validate(_minimal_team("planner", "dev_lead", "dev_agent"))
        assert [a.id for a in config.agents] == ["planner", "dev_lead", "dev_agent"]


# ===========================================================================
# MCPServerDefinition
# ===========================================================================


class TestMCPServerDefinition:
    """Tests for MCPServerDefinition Pydantic model validation."""

    def test_valid_http_definition(self) -> None:
        defn = MCPServerDefinition(type="http", url="http://127.0.0.1:8100/mcp")
        assert defn.type == "http"
        assert defn.url == "http://127.0.0.1:8100/mcp"
        assert defn.streamable is True
        assert defn.tool_filter is None
        assert defn.cache_tools_list is False

    def test_valid_sse_definition(self) -> None:
        defn = MCPServerDefinition(type="sse", url="http://127.0.0.1:8104/mcp/sse")
        assert defn.type == "sse"
        assert defn.url == "http://127.0.0.1:8104/mcp/sse"

    def test_valid_stdio_definition(self) -> None:
        defn = MCPServerDefinition(type="stdio", command="python", args=["server.py"])
        assert defn.type == "stdio"
        assert defn.command == "python"
        assert defn.args == ["server.py"]

    def test_http_with_headers(self) -> None:
        defn = MCPServerDefinition(
            type="http",
            url="http://127.0.0.1:8100/mcp",
            headers={"Authorization": "Bearer tok"},
        )
        assert defn.headers == {"Authorization": "Bearer tok"}

    def test_http_with_tool_filter(self) -> None:
        defn = MCPServerDefinition(
            type="http",
            url="http://127.0.0.1:8100/mcp",
            tool_filter=["create_issue", "search_issues"],
        )
        assert defn.tool_filter == ["create_issue", "search_issues"]

    def test_http_with_cache_tools_list(self) -> None:
        defn = MCPServerDefinition(type="http", url="http://127.0.0.1:8100/mcp", cache_tools_list=True)
        assert defn.cache_tools_list is True

    def test_http_missing_url_raises(self) -> None:
        """HTTP type without url must raise a validation error."""
        with pytest.raises(ValidationError, match="url"):
            MCPServerDefinition(type="http")

    def test_sse_missing_url_raises(self) -> None:
        """SSE type without url must raise a validation error."""
        with pytest.raises(ValidationError, match="url"):
            MCPServerDefinition(type="sse")

    def test_stdio_missing_command_raises(self) -> None:
        """Stdio type without command must raise a validation error."""
        with pytest.raises(ValidationError, match="command"):
            MCPServerDefinition(type="stdio")

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            MCPServerDefinition(type="grpc", url="http://x")

    def test_stdio_args_default_empty_list(self) -> None:
        defn = MCPServerDefinition(type="stdio", command="python")
        assert defn.args == []

    def test_stdio_env_none_by_default(self) -> None:
        defn = MCPServerDefinition(type="stdio", command="python")
        assert defn.env is None


# ===========================================================================
# MCPServerRef
# ===========================================================================


class TestMCPServerRef:
    """Tests for MCPServerRef Pydantic model validation."""

    def test_plain_server_ref(self) -> None:
        ref = MCPServerRef(server="jira")
        assert ref.server == "jira"
        assert ref.tool_filter is None

    def test_override_ref_with_tool_filter(self) -> None:
        ref = MCPServerRef(server="jira", tool_filter=["create_issue", "search_issues"])
        assert ref.server == "jira"
        assert ref.tool_filter == ["create_issue", "search_issues"]

    def test_blank_server_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            MCPServerRef(server="")

    def test_tool_filter_empty_list_allowed(self) -> None:
        ref = MCPServerRef(server="jira", tool_filter=[])
        assert ref.tool_filter == []


# ===========================================================================
# AgentConfig — mcps field
# ===========================================================================


class TestAgentConfigMcps:
    """Tests for AgentConfig.mcps field with plain string and override refs."""

    def _make_cfg(self, mcps: list) -> AgentConfig:
        return AgentConfig(
            id="planner",
            role="Product Planner",
            goal="Plan stuff.",
            backstory="You plan.",
            llm=LLMConfig(provider="openai", model="gpt-4o"),
            mcps=mcps,
        )

    def test_empty_mcps_default(self) -> None:
        cfg = self._make_cfg(mcps=[])
        assert cfg.mcps == []

    def test_plain_string_mcps(self) -> None:
        cfg = self._make_cfg(mcps=["jira", "slack"])
        assert cfg.mcps == ["jira", "slack"]

    def test_override_ref_mcps(self) -> None:
        ref = MCPServerRef(server="jira", tool_filter=["create_issue"])
        cfg = self._make_cfg(mcps=[ref])
        assert len(cfg.mcps) == 1
        assert isinstance(cfg.mcps[0], MCPServerRef)
        assert cfg.mcps[0].server == "jira"

    def test_mixed_plain_and_override_mcps(self) -> None:
        ref = MCPServerRef(server="jira", tool_filter=["search_issues"])
        cfg = self._make_cfg(mcps=[ref, "slack"])
        assert len(cfg.mcps) == 2
        assert isinstance(cfg.mcps[0], MCPServerRef)
        assert cfg.mcps[1] == "slack"

    def test_mcps_parsed_from_dict(self) -> None:
        """Override mapping syntax parses correctly from a raw dict (YAML form)."""
        cfg = AgentConfig.model_validate(
            {
                "id": "planner",
                "role": "Planner",
                "goal": "Plan.",
                "backstory": "You plan.",
                "llm": {"provider": "openai", "model": "gpt-4o"},
                "mcps": [
                    "slack",
                    {"server": "jira", "tool_filter": ["create_issue"]},
                ],
            }
        )
        assert len(cfg.mcps) == 2
        assert cfg.mcps[0] == "slack"
        assert isinstance(cfg.mcps[1], MCPServerRef)
        assert cfg.mcps[1].tool_filter == ["create_issue"]


# ===========================================================================
# AgentTeamConfig — mcp_servers registry
# ===========================================================================


class TestAgentTeamConfigMcpServers:
    """Tests for AgentTeamConfig.mcp_servers registry and cross-ref validator."""

    def _make_team_data(
        self,
        mcp_servers: dict | None = None,
        agent_mcps: list | None = None,
    ) -> dict:
        return {
            "mcp_servers": mcp_servers or {},
            "agents": [
                {
                    **_minimal_agent("planner"),
                    "mcps": agent_mcps or [],
                }
            ],
        }

    def test_empty_mcp_servers_is_valid(self) -> None:
        config = AgentTeamConfig.model_validate(self._make_team_data())
        assert config.mcp_servers == {}

    def test_single_http_server_registered(self) -> None:
        config = AgentTeamConfig.model_validate(
            self._make_team_data(
                mcp_servers={"jira": {"type": "http", "url": "http://127.0.0.1:8100/mcp"}},
            )
        )
        assert "jira" in config.mcp_servers
        assert isinstance(config.mcp_servers["jira"], MCPServerDefinition)

    def test_multiple_servers_registered(self) -> None:
        config = AgentTeamConfig.model_validate(
            self._make_team_data(
                mcp_servers={
                    "jira": {"type": "http", "url": "http://127.0.0.1:8100/mcp"},
                    "slack": {"type": "http", "url": "http://127.0.0.1:8103/mcp"},
                },
            )
        )
        assert len(config.mcp_servers) == 2

    def test_agent_plain_ref_resolves_to_known_server(self) -> None:
        config = AgentTeamConfig.model_validate(
            self._make_team_data(
                mcp_servers={"jira": {"type": "http", "url": "http://127.0.0.1:8100/mcp"}},
                agent_mcps=["jira"],
            )
        )
        assert config.agents[0].mcps == ["jira"]

    def test_agent_unknown_ref_raises_validation_error(self) -> None:
        """Referencing an MCP server ID that isn't in mcp_servers raises at load time."""
        with pytest.raises(ValidationError, match="unknown MCP server"):
            AgentTeamConfig.model_validate(
                self._make_team_data(
                    mcp_servers={"jira": {"type": "http", "url": "http://127.0.0.1:8100/mcp"}},
                    agent_mcps=["jirs"],  # typo
                )
            )

    def test_agent_unknown_ref_error_names_the_bad_id(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            AgentTeamConfig.model_validate(
                self._make_team_data(
                    mcp_servers={},
                    agent_mcps=["nonexistent"],
                )
            )
        assert "nonexistent" in str(exc_info.value)

    def test_agent_override_ref_resolves(self) -> None:
        config = AgentTeamConfig.model_validate(
            self._make_team_data(
                mcp_servers={"jira": {"type": "http", "url": "http://127.0.0.1:8100/mcp"}},
                agent_mcps=[{"server": "jira", "tool_filter": ["create_issue"]}],
            )
        )
        assert isinstance(config.agents[0].mcps[0], MCPServerRef)

    def test_agent_override_ref_unknown_server_raises(self) -> None:
        with pytest.raises(ValidationError, match="unknown MCP server"):
            AgentTeamConfig.model_validate(
                self._make_team_data(
                    mcp_servers={},
                    agent_mcps=[{"server": "clickup", "tool_filter": ["create_task"]}],
                )
            )

    def test_mcp_servers_default_to_empty_dict(self) -> None:
        """mcp_servers field is optional; defaults to {}."""
        config = AgentTeamConfig.model_validate(
            {
                "agents": [_minimal_agent("planner")],
            }
        )
        assert config.mcp_servers == {}

    def test_multiple_agents_multiple_refs_all_validated(self) -> None:
        """Cross-ref validator checks all agents, not just the first."""
        with pytest.raises(ValidationError, match="unknown MCP server"):
            AgentTeamConfig.model_validate(
                {
                    "mcp_servers": {
                        "jira": {"type": "http", "url": "http://127.0.0.1:8100/mcp"},
                    },
                    "agents": [
                        {**_minimal_agent("planner"), "mcps": ["jira"]},
                        {**_minimal_agent("dev_agent"), "mcps": ["github"]},  # github not registered
                    ],
                }
            )
