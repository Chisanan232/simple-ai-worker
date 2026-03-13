"""
Unit tests for :class:`src.agents.registry.AgentRegistry` and
:func:`src.agents.registry.build_registry`.

crewai.Agent and crewai.LLM are patched wherever needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from src.agents.registry import AgentRegistry, build_registry
from src.config.agent_config import (
    AgentConfig,
    AgentTeamConfig,
    LLMConfig,
    LLMOptions,
    MCPServerDefinition,
)
from src.config.settings import AppSettings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_agent(role: str = "Planner") -> MagicMock:
    agent = MagicMock()
    agent.role = role
    return agent


def _make_agent_cfg(agent_id: str, role: str = "Planner", mcps: list | None = None) -> AgentConfig:
    return AgentConfig(
        id=agent_id,
        role=role,
        goal="Do stuff.",
        backstory="A person.",
        llm=LLMConfig(provider="openai", model="gpt-4o", options=LLMOptions()),
        mcps=mcps or [],
    )


def _make_team(*agent_ids: str) -> AgentTeamConfig:
    return AgentTeamConfig(agents=[_make_agent_cfg(aid) for aid in agent_ids])


def _make_team_with_mcps(mcp_servers: dict, agent_mcps: dict[str, list]) -> AgentTeamConfig:
    """Build an AgentTeamConfig with mcp_servers registry and per-agent mcps."""
    agents = [
        _make_agent_cfg(aid, mcps=agent_mcps.get(aid, []))
        for aid in agent_mcps
    ]
    return AgentTeamConfig(
        mcp_servers={k: MCPServerDefinition(**v) for k, v in mcp_servers.items()},
        agents=agents,
    )


def _make_settings() -> AppSettings:
    return AppSettings(OPENAI_API_KEY="sk-test")


# ===========================================================================
# AgentRegistry — basic container operations
# ===========================================================================


class TestAgentRegistryBasics:
    def test_empty_registry_has_len_zero(self) -> None:
        r = AgentRegistry()
        assert len(r) == 0

    def test_register_increases_len(self) -> None:
        r = AgentRegistry()
        r.register("planner", _mock_agent())
        assert len(r) == 1

    def test_register_two_agents(self) -> None:
        r = AgentRegistry()
        r.register("planner", _mock_agent("Planner"))
        r.register("dev_lead", _mock_agent("Dev Lead"))
        assert len(r) == 2

    def test_getitem_returns_correct_agent(self) -> None:
        r = AgentRegistry()
        agent = _mock_agent("Planner")
        r.register("planner", agent)
        assert r["planner"] is agent

    def test_getitem_missing_raises_key_error(self) -> None:
        r = AgentRegistry()
        with pytest.raises(KeyError, match="planner"):
            _ = r["planner"]

    def test_getitem_error_message_lists_available_ids(self) -> None:
        r = AgentRegistry()
        r.register("dev_lead", _mock_agent())
        with pytest.raises(KeyError) as exc_info:
            _ = r["planner"]
        assert "dev_lead" in str(exc_info.value)

    def test_get_returns_agent_when_present(self) -> None:
        r = AgentRegistry()
        agent = _mock_agent()
        r.register("planner", agent)
        assert r.get("planner") is agent

    def test_get_returns_none_when_absent(self) -> None:
        r = AgentRegistry()
        assert r.get("planner") is None

    def test_get_returns_default_when_absent(self) -> None:
        r = AgentRegistry()
        sentinel = _mock_agent()
        assert r.get("planner", sentinel) is sentinel

    def test_contains_true_when_registered(self) -> None:
        r = AgentRegistry()
        r.register("planner", _mock_agent())
        assert "planner" in r

    def test_contains_false_when_not_registered(self) -> None:
        r = AgentRegistry()
        assert "planner" not in r

    def test_iter_yields_ids(self) -> None:
        r = AgentRegistry()
        r.register("planner", _mock_agent())
        r.register("dev_lead", _mock_agent())
        ids = list(r)
        assert "planner" in ids
        assert "dev_lead" in ids

    def test_items_yields_id_agent_pairs(self) -> None:
        r = AgentRegistry()
        a1, a2 = _mock_agent("P"), _mock_agent("D")
        r.register("planner", a1)
        r.register("dev_lead", a2)
        pairs = dict(r.items())
        assert pairs["planner"] is a1
        assert pairs["dev_lead"] is a2

    def test_agent_ids_returns_sorted_list(self) -> None:
        r = AgentRegistry()
        r.register("dev_lead", _mock_agent())
        r.register("planner", _mock_agent())
        r.register("dev_agent", _mock_agent())
        assert r.agent_ids() == ["dev_agent", "dev_lead", "planner"]

    def test_repr_contains_class_name(self) -> None:
        r = AgentRegistry()
        assert "AgentRegistry" in repr(r)

    def test_repr_contains_agent_ids(self) -> None:
        r = AgentRegistry()
        r.register("planner", _mock_agent())
        assert "planner" in repr(r)

    def test_register_overwrites_existing_id(self) -> None:
        r = AgentRegistry()
        a1 = _mock_agent("Old")
        a2 = _mock_agent("New")
        r.register("planner", a1)
        r.register("planner", a2)
        assert r["planner"] is a2
        assert len(r) == 1


# ===========================================================================
# build_registry
# ===========================================================================


class TestBuildRegistry:
    """Tests for the build_registry() convenience function."""

    def test_returns_agent_registry_instance(self) -> None:
        team = _make_team("planner")
        settings = _make_settings()

        with (
            patch("src.agents.llm_factory.LLM"),
            patch("src.agents.factory.Agent", return_value=_mock_agent()),
        ):
            registry = build_registry(team, settings)

        assert isinstance(registry, AgentRegistry)

    def test_registry_len_matches_team_agent_count(self) -> None:
        team = _make_team("planner", "dev_lead", "dev_agent")
        settings = _make_settings()

        with (
            patch("src.agents.llm_factory.LLM"),
            patch("src.agents.factory.Agent", return_value=_mock_agent()),
        ):
            registry = build_registry(team, settings)

        assert len(registry) == 3

    def test_all_agent_ids_registered(self) -> None:
        team = _make_team("planner", "dev_lead", "dev_agent")
        settings = _make_settings()

        with (
            patch("src.agents.llm_factory.LLM"),
            patch("src.agents.factory.Agent", return_value=_mock_agent()),
        ):
            registry = build_registry(team, settings)

        assert "planner" in registry
        assert "dev_lead" in registry
        assert "dev_agent" in registry

    def test_single_agent_team(self) -> None:
        team = _make_team("planner")
        settings = _make_settings()

        with (
            patch("src.agents.llm_factory.LLM"),
            patch("src.agents.factory.Agent", return_value=_mock_agent()),
        ):
            registry = build_registry(team, settings)

        assert len(registry) == 1
        assert "planner" in registry

    def test_agent_factory_called_for_each_agent(self) -> None:
        """AgentFactory.build must be called once per agent in the team."""
        team = _make_team("planner", "dev_lead")
        settings = _make_settings()

        with (
            patch("src.agents.llm_factory.LLM"),
            patch("src.agents.factory.Agent") as MockAgent,
        ):
            MockAgent.return_value = _mock_agent()
            build_registry(team, settings)

        assert MockAgent.call_count == 2


# ===========================================================================
# build_registry — MCP server registry propagation
# ===========================================================================


class TestBuildRegistryMcp:
    """Verify build_registry forwards mcp_servers to AgentFactory.build()."""

    def test_build_registry_passes_mcp_servers_to_factory(self) -> None:
        """build_registry forwards team_config.mcp_servers to every AgentFactory.build call."""
        mcp_defs = {
            "jira": {"type": "http", "url": "http://127.0.0.1:8100/mcp"},
        }
        team = _make_team_with_mcps(
            mcp_servers=mcp_defs,
            agent_mcps={"planner": ["jira"]},
        )
        settings = _make_settings()

        with (
            patch("src.agents.llm_factory.LLM"),
            patch("src.agents.factory.Agent", return_value=_mock_agent()),
            patch("src.agents.factory.AgentFactory.build", return_value=_mock_agent()) as mock_build,
        ):
            build_registry(team, settings)

        mock_build.assert_called_once()
        _, kwargs = mock_build.call_args
        assert "mcp_servers" in kwargs
        assert "jira" in kwargs["mcp_servers"]

    def test_build_registry_passes_same_mcp_servers_to_all_agents(self) -> None:
        """The same mcp_servers dict is passed to every AgentFactory.build call."""
        mcp_defs = {
            "jira": {"type": "http", "url": "http://127.0.0.1:8100/mcp"},
            "slack": {"type": "http", "url": "http://127.0.0.1:8103/mcp"},
        }
        team = _make_team_with_mcps(
            mcp_servers=mcp_defs,
            agent_mcps={
                "planner": ["jira", "slack"],
                "dev_lead": ["jira", "slack"],
            },
        )
        settings = _make_settings()

        with (
            patch("src.agents.llm_factory.LLM"),
            patch("src.agents.factory.Agent", return_value=_mock_agent()),
            patch("src.agents.factory.AgentFactory.build", return_value=_mock_agent()) as mock_build,
        ):
            build_registry(team, settings)

        assert mock_build.call_count == 2
        for c in mock_build.call_args_list:
            _, kwargs = c
            assert kwargs["mcp_servers"] is team.mcp_servers

    def test_build_registry_empty_mcp_servers_still_works(self) -> None:
        """build_registry works normally when mcp_servers is empty."""
        team = _make_team("planner")
        settings = _make_settings()

        with (
            patch("src.agents.llm_factory.LLM"),
            patch("src.agents.factory.Agent", return_value=_mock_agent()),
        ):
            registry = build_registry(team, settings)

        assert "planner" in registry

