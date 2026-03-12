"""
Unit tests for :class:`src.agents.factory.AgentFactory`.

crewai.Agent and crewai.LLM are both patched so no live API calls occur.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.agents.factory import AgentFactory
from src.config.agent_config import AgentConfig, LLMConfig, LLMOptions
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
    apps: list = None,  # type: ignore[assignment]
    allow_delegation: bool = False,
    verbose: bool = False,
) -> AgentConfig:
    return AgentConfig(
        id=agent_id,
        role=role,
        goal=goal,
        backstory=backstory,
        llm=LLMConfig(
            provider=provider,  # type: ignore[arg-type]
            model=model,
            options=LLMOptions(),
        ),
        apps=apps or [],
        allow_delegation=allow_delegation,
        verbose=verbose,
    )


def _make_settings(openai_key: str = "sk-test") -> AppSettings:
    return AppSettings(OPENAI_API_KEY=openai_key)  # type: ignore[arg-type]


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

    def test_build_passes_empty_apps(self) -> None:
        """build() must pass an empty list when apps is empty."""
        cfg = _make_agent_config(apps=[])
        settings = _make_settings()

        with (
            patch("src.agents.llm_factory.LLM"),
            patch("src.agents.factory.Agent") as MockAgent,
        ):
            AgentFactory.build(cfg, settings)

        kwargs = MockAgent.call_args.kwargs
        assert kwargs["apps"] == []

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

