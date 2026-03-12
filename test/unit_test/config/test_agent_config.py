from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.agent_config import AgentConfig, AgentTeamConfig, LLMConfig, LLMOptions


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
            LLMConfig(provider="gemini", model="gemini-pro")  # type: ignore[arg-type]

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
            options={"temperature": 0.1},  # type: ignore[arg-type]
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
        config = AgentTeamConfig.model_validate(
            _minimal_team("planner", "dev_lead", "dev_agent")
        )
        assert len(config.agents) == 3

    def test_empty_agents_raises(self) -> None:
        with pytest.raises(ValidationError):
            AgentTeamConfig.model_validate({"agents": []})

    def test_missing_agents_raises(self) -> None:
        with pytest.raises(ValidationError):
            AgentTeamConfig.model_validate({"process": "sequential"})

    def test_duplicate_agent_ids_raise(self) -> None:
        data: dict = {
            "agents": [_minimal_agent("planner"), _minimal_agent("planner")]
        }
        with pytest.raises(ValidationError, match="duplicate"):
            AgentTeamConfig.model_validate(data)

    def test_duplicate_id_error_names_the_offender(self) -> None:
        data: dict = {
            "agents": [_minimal_agent("dev_agent"), _minimal_agent("dev_agent")]
        }
        with pytest.raises(ValidationError) as exc_info:
            AgentTeamConfig.model_validate(data)
        assert "dev_agent" in str(exc_info.value)

    def test_agents_order_preserved(self) -> None:
        config = AgentTeamConfig.model_validate(
            _minimal_team("planner", "dev_lead", "dev_agent")
        )
        assert [a.id for a in config.agents] == ["planner", "dev_lead", "dev_agent"]
