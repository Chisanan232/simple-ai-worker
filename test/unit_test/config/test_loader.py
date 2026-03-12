"""
Unit tests for :mod:`src.config.loader`.

Tests :func:`load_agent_config` and :class:`AgentConfigLoadError`.

Strategy
--------
- Every test creates a temporary file (or uses a non-existent path) so
  the tests are fully hermetic — they never depend on the real
  ``config/agents.yaml`` on disk.
- ``tmp_path`` (pytest built-in fixture) provides an isolated temp directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.agent_config import AgentTeamConfig
from src.config.loader import AgentConfigLoadError, load_agent_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_YAML = """\
process: sequential
agents:
  - id: planner
    role: Product Planner
    goal: Plan the product.
    backstory: You are a planner.
    llm:
      provider: openai
      model: gpt-4o
"""

_MINIMAL_YAML_NO_PROCESS = """\
agents:
  - id: dev_lead
    role: Dev Lead
    goal: Break down epics.
    backstory: You are a lead.
    llm:
      provider: anthropic
      model: claude-3-5-sonnet-latest
"""

_THREE_AGENT_YAML = """\
process: sequential
agents:
  - id: planner
    role: Planner
    goal: Plan.
    backstory: Planner person.
    llm:
      provider: openai
      model: gpt-4o
  - id: dev_lead
    role: Dev Lead
    goal: Lead.
    backstory: Lead person.
    llm:
      provider: openai
      model: gpt-4o
  - id: dev_agent
    role: Developer
    goal: Develop.
    backstory: Dev person.
    llm:
      provider: openai
      model: gpt-4o
"""


def _write(tmp_path: Path, content: str, filename: str = "agents.yaml") -> Path:
    """Write content to a temp file and return its Path."""
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return p


# ===========================================================================
# Happy-path tests
# ===========================================================================

class TestLoadAgentConfigHappyPath:
    def test_returns_agent_team_config(self, tmp_path: Path) -> None:
        """load_agent_config() must return an AgentTeamConfig instance."""
        path = _write(tmp_path, _VALID_YAML)
        config = load_agent_config(path)
        assert isinstance(config, AgentTeamConfig)

    def test_process_is_sequential(self, tmp_path: Path) -> None:
        """Explicitly set process=sequential must be preserved."""
        path = _write(tmp_path, _VALID_YAML)
        config = load_agent_config(path)
        assert config.process == "sequential"

    def test_process_defaults_to_sequential_when_omitted(self, tmp_path: Path) -> None:
        """Omitting process must default to sequential."""
        path = _write(tmp_path, _MINIMAL_YAML_NO_PROCESS)
        config = load_agent_config(path)
        assert config.process == "sequential"

    def test_single_agent_loaded(self, tmp_path: Path) -> None:
        """A YAML with one agent must produce exactly one AgentConfig."""
        path = _write(tmp_path, _VALID_YAML)
        config = load_agent_config(path)
        assert len(config.agents) == 1
        assert config.agents[0].id == "planner"

    def test_three_agents_loaded(self, tmp_path: Path) -> None:
        """A YAML with three agents must produce three AgentConfig entries."""
        path = _write(tmp_path, _THREE_AGENT_YAML)
        config = load_agent_config(path)
        assert len(config.agents) == 3
        assert [a.id for a in config.agents] == ["planner", "dev_lead", "dev_agent"]

    def test_agent_fields_populated(self, tmp_path: Path) -> None:
        """All AgentConfig fields must be populated from the YAML."""
        path = _write(tmp_path, _VALID_YAML)
        agent = load_agent_config(path).agents[0]
        assert agent.id == "planner"
        assert agent.role == "Product Planner"
        assert "Plan" in agent.goal
        assert agent.llm.provider == "openai"
        assert agent.llm.model == "gpt-4o"

    def test_accepts_str_path(self, tmp_path: Path) -> None:
        """load_agent_config() must accept a plain str path as well as Path."""
        path = _write(tmp_path, _VALID_YAML)
        config = load_agent_config(str(path))
        assert isinstance(config, AgentTeamConfig)

    def test_real_agents_yaml_is_valid(self) -> None:
        """config/agents.yaml (ship file) must load without error."""
        config = load_agent_config("config/agents.yaml")
        assert len(config.agents) >= 1

    def test_real_agents_example_yaml_is_valid(self) -> None:
        """config/agents.example.yaml (ship file) must load without error."""
        config = load_agent_config("config/agents.example.yaml")
        assert len(config.agents) >= 1

    def test_apps_empty_list_loaded(self, tmp_path: Path) -> None:
        """An empty apps list must be loaded as an empty Python list."""
        yaml_text = (
            _VALID_YAML.rstrip() + "\n    apps: []\n"
        )
        path = _write(tmp_path, yaml_text)
        config = load_agent_config(path)
        assert config.agents[0].apps == []

    def test_apps_populated_loaded(self, tmp_path: Path) -> None:
        """A populated apps list must be loaded correctly."""
        yaml_text = (
            _VALID_YAML.rstrip()
            + "\n    apps:\n      - jira/create_issue\n      - slack/reply_to_thread\n"
        )
        path = _write(tmp_path, yaml_text)
        config = load_agent_config(path)
        assert config.agents[0].apps == ["jira/create_issue", "slack/reply_to_thread"]


# ===========================================================================
# Error-path tests
# ===========================================================================

class TestLoadAgentConfigErrors:
    def test_missing_file_raises_load_error(self, tmp_path: Path) -> None:
        """A non-existent path must raise AgentConfigLoadError."""
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(AgentConfigLoadError):
            load_agent_config(missing)

    def test_missing_file_error_has_correct_path(self, tmp_path: Path) -> None:
        """AgentConfigLoadError.path must match the requested path."""
        missing = tmp_path / "missing.yaml"
        with pytest.raises(AgentConfigLoadError) as exc_info:
            load_agent_config(missing)
        assert exc_info.value.path == missing

    def test_missing_file_error_chained_oserror(self, tmp_path: Path) -> None:
        """The original OSError must be chained as __cause__."""
        missing = tmp_path / "missing.yaml"
        with pytest.raises(AgentConfigLoadError) as exc_info:
            load_agent_config(missing)
        assert isinstance(exc_info.value.__cause__, OSError)

    def test_invalid_yaml_syntax_raises_load_error(self, tmp_path: Path) -> None:
        """A YAML file with syntax errors must raise AgentConfigLoadError."""
        path = _write(tmp_path, "agents: [\n  - id: oops\n  bad: {{{")
        with pytest.raises(AgentConfigLoadError):
            load_agent_config(path)

    def test_invalid_yaml_error_chained_yaml_error(self, tmp_path: Path) -> None:
        """The original yaml.YAMLError must be chained as __cause__."""
        import yaml

        path = _write(tmp_path, "agents: [\n  bad: {{{")
        with pytest.raises(AgentConfigLoadError) as exc_info:
            load_agent_config(path)
        assert isinstance(exc_info.value.__cause__, yaml.YAMLError)

    def test_yaml_list_at_root_raises_load_error(self, tmp_path: Path) -> None:
        """A YAML file whose root is a list must raise AgentConfigLoadError."""
        path = _write(tmp_path, "- id: planner\n- id: dev_lead\n")
        with pytest.raises(AgentConfigLoadError):
            load_agent_config(path)

    def test_schema_validation_failure_raises_load_error(self, tmp_path: Path) -> None:
        """A syntactically valid YAML that fails Pydantic schema must raise."""
        bad_yaml = """\
agents:
  - id: planner
    role: Planner
    goal: Plan.
    backstory: A planner.
    llm:
      provider: INVALID_PROVIDER
      model: gpt-4o
"""
        path = _write(tmp_path, bad_yaml)
        with pytest.raises(AgentConfigLoadError):
            load_agent_config(path)

    def test_schema_validation_error_chained_pydantic_error(
        self, tmp_path: Path
    ) -> None:
        """The original pydantic ValidationError must be chained as __cause__."""
        from pydantic import ValidationError

        bad_yaml = """\
agents:
  - id: planner
    role: Planner
    goal: Plan.
    backstory: A planner.
    llm:
      provider: INVALID_PROVIDER
      model: gpt-4o
"""
        path = _write(tmp_path, bad_yaml)
        with pytest.raises(AgentConfigLoadError) as exc_info:
            load_agent_config(path)
        assert isinstance(exc_info.value.__cause__, ValidationError)

    def test_empty_agents_list_raises_load_error(self, tmp_path: Path) -> None:
        """An agents: [] YAML must raise AgentConfigLoadError (min_length=1)."""
        path = _write(tmp_path, "agents: []\n")
        with pytest.raises(AgentConfigLoadError):
            load_agent_config(path)

    def test_duplicate_agent_ids_raises_load_error(self, tmp_path: Path) -> None:
        """A YAML with duplicate agent IDs must raise AgentConfigLoadError."""
        dup_yaml = """\
agents:
  - id: planner
    role: Planner
    goal: Plan.
    backstory: A planner.
    llm:
      provider: openai
      model: gpt-4o
  - id: planner
    role: Planner Again
    goal: Plan again.
    backstory: Another planner.
    llm:
      provider: openai
      model: gpt-4o
"""
        path = _write(tmp_path, dup_yaml)
        with pytest.raises(AgentConfigLoadError):
            load_agent_config(path)


# ===========================================================================
# AgentConfigLoadError
# ===========================================================================

class TestAgentConfigLoadError:
    def test_error_str_contains_path(self) -> None:
        """str(AgentConfigLoadError) must contain the path."""
        err = AgentConfigLoadError("/some/path/agents.yaml", "something went wrong")
        assert "/some/path/agents.yaml" in str(err)

    def test_error_str_contains_message(self) -> None:
        """str(AgentConfigLoadError) must contain the message."""
        err = AgentConfigLoadError("/path", "validation failed")
        assert "validation failed" in str(err)

    def test_error_path_attribute(self) -> None:
        """AgentConfigLoadError.path must be a Path object."""
        err = AgentConfigLoadError("/some/path.yaml", "oops")
        assert err.path == Path("/some/path.yaml")

    def test_error_message_attribute(self) -> None:
        """AgentConfigLoadError.message must match what was passed."""
        err = AgentConfigLoadError("/p", "my message")
        assert err.message == "my message"

