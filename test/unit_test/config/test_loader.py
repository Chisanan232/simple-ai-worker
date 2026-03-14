from __future__ import annotations

from pathlib import Path

import pytest

from src.config.agent_config import AgentTeamConfig, MCPServerDefinition, MCPServerRef
from src.config.loader import AgentConfigLoadError, load_agent_config
from src.config.settings import AppSettings

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

_MCP_YAML_WITH_REGISTRY = """\
process: sequential

mcp_servers:
  jira:
    type: http
    url: "http://127.0.0.1:8100/mcp"
    headers:
      Authorization: "Bearer tok-jira"
    tool_filter:
      - create_issue
      - search_issues
    cache_tools_list: true
  slack:
    type: http
    url: "http://127.0.0.1:8103/mcp"

agents:
  - id: planner
    role: Product Planner
    goal: Plan the product.
    backstory: You are a planner.
    llm:
      provider: openai
      model: gpt-4o
    mcps:
      - jira
      - slack
"""

_MCP_YAML_WITH_OVERRIDE = """\
process: sequential

mcp_servers:
  jira:
    type: http
    url: "http://127.0.0.1:8100/mcp"
    tool_filter:
      - create_issue
      - search_issues
      - update_issue

agents:
  - id: dev_lead
    role: Tech Lead
    goal: Lead.
    backstory: You lead.
    llm:
      provider: openai
      model: gpt-4o
    mcps:
      - server: jira
        tool_filter:
          - search_issues
          - update_issue
"""

_MCP_YAML_WITH_PLACEHOLDER = """\
process: sequential

mcp_servers:
  jira:
    type: http
    url: "http://127.0.0.1:8100/mcp"
    headers:
      Authorization: "Bearer ${MCP_JIRA_TOKEN}"

agents:
  - id: planner
    role: Planner
    goal: Plan.
    backstory: You plan.
    llm:
      provider: openai
      model: gpt-4o
    mcps:
      - jira
"""

_MCP_YAML_WITH_URL_PLACEHOLDER = """\
process: sequential

mcp_servers:
  jira:
    type: http
    url: "${MCP_JIRA_URL}"
    headers:
      Authorization: "Bearer ${MCP_JIRA_TOKEN}"

agents:
  - id: planner
    role: Planner
    goal: Plan.
    backstory: You plan.
    llm:
      provider: openai
      model: gpt-4o
    mcps:
      - jira
"""

_MCP_YAML_UNKNOWN_REF = """\
process: sequential

mcp_servers:
  jira:
    type: http
    url: "http://127.0.0.1:8100/mcp"

agents:
  - id: planner
    role: Planner
    goal: Plan.
    backstory: You plan.
    llm:
      provider: openai
      model: gpt-4o
    mcps:
      - clickup
"""

_REAL_FILE_MCP_ENVS = {
    "MCP_JIRA_URL": "http://127.0.0.1:8100/mcp",
    "MCP_CLICKUP_URL": "http://127.0.0.1:8101/mcp",
    "MCP_GITHUB_URL": "http://127.0.0.1:8102/mcp",
    "MCP_SLACK_URL": "http://127.0.0.1:8103/mcp",
}


def _write(tmp_path: Path, content: str, filename: str = "agents.yaml") -> Path:
    """Write content to a temp file and return its Path."""
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return p


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

    def test_real_agents_yaml_is_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """config/agents.yaml must load without error.

        Injects MCP URL env vars because agents.yaml now uses ${MCP_*_URL}
        placeholders so the same file works locally and inside Docker Compose.
        """
        for key, val in _REAL_FILE_MCP_ENVS.items():
            monkeypatch.setenv(key, val)
        config = load_agent_config("config/agents.yaml")
        assert len(config.agents) >= 1

    def test_real_agents_example_yaml_is_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """config/agents.example.yaml must load without error."""
        for key, val in _REAL_FILE_MCP_ENVS.items():
            monkeypatch.setenv(key, val)
        config = load_agent_config("config/agents.example.yaml")
        assert len(config.agents) >= 1

    def test_apps_empty_list_loaded(self, tmp_path: Path) -> None:
        """An empty apps list must be loaded as an empty Python list."""
        yaml_text = _VALID_YAML.rstrip() + "\n    apps: []\n"
        path = _write(tmp_path, yaml_text)
        config = load_agent_config(path)
        assert config.agents[0].apps == []

    def test_apps_populated_loaded(self, tmp_path: Path) -> None:
        """A populated apps list must be loaded correctly."""
        yaml_text = _VALID_YAML.rstrip() + "\n    apps:\n      - jira/create_issue\n      - slack/reply_to_thread\n"
        path = _write(tmp_path, yaml_text)
        config = load_agent_config(path)
        assert config.agents[0].apps == ["jira/create_issue", "slack/reply_to_thread"]


class TestLoadAgentConfigErrors:
    def test_missing_file_raises_load_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(AgentConfigLoadError):
            load_agent_config(missing)

    def test_missing_file_error_has_correct_path(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.yaml"
        with pytest.raises(AgentConfigLoadError) as exc_info:
            load_agent_config(missing)
        assert exc_info.value.path == missing

    def test_missing_file_error_chained_oserror(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.yaml"
        with pytest.raises(AgentConfigLoadError) as exc_info:
            load_agent_config(missing)
        assert isinstance(exc_info.value.__cause__, OSError)

    def test_invalid_yaml_syntax_raises_load_error(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "agents: [\n  - id: oops\n  bad: {{{")
        with pytest.raises(AgentConfigLoadError):
            load_agent_config(path)

    def test_invalid_yaml_error_chained_yaml_error(self, tmp_path: Path) -> None:
        import yaml

        path = _write(tmp_path, "agents: [\n  bad: {{{")
        with pytest.raises(AgentConfigLoadError) as exc_info:
            load_agent_config(path)
        assert isinstance(exc_info.value.__cause__, yaml.YAMLError)

    def test_yaml_list_at_root_raises_load_error(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "- id: planner\n- id: dev_lead\n")
        with pytest.raises(AgentConfigLoadError):
            load_agent_config(path)

    def test_schema_validation_failure_raises_load_error(self, tmp_path: Path) -> None:
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

    def test_schema_validation_error_chained_pydantic_error(self, tmp_path: Path) -> None:
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
        path = _write(tmp_path, "agents: []\n")
        with pytest.raises(AgentConfigLoadError):
            load_agent_config(path)

    def test_duplicate_agent_ids_raises_load_error(self, tmp_path: Path) -> None:
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


class TestAgentConfigLoadError:
    def test_error_str_contains_path(self) -> None:
        err = AgentConfigLoadError("/some/path/agents.yaml", "something went wrong")
        assert "/some/path/agents.yaml" in str(err)

    def test_error_str_contains_message(self) -> None:
        err = AgentConfigLoadError("/path", "validation failed")
        assert "validation failed" in str(err)

    def test_error_path_attribute(self) -> None:
        err = AgentConfigLoadError("/some/path.yaml", "oops")
        assert err.path == Path("/some/path.yaml")

    def test_error_message_attribute(self) -> None:
        err = AgentConfigLoadError("/p", "my message")
        assert err.message == "my message"


class TestLoadAgentConfigMcp:
    def test_mcp_registry_loaded(self, tmp_path: Path) -> None:
        path = _write(tmp_path, _MCP_YAML_WITH_REGISTRY)
        config = load_agent_config(path)
        assert "jira" in config.mcp_servers
        assert "slack" in config.mcp_servers
        assert isinstance(config.mcp_servers["jira"], MCPServerDefinition)

    def test_mcp_server_fields_populated(self, tmp_path: Path) -> None:
        path = _write(tmp_path, _MCP_YAML_WITH_REGISTRY)
        config = load_agent_config(path)
        jira = config.mcp_servers["jira"]
        assert jira.url == "http://127.0.0.1:8100/mcp"
        assert jira.tool_filter == ["create_issue", "search_issues"]
        assert jira.cache_tools_list is True

    def test_agent_mcps_plain_refs_loaded(self, tmp_path: Path) -> None:
        path = _write(tmp_path, _MCP_YAML_WITH_REGISTRY)
        config = load_agent_config(path)
        assert config.agents[0].mcps == ["jira", "slack"]

    def test_agent_mcps_override_ref_loaded(self, tmp_path: Path) -> None:
        path = _write(tmp_path, _MCP_YAML_WITH_OVERRIDE)
        config = load_agent_config(path)
        ref = config.agents[0].mcps[0]
        assert isinstance(ref, MCPServerRef)
        assert ref.server == "jira"
        assert ref.tool_filter == ["search_issues", "update_issue"]

    def test_unknown_mcp_ref_raises_load_error(self, tmp_path: Path) -> None:
        path = _write(tmp_path, _MCP_YAML_UNKNOWN_REF)
        with pytest.raises(AgentConfigLoadError, match="unknown MCP server"):
            load_agent_config(path)

    def test_no_mcp_servers_section_is_valid(self, tmp_path: Path) -> None:
        path = _write(tmp_path, _VALID_YAML)
        config = load_agent_config(path)
        assert config.mcp_servers == {}

    def test_header_placeholder_resolved_from_settings(self, tmp_path: Path) -> None:
        path = _write(tmp_path, _MCP_YAML_WITH_PLACEHOLDER)
        settings = AppSettings(OPENAI_API_KEY="sk-test", MCP_JIRA_TOKEN="resolved-tok")
        config = load_agent_config(path, settings)
        headers = config.mcp_servers["jira"].headers
        assert headers is not None
        assert headers["Authorization"] == "Bearer resolved-tok"

    def test_header_placeholder_resolved_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MCP_JIRA_TOKEN", "env-tok")
        path = _write(tmp_path, _MCP_YAML_WITH_PLACEHOLDER)
        config = load_agent_config(path)
        headers = config.mcp_servers["jira"].headers
        assert headers is not None
        assert headers["Authorization"] == "Bearer env-tok"

    def test_unresolvable_placeholder_left_unchanged(self, tmp_path: Path) -> None:
        path = _write(tmp_path, _MCP_YAML_WITH_PLACEHOLDER)
        config = load_agent_config(path)
        headers = config.mcp_servers["jira"].headers
        assert headers is not None
        assert headers["Authorization"] == "Bearer ${MCP_JIRA_TOKEN}"

    def test_plain_header_value_unchanged(self, tmp_path: Path) -> None:
        path = _write(tmp_path, _MCP_YAML_WITH_REGISTRY)
        config = load_agent_config(path)
        assert config.mcp_servers["jira"].headers
        assert config.mcp_servers["jira"].headers["Authorization"] == "Bearer tok-jira"

    def test_url_placeholder_resolved_from_settings(self, tmp_path: Path) -> None:
        """${MCP_JIRA_URL} in url is resolved via AppSettings (Docker Compose use-case)."""
        path = _write(tmp_path, _MCP_YAML_WITH_URL_PLACEHOLDER)
        settings = AppSettings(
            OPENAI_API_KEY="sk-test",
            MCP_JIRA_URL="http://mcp-jira:8100/mcp",
            MCP_JIRA_TOKEN="tok",
        )
        config = load_agent_config(path, settings)
        assert config.mcp_servers["jira"].url == "http://mcp-jira:8100/mcp"

    def test_url_placeholder_resolved_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MCP_JIRA_URL", "http://mcp-jira:8100/mcp")
        path = _write(tmp_path, _MCP_YAML_WITH_URL_PLACEHOLDER)
        config = load_agent_config(path)
        assert config.mcp_servers["jira"].url == "http://mcp-jira:8100/mcp"

    def test_unresolvable_url_placeholder_left_unchanged(self, tmp_path: Path) -> None:
        path = _write(tmp_path, _MCP_YAML_WITH_URL_PLACEHOLDER)
        config = load_agent_config(path)
        assert config.mcp_servers["jira"].url == "${MCP_JIRA_URL}"

    def test_docker_compose_urls_all_resolved(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """All four MCP_*_URL env vars resolve correctly (Docker Compose scenario)."""
        yaml_text = """\
process: sequential

mcp_servers:
  jira:
    type: http
    url: "${MCP_JIRA_URL}"
  clickup:
    type: http
    url: "${MCP_CLICKUP_URL}"
  github:
    type: http
    url: "${MCP_GITHUB_URL}"
  slack:
    type: http
    url: "${MCP_SLACK_URL}"

agents:
  - id: dev_agent
    role: Developer
    goal: Develop.
    backstory: You develop.
    llm:
      provider: openai
      model: gpt-4o
    mcps:
      - jira
      - clickup
      - github
      - slack
"""
        for key, val in {
            "MCP_JIRA_URL": "http://mcp-jira:8100/mcp",
            "MCP_CLICKUP_URL": "http://mcp-clickup:8101/mcp",
            "MCP_GITHUB_URL": "http://mcp-github:8102/mcp",
            "MCP_SLACK_URL": "http://mcp-slack:8103/mcp",
        }.items():
            monkeypatch.setenv(key, val)
        path = _write(tmp_path, yaml_text)
        config = load_agent_config(path)
        assert config.mcp_servers["jira"].url == "http://mcp-jira:8100/mcp"
        assert config.mcp_servers["clickup"].url == "http://mcp-clickup:8101/mcp"
        assert config.mcp_servers["github"].url == "http://mcp-github:8102/mcp"
        assert config.mcp_servers["slack"].url == "http://mcp-slack:8103/mcp"
