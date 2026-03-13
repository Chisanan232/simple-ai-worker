"""
Root-level pytest configuration and shared fixtures for the entire test suite.

Problem this file solves
------------------------
``AppSettings`` (pydantic-settings ``BaseSettings``) automatically reads
``env_file=".env"`` when the class is instantiated.  Once the developer
creates a real ``.env`` file at the project root (which contains live API
keys and tokens), those values leak into unit tests that construct
``AppSettings()`` with the expectation that optional secret fields are
``None`` or that default values are in effect.

In addition, if the developer's shell session already has those variables
exported (e.g. via ``direnv``, ``dotenv`` shell hooks, or a manual
``export``), pydantic-settings will still pick them up from ``os.environ``
even after ``env_file`` is redirected.  Both sources must be neutralised.

Solution
--------
A function-scoped autouse fixture:
1. Patches ``AppSettings.model_config["env_file"]`` to a path that is
   guaranteed not to exist (``".env.test"``), preventing file-based leakage.
2. Calls ``monkeypatch.delenv`` for every environment variable that the
   project's ``.env`` template defines, preventing OS-environment leakage.

Using function scope (instead of session scope) ensures the patch applies
to every individual test invocation, including retries triggered by
``pytest-rerunfailures``.

Tests that need a specific value must supply it explicitly:
- Constructor kwargs:      ``AppSettings(OPENAI_API_KEY="sk-x")``
- ``monkeypatch.setenv``:  ``monkeypatch.setenv("OPENAI_API_KEY", "sk-x")``
"""

from __future__ import annotations

import pytest

from src.config.settings import AppSettings

# ---------------------------------------------------------------------------
# Complete list of environment variables defined in the project's .env
# template.  Any variable present here will be removed from os.environ for
# the duration of each test so that tests are fully hermetic.
# ---------------------------------------------------------------------------
_PROJECT_ENV_VARS: tuple[str, ...] = (
    # CrewAI Enterprise Platform
    "CREWAI_PLATFORM_INTEGRATION_TOKEN",
    # LLM / AI Providers
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DEFAULT_LLM_PROVIDER",
    # Slack Bolt
    "SLACK_BOT_TOKEN",
    "SLACK_SIGNING_SECRET",
    "SLACK_PORT",
    # Scheduler
    "SCHEDULER_INTERVAL_SECONDS",
    "SCHEDULER_TIMEZONE",
    "MAX_CONCURRENT_DEV_AGENTS",
    # Agent Config
    "AGENT_CONFIG_PATH",
    # MCP tokens
    "MCP_JIRA_TOKEN",
    "MCP_CLICKUP_TOKEN",
    "MCP_GITHUB_TOKEN",
    "MCP_SLACK_TOKEN",
    # MCP URLs
    "MCP_JIRA_URL",
    "MCP_CLICKUP_URL",
    "MCP_GITHUB_URL",
    "MCP_SLACK_URL",
    # Atlassian / JIRA
    "ATLASSIAN_URL",
    "ATLASSIAN_EMAIL",
    # ClickUp
    "CLICKUP_TEAM_ID",
)


@pytest.fixture(autouse=True)
def _isolate_app_settings_from_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent AppSettings from reading the real .env file or OS env during tests.

    Two-step isolation:
    1. Patches ``AppSettings.model_config["env_file"]`` to a non-existent path
       so that no values from the developer's ``.env`` file leak into tests.
    2. Removes every project-defined env var from ``os.environ`` so that
       values exported into the shell session (e.g. via direnv) do not leak
       into tests either.

    Scoped to ``function`` (the default) so it applies to every test
    including retries from ``pytest-rerunfailures``.
    """
    # 1. Redirect .env file loading to a path that does not exist.
    monkeypatch.setitem(AppSettings.model_config, "env_file", ".env.test")

    # 2. Strip project env vars from the OS environment.
    for var in _PROJECT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)



