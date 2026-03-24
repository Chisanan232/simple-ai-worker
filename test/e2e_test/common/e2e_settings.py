"""
E2E-only Pydantic-Settings model.

Reads exclusively from ``test/e2e_test/.env.e2e`` — completely separate
from the application's ``.env`` at the project root.  The application
layer (``AppSettings``) never loads this file.

Usage in ``test/e2e_test/conftest.py``::

    from test.e2e_test.common.e2e_settings import get_e2e_settings, E2ESettings

    _e2e_settings = get_e2e_settings()
    _LLM_KEY_AVAILABLE = _e2e_settings.has_llm_key

    @pytest.fixture(scope="session")
    def e2e_settings() -> E2ESettings:
        return _e2e_settings
"""

from __future__ import annotations

import functools
from typing import Optional

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class E2ESettings(BaseSettings):
    """Settings for E2E tests only.

    Reads from ``test/e2e_test/.env.e2e`` (relative to the project root,
    which is the working directory when running ``uv run pytest``).
    """

    model_config = SettingsConfigDict(
        env_file="./test/e2e_test/.env.e2e",
        env_file_encoding="utf-8",
        env_prefix="E2E_",
        extra="ignore",
        env_ignore_empty=True,
    )

    # ------------------------------------------------------------------
    # LLM API Keys — canonical names, no E2E_ prefix.
    # AliasChoices allows BOTH the canonical name and the E2E_-prefixed
    # form so users can write either ``OPENAI_API_KEY=`` or
    # ``E2E_OPENAI_API_KEY=`` in test/e2e_test/.env.e2e.
    # ------------------------------------------------------------------

    OPENAI_API_KEY: Optional[SecretStr] = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "E2E_OPENAI_API_KEY"),
    )
    ANTHROPIC_API_KEY: Optional[SecretStr] = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "E2E_ANTHROPIC_API_KEY"),
    )

    # ------------------------------------------------------------------
    # E2E LLM Config (with E2E_ prefix)
    # ------------------------------------------------------------------

    LLM_MODEL: str = "gpt-4o-mini"
    LLM_PROVIDER: str = "openai"

    # ------------------------------------------------------------------
    # MCP Server URLs
    # Empty  → stub mode (pytest-httpserver).
    # Non-empty → live mode (Docker Compose services).
    # ------------------------------------------------------------------

    MCP_JIRA_URL: Optional[str] = None
    MCP_CLICKUP_URL: Optional[str] = None
    MCP_GITHUB_URL: Optional[str] = None
    MCP_SLACK_URL: Optional[str] = None

    # ------------------------------------------------------------------
    # Atlassian / JIRA credentials (live mode only)
    # ------------------------------------------------------------------

    ATLASSIAN_URL: Optional[str] = None
    ATLASSIAN_EMAIL: Optional[str] = None
    MCP_JIRA_TOKEN: Optional[SecretStr] = None
    JIRA_PROJECT_KEY: Optional[str] = None

    # ------------------------------------------------------------------
    # ClickUp credentials (live mode only)
    # ------------------------------------------------------------------

    MCP_CLICKUP_TOKEN: Optional[SecretStr] = None
    CLICKUP_LIST_ID: Optional[str] = None
    CLICKUP_TEAM_ID: Optional[str] = None

    # ------------------------------------------------------------------
    # GitHub credentials (live mode only)
    # ------------------------------------------------------------------

    MCP_GITHUB_TOKEN: Optional[SecretStr] = None
    GITHUB_TEST_REPO: Optional[str] = None

    # ------------------------------------------------------------------
    # Slack credentials (live mode only)
    # ------------------------------------------------------------------

    MCP_SLACK_TOKEN: Optional[SecretStr] = None
    SLACK_TEST_CHANNEL_ID: Optional[str] = None

    # ------------------------------------------------------------------
    # FakeLLM / Testcontainers toggles  (NEW)
    # ------------------------------------------------------------------

    USE_FAKE_LLM: bool = False
    """Replace LLMFactory.build() with FakeLLM for the entire test session.

    When ``true``, no real LLM API key is needed.  Tests exercise
    orchestration / handler logic but do NOT validate LLM reasoning quality.

    Set via ``E2E_USE_FAKE_LLM=true`` in ``test/e2e_test/.env.e2e``.
    """

    USE_TESTCONTAINERS: bool = False
    """Automatically start the Docker Compose MCP stack via testcontainers.

    When ``true``, the ``live_mcp_stack`` session fixture spins up
    ``docker-compose.e2e.yml`` and derives container-mapped port URLs
    instead of reading ``E2E_MCP_*_URL`` from the env file.

    Set via ``E2E_USE_TESTCONTAINERS=true`` in ``test/e2e_test/.env.e2e``.
    """

    # ------------------------------------------------------------------
    # Computed helpers — the sole decision points used by conftest.py
    # ------------------------------------------------------------------

    @property
    def has_llm_key(self) -> bool:
        """True if at least one LLM API key is configured, OR fake LLM is enabled.

        The ``skip_without_llm`` marker checks this property, so setting
        ``USE_FAKE_LLM=true`` prevents tests from being skipped even when
        no real provider API key is available.
        """
        return bool(self.OPENAI_API_KEY or self.ANTHROPIC_API_KEY or self.USE_FAKE_LLM)

    @property
    def is_live_mode(self) -> bool:
        """True when testcontainers mode is active (``USE_TESTCONTAINERS=true``).

        ``USE_FAKE_LLM`` is **orthogonal** — it controls the LLM tier only.
        When both flags are ``true``, real Docker containers ARE running and
        FakeLLM is used to drive tool calls against them.

        Decision table
        --------------
        USE_TESTCONTAINERS=true  + USE_FAKE_LLM=false → live mode  (containers + real LLM)
        USE_TESTCONTAINERS=true  + USE_FAKE_LLM=true  → live mode  (containers + FakeLLM)
        USE_TESTCONTAINERS=false + anything            → stub mode  (in-process stub)
        """
        return self.USE_TESTCONTAINERS

    @property
    def configured_tc_services(self) -> list[str]:
        """Docker Compose service names whose credentials are fully configured.

        Only services with all required credentials present are included so
        that ``docker compose up --wait`` is only called against services that
        can actually pass their healthchecks.

        Services are mapped as:
          ``mcp-jira-e2e``    → requires ATLASSIAN_URL + ATLASSIAN_EMAIL + MCP_JIRA_TOKEN
          ``mcp-clickup-e2e`` → requires MCP_CLICKUP_TOKEN
          ``mcp-github-e2e``  → requires MCP_GITHUB_TOKEN
          ``mcp-slack-e2e``   → requires MCP_SLACK_TOKEN

        Returns an empty list when no service credentials are configured (i.e.
        testcontainers mode was requested but no credentials are present).
        """
        services: list[str] = []
        if self.MCP_CLICKUP_TOKEN:
            services.append("mcp-clickup-e2e")
        if self.MCP_GITHUB_TOKEN:
            services.append("mcp-github-e2e")
        if self.MCP_SLACK_TOKEN:
            services.append("mcp-slack-e2e")
        if self.ATLASSIAN_URL and self.ATLASSIAN_EMAIL and self.MCP_JIRA_TOKEN:
            services.append("mcp-jira-e2e")
        return services

    @property
    def llm_key_value(self) -> str | None:
        """Raw LLM API key string (OpenAI preferred, Anthropic fallback)."""
        if self.OPENAI_API_KEY:
            return self.OPENAI_API_KEY.get_secret_value()
        if self.ANTHROPIC_API_KEY:
            return self.ANTHROPIC_API_KEY.get_secret_value()
        return None

    @property
    def llm_key_env_var(self) -> str:
        """Environment variable name for the available LLM key.

        Used when temporarily injecting the key into ``os.environ`` for
        CrewAI's internal LLM validator.
        Returns ``"OPENAI_API_KEY"`` or ``"ANTHROPIC_API_KEY"``.
        """
        if self.OPENAI_API_KEY:
            return "OPENAI_API_KEY"
        return "ANTHROPIC_API_KEY"


@functools.lru_cache(maxsize=1)
def get_e2e_settings() -> E2ESettings:
    """Return the cached E2ESettings singleton.

    The ``test/e2e_test/.env.e2e`` file is read only once per process.
    Call ``get_e2e_settings.cache_clear()`` to force re-evaluation.
    """
    return E2ESettings()
