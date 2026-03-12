"""
Pydantic-Settings model for simple-ai-worker.

All application secrets and environment-driven configuration are defined here
and loaded **exclusively** through ``pydantic-settings`` ``BaseSettings``.
No ``load_dotenv()`` call exists anywhere in the application — ``BaseSettings``
handles ``.env`` file parsing natively.

Field Groups
------------

CrewAI Enterprise Platform
    ``CREWAI_PLATFORM_INTEGRATION_TOKEN`` — used by every native
    ``Agent(apps=[...])`` integration (JIRA, ClickUp, GitHub, Slack via
    CrewAI AMP).  Required at runtime for phases 5+; optional in earlier
    phases (defaults to ``None``).

LLM / AI Providers
    ``OPENAI_API_KEY``        — required for OpenAI models.
    ``ANTHROPIC_API_KEY``     — optional; required only when using Anthropic.
    ``DEFAULT_LLM_PROVIDER``  — selects the default provider (``"openai"`` or
                                ``"anthropic"``).

Slack Bolt — Events API HTTP server
    These fields are used *directly* by the Slack Bolt Events API HTTP server
    (``src.slack_app``).  They are **separate** from any Slack credentials
    managed by CrewAI AMP.
    ``SLACK_BOT_TOKEN``      — ``xoxb-…``  bot user OAuth token.
    ``SLACK_SIGNING_SECRET`` — verifies incoming event-payload signatures.
    ``SLACK_PORT``           — TCP port for ``AsyncApp.start()`` (default ``3000``).

Scheduler
    ``SCHEDULER_INTERVAL_SECONDS`` — default polling interval for all jobs.
    ``SCHEDULER_TIMEZONE``         — timezone string (e.g. ``"UTC"``).
    ``MAX_CONCURRENT_DEV_AGENTS``  — thread-pool cap for dev-agent execution.

Agent Config
    ``AGENT_CONFIG_PATH`` — path (relative to project root) of the YAML agent
                            config file loaded in Phase 3.

Usage::

    from src.config import get_settings

    settings = get_settings()
    interval = settings.SCHEDULER_INTERVAL_SECONDS   # int
    token = settings.OPENAI_API_KEY.get_secret_value()  # str

Overriding in tests::

    import os
    from src.config import get_settings

    os.environ["OPENAI_API_KEY"] = "sk-test"
    get_settings.cache_clear()
    settings = get_settings()
"""

from __future__ import annotations

import functools
from typing import List, Literal, Optional

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__: List[str] = [
    "AppSettings",
    "get_settings",
]


class AppSettings(BaseSettings):
    """Centralised, strongly-typed application settings.

    All values are resolved from environment variables and/or the ``.env``
    file at the project root.  ``pydantic-settings`` performs this natively —
    no ``load_dotenv()`` call is needed anywhere in application code.

    Attributes:
        CREWAI_PLATFORM_INTEGRATION_TOKEN: CrewAI Enterprise platform token
            used for all native ``Agent(apps=[...])`` integrations
            (JIRA / ClickUp / GitHub / Slack via CrewAI AMP).  Optional in
            phases 1–4; **required** from Phase 5 onward.
        OPENAI_API_KEY: OpenAI API key.  Required when
            ``DEFAULT_LLM_PROVIDER`` is ``"openai"``.
        ANTHROPIC_API_KEY: Anthropic API key.  Required only when
            ``DEFAULT_LLM_PROVIDER`` is ``"anthropic"``.
        DEFAULT_LLM_PROVIDER: Selects which LLM provider is used as the
            default for all agents.  One of ``"openai"`` or ``"anthropic"``.
        SLACK_BOT_TOKEN: Slack bot user OAuth token (``xoxb-…``).  Used
            directly by the Slack Bolt Events API HTTP server to authenticate
            requests.  Required from Phase 6 onward (``simple-ai-slack``
            process only).
        SLACK_SIGNING_SECRET: Slack signing secret.  Used by Bolt to verify
            the authenticity of incoming event payloads.  Required from
            Phase 6 onward (``simple-ai-slack`` process only).
        SLACK_PORT: TCP port for the Slack Bolt built-in HTTP server
            (Events API endpoint).  ``AsyncApp.start()`` listens on this port
            at ``POST /slack/events``.  Defaults to ``3000``.
        SCHEDULER_INTERVAL_SECONDS: Default polling interval (seconds) applied
            to all interval-based APScheduler jobs.  Defaults to ``60``.
        SCHEDULER_TIMEZONE: Timezone string understood by APScheduler / pytz
            (e.g. ``"UTC"``, ``"Asia/Taipei"``).  Defaults to ``"UTC"``.
        MAX_CONCURRENT_DEV_AGENTS: Upper bound on the number of dev-agent
            Crew executions that may run concurrently inside the bounded
            ``ThreadPoolExecutor``.  Defaults to ``3``.
        AGENT_CONFIG_PATH: Relative (or absolute) path to the YAML agent
            configuration file.  Defaults to ``"config/agents.yaml"``.
    """

    model_config = SettingsConfigDict(
        # Load from a .env file at the project root.
        # pydantic-settings handles this natively — no load_dotenv() call
        # anywhere in application code.
        env_file=".env",
        env_file_encoding="utf-8",
        # Extra keys in the environment / .env are silently ignored so that
        # third-party tools that inject their own env vars don't break startup.
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # CrewAI Enterprise Platform
    # ------------------------------------------------------------------

    CREWAI_PLATFORM_INTEGRATION_TOKEN: Optional[SecretStr] = None
    """CrewAI Enterprise platform integration token.

    Required for all native ``Agent(apps=[...])`` integrations (Phase 5+).
    Must be set to the token copied from the CrewAI AMP → Integrations page.
    """

    # ------------------------------------------------------------------
    # LLM / AI Providers
    # ------------------------------------------------------------------

    OPENAI_API_KEY: Optional[SecretStr] = None
    """OpenAI API key (``sk-…``).

    Required when ``DEFAULT_LLM_PROVIDER`` is ``"openai"`` (the default).
    """

    ANTHROPIC_API_KEY: Optional[SecretStr] = None
    """Anthropic API key.

    Required only when ``DEFAULT_LLM_PROVIDER`` is ``"anthropic"``.
    """

    DEFAULT_LLM_PROVIDER: Literal["openai", "anthropic"] = "openai"
    """Default LLM provider for all agents.

    Must be one of ``"openai"`` or ``"anthropic"``.  Defaults to
    ``"openai"``.
    """

    # ------------------------------------------------------------------
    # Slack Bolt — direct credentials for the Events API HTTP server
    # (separate from CrewAI AMP OAuth)
    # ------------------------------------------------------------------

    SLACK_BOT_TOKEN: Optional[SecretStr] = None
    """Slack bot user OAuth token (``xoxb-…``).

    Used directly by the Slack Bolt Events API HTTP server.
    Required from Phase 6 onward (``simple-ai-slack`` process only).
    """

    SLACK_SIGNING_SECRET: Optional[SecretStr] = None
    """Slack signing secret.

    Used by Bolt to verify the authenticity of incoming event payloads.
    Required from Phase 6 onward (``simple-ai-slack`` process only).
    """

    SLACK_PORT: int = 3000
    """TCP port for the Slack Bolt built-in HTTP server (Events API endpoint).

    ``AsyncApp.start()`` listens on this port at ``POST /slack/events``.
    Binds to ``0.0.0.0`` internally — only the port is configurable.
    Defaults to ``3000``.
    """

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    SCHEDULER_INTERVAL_SECONDS: int = 60
    """Default polling interval for all APScheduler interval-based jobs.

    Unit: seconds.  Defaults to ``60``.
    """

    SCHEDULER_TIMEZONE: str = "UTC"
    """Timezone string for the APScheduler ``BackgroundScheduler``.

    Any timezone string recognised by APScheduler / pytz is valid (e.g.
    ``"UTC"``, ``"Asia/Taipei"``, ``"America/New_York"``).
    Defaults to ``"UTC"``.
    """

    MAX_CONCURRENT_DEV_AGENTS: int = 3
    """Maximum number of dev-agent Crews running concurrently.

    Controls the size of the bounded ``ThreadPoolExecutor`` used for
    dev-agent dispatching.  Defaults to ``3``.
    """

    # ------------------------------------------------------------------
    # Agent Config
    # ------------------------------------------------------------------

    AGENT_CONFIG_PATH: str = "config/agents.yaml"
    """Path to the YAML agent configuration file.

    Relative paths are resolved from the working directory where the
    application is launched.  Defaults to ``"config/agents.yaml"``.
    """


@functools.lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return the cached :class:`AppSettings` singleton.

    The settings object is instantiated once per process and cached via
    :func:`functools.lru_cache`.  To force re-evaluation (e.g. in tests
    after changing environment variables), call
    ``get_settings.cache_clear()`` before calling this function again.

    Returns:
        The application settings instance loaded from the environment
        and/or ``.env`` file.

    Example::

        from src.config import get_settings

        settings = get_settings()
        print(settings.SCHEDULER_INTERVAL_SECONDS)  # 60

    Test override example::

        import os
        from src.config import get_settings

        os.environ["SCHEDULER_INTERVAL_SECONDS"] = "30"
        get_settings.cache_clear()
        assert get_settings().SCHEDULER_INTERVAL_SECONDS == 30
    """
    return AppSettings()
