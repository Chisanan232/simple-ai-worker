"""
Unit tests for :class:`src.config.settings.AppSettings` and
:func:`src.config.settings.get_settings`.

Strategy
--------
- All tests **set environment variables explicitly** via
  ``monkeypatch.setenv`` rather than relying on a ``.env`` file on disk.
  This keeps tests hermetic and avoids side-effects from a developer's
  local ``.env``.
- ``get_settings`` is cached with ``@lru_cache``.  Each test that exercises
  ``get_settings`` must call ``get_settings.cache_clear()`` in its teardown
  to prevent cache pollution between tests.
"""

from __future__ import annotations

import pytest

from src.config.settings import AppSettings, get_settings


class TestAppSettingsDefaults:
    """Tests that verify the default values applied when env vars are absent."""

    def test_default_llm_provider_is_openai(self) -> None:
        """DEFAULT_LLM_PROVIDER must default to 'openai'."""
        settings = AppSettings()
        assert settings.DEFAULT_LLM_PROVIDER == "openai"

    def test_default_scheduler_interval_is_60(self) -> None:
        """SCHEDULER_INTERVAL_SECONDS must default to 60."""
        settings = AppSettings()
        assert settings.SCHEDULER_INTERVAL_SECONDS == 60

    def test_default_scheduler_timezone_is_utc(self) -> None:
        """SCHEDULER_TIMEZONE must default to 'UTC'."""
        settings = AppSettings()
        assert settings.SCHEDULER_TIMEZONE == "UTC"

    def test_default_max_concurrent_dev_agents_is_3(self) -> None:
        """MAX_CONCURRENT_DEV_AGENTS must default to 3."""
        settings = AppSettings()
        assert settings.MAX_CONCURRENT_DEV_AGENTS == 3

    def test_default_agent_config_path(self) -> None:
        """AGENT_CONFIG_PATH must default to 'config/agents.yaml'."""
        settings = AppSettings()
        assert settings.AGENT_CONFIG_PATH == "config/agents.yaml"

    def test_optional_secrets_default_to_none(self) -> None:
        """All optional SecretStr fields must default to None."""
        settings = AppSettings()
        assert settings.CREWAI_PLATFORM_INTEGRATION_TOKEN is None
        assert settings.OPENAI_API_KEY is None
        assert settings.ANTHROPIC_API_KEY is None
        assert settings.SLACK_BOT_TOKEN is None
        assert settings.SLACK_SIGNING_SECRET is None


class TestAppSettingsFromEnv:
    """Tests that verify values are correctly loaded from environment variables."""

    def test_openai_api_key_loaded_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OPENAI_API_KEY must be loaded as a SecretStr from the environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-key")
        settings = AppSettings()
        assert settings.OPENAI_API_KEY is not None
        assert settings.OPENAI_API_KEY.get_secret_value() == "sk-test-openai-key"

    def test_anthropic_api_key_loaded_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ANTHROPIC_API_KEY must be loaded as a SecretStr from the environment."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        settings = AppSettings()
        assert settings.ANTHROPIC_API_KEY is not None
        assert settings.ANTHROPIC_API_KEY.get_secret_value() == "sk-ant-test-key"

    def test_crewai_token_loaded_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CREWAI_PLATFORM_INTEGRATION_TOKEN must be loaded as a SecretStr."""
        monkeypatch.setenv("CREWAI_PLATFORM_INTEGRATION_TOKEN", "crewai-token-xyz")
        settings = AppSettings()
        assert settings.CREWAI_PLATFORM_INTEGRATION_TOKEN is not None
        assert settings.CREWAI_PLATFORM_INTEGRATION_TOKEN.get_secret_value() == "crewai-token-xyz"

    def test_slack_bot_token_loaded_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SLACK_BOT_TOKEN must be loaded as a SecretStr from the environment."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-bot-token")
        settings = AppSettings()
        assert settings.SLACK_BOT_TOKEN is not None
        assert settings.SLACK_BOT_TOKEN.get_secret_value() == "xoxb-test-bot-token"

    def test_slack_port_loaded_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SLACK_PORT must be coerced to int from the environment."""
        monkeypatch.setenv("SLACK_PORT", "4000")
        settings = AppSettings()
        assert settings.SLACK_PORT == 4000

    def test_slack_signing_secret_loaded_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SLACK_SIGNING_SECRET must be loaded as a SecretStr from the environment."""
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "slack-signing-secret-abc")
        settings = AppSettings()
        assert settings.SLACK_SIGNING_SECRET is not None
        assert settings.SLACK_SIGNING_SECRET.get_secret_value() == "slack-signing-secret-abc"

    def test_scheduler_interval_loaded_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SCHEDULER_INTERVAL_SECONDS must be coerced to int from the environment."""
        monkeypatch.setenv("SCHEDULER_INTERVAL_SECONDS", "120")
        settings = AppSettings()
        assert settings.SCHEDULER_INTERVAL_SECONDS == 120

    def test_scheduler_timezone_loaded_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SCHEDULER_TIMEZONE must be loaded as a str from the environment."""
        monkeypatch.setenv("SCHEDULER_TIMEZONE", "Asia/Taipei")
        settings = AppSettings()
        assert settings.SCHEDULER_TIMEZONE == "Asia/Taipei"

    def test_max_concurrent_dev_agents_loaded_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MAX_CONCURRENT_DEV_AGENTS must be coerced to int from the environment."""
        monkeypatch.setenv("MAX_CONCURRENT_DEV_AGENTS", "5")
        settings = AppSettings()
        assert settings.MAX_CONCURRENT_DEV_AGENTS == 5

    def test_agent_config_path_loaded_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AGENT_CONFIG_PATH must be loaded as a str from the environment."""
        monkeypatch.setenv("AGENT_CONFIG_PATH", "custom/path/agents.yaml")
        settings = AppSettings()
        assert settings.AGENT_CONFIG_PATH == "custom/path/agents.yaml"

    def test_default_llm_provider_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DEFAULT_LLM_PROVIDER must accept 'anthropic'."""
        monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "anthropic")
        settings = AppSettings()
        assert settings.DEFAULT_LLM_PROVIDER == "anthropic"


class TestAppSettingsSecretStr:
    """Tests that verify SecretStr fields are never accidentally serialised."""

    def test_openai_api_key_repr_does_not_expose_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The repr/str of OPENAI_API_KEY must not contain the raw secret value."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret")
        settings = AppSettings()
        assert settings.OPENAI_API_KEY is not None
        # pydantic SecretStr repr shows '**********' — not the raw value
        assert "sk-super-secret" not in repr(settings.OPENAI_API_KEY)
        assert "sk-super-secret" not in str(settings.OPENAI_API_KEY)

    def test_slack_bot_token_repr_does_not_expose_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The repr/str of SLACK_BOT_TOKEN must not contain the raw token."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-very-secret")
        settings = AppSettings()
        assert settings.SLACK_BOT_TOKEN is not None
        assert "xoxb-very-secret" not in repr(settings.SLACK_BOT_TOKEN)
        assert "xoxb-very-secret" not in str(settings.SLACK_BOT_TOKEN)


class TestAppSettingsValidation:
    """Tests for field-level validation rules."""

    def test_invalid_llm_provider_raises_validation_error(self) -> None:
        """DEFAULT_LLM_PROVIDER must reject values outside the Literal set."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AppSettings(DEFAULT_LLM_PROVIDER="gemini")

    def test_scheduler_interval_must_be_int(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SCHEDULER_INTERVAL_SECONDS must be coercible to int; invalid str raises."""
        from pydantic import ValidationError

        monkeypatch.setenv("SCHEDULER_INTERVAL_SECONDS", "not-a-number")
        with pytest.raises(ValidationError):
            AppSettings()

    def test_max_concurrent_dev_agents_must_be_int(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MAX_CONCURRENT_DEV_AGENTS must be coercible to int."""
        from pydantic import ValidationError

        monkeypatch.setenv("MAX_CONCURRENT_DEV_AGENTS", "abc")
        with pytest.raises(ValidationError):
            AppSettings()


class TestGetSettings:
    """Tests for the :func:`get_settings` cached factory function."""

    def setup_method(self) -> None:
        """Clear the lru_cache before each test."""
        get_settings.cache_clear()

    def teardown_method(self) -> None:
        """Clear the lru_cache after each test to avoid pollution."""
        get_settings.cache_clear()

    def test_get_settings_returns_app_settings_instance(self) -> None:
        """get_settings() must return an AppSettings instance."""
        settings = get_settings()
        assert isinstance(settings, AppSettings)

    def test_get_settings_is_cached(self) -> None:
        """Calling get_settings() twice must return the same object."""
        first = get_settings()
        second = get_settings()
        assert first is second

    def test_get_settings_cache_clear_returns_fresh_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """After cache_clear(), get_settings() must return a new instance."""
        first = get_settings()
        get_settings.cache_clear()

        monkeypatch.setenv("SCHEDULER_INTERVAL_SECONDS", "999")
        second = get_settings()

        assert first is not second
        assert second.SCHEDULER_INTERVAL_SECONDS == 999

    def test_get_settings_uses_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_settings() must reflect env vars set before first call."""
        monkeypatch.setenv("SCHEDULER_TIMEZONE", "America/New_York")
        settings = get_settings()
        assert settings.SCHEDULER_TIMEZONE == "America/New_York"
