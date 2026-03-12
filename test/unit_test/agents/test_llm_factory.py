"""
Unit tests for :class:`src.agents.llm_factory.LLMFactory`.

Strategy
--------
- ``crewai.LLM`` validates an API key at construction time.  All tests
  that call ``LLMFactory.build()`` patch ``crewai.LLM`` so no real
  network call or env-key validation occurs.
- Tests for ``_model_string`` and ``_resolve_api_key`` are isolated
  helper tests that do NOT call the real ``crewai.LLM`` constructor.
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from src.agents.llm_factory import LLMFactory
from src.config.agent_config import LLMConfig, LLMOptions
from src.config.settings import AppSettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(
    openai_key: Optional[str] = None,
    anthropic_key: Optional[str] = None,
) -> AppSettings:
    """Return an AppSettings instance with optional secret values."""
    env: dict = {}
    if openai_key:
        env["OPENAI_API_KEY"] = openai_key
    if anthropic_key:
        env["ANTHROPIC_API_KEY"] = anthropic_key
    return AppSettings(**env)  # type: ignore[arg-type]


def _make_llm_config(
    provider: str = "openai",
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    top_p: float = 1.0,
    timeout: int = 120,
) -> LLMConfig:
    return LLMConfig(
        provider=provider,  # type: ignore[arg-type]
        model=model,
        options=LLMOptions(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            timeout=timeout,
        ),
    )


# ===========================================================================
# _model_string (private, pure helper — no mock needed)
# ===========================================================================

class TestModelString:
    def test_openai_format(self) -> None:
        assert LLMFactory._model_string("openai", "gpt-4o") == "openai/gpt-4o"

    def test_anthropic_format(self) -> None:
        result = LLMFactory._model_string("anthropic", "claude-3-5-sonnet-latest")
        assert result == "anthropic/claude-3-5-sonnet-latest"

    def test_separator_is_slash(self) -> None:
        result = LLMFactory._model_string("openai", "gpt-4-turbo")
        assert "/" in result
        parts = result.split("/")
        assert len(parts) == 2
        assert parts[0] == "openai"
        assert parts[1] == "gpt-4-turbo"


# ===========================================================================
# _resolve_api_key (private, pure helper — no mock needed)
# ===========================================================================

class TestResolveApiKey:
    def test_openai_key_returned_when_set(self) -> None:
        settings = _make_settings(openai_key="sk-test-openai")
        result = LLMFactory._resolve_api_key("openai", settings)
        assert result == "sk-test-openai"

    def test_anthropic_key_returned_when_set(self) -> None:
        settings = _make_settings(anthropic_key="sk-ant-test")
        result = LLMFactory._resolve_api_key("anthropic", settings)
        assert result == "sk-ant-test"

    def test_openai_key_returns_none_when_not_set(self) -> None:
        settings = _make_settings()
        result = LLMFactory._resolve_api_key("openai", settings)
        assert result is None

    def test_anthropic_key_returns_none_when_not_set(self) -> None:
        settings = _make_settings()
        result = LLMFactory._resolve_api_key("anthropic", settings)
        assert result is None

    def test_unknown_provider_returns_none(self) -> None:
        settings = _make_settings(openai_key="sk-x")
        result = LLMFactory._resolve_api_key("unknown_provider", settings)
        assert result is None


# ===========================================================================
# LLMFactory.build (patched crewai.LLM)
# ===========================================================================

class TestLLMFactoryBuild:
    """Tests for LLMFactory.build() — crewai.LLM is patched throughout."""

    def test_build_returns_llm_instance(self) -> None:
        """build() must return whatever crewai.LLM() returns."""
        cfg = _make_llm_config()
        settings = _make_settings(openai_key="sk-test")
        mock_llm = MagicMock()

        with patch("src.agents.llm_factory.LLM", return_value=mock_llm) as MockLLM:
            result = LLMFactory.build(cfg, settings)

        assert result is mock_llm
        MockLLM.assert_called_once()

    def test_build_passes_correct_model_string(self) -> None:
        """build() must call LLM(model='openai/gpt-4o', ...)."""
        cfg = _make_llm_config(provider="openai", model="gpt-4o")
        settings = _make_settings(openai_key="sk-test")

        with patch("src.agents.llm_factory.LLM") as MockLLM:
            LLMFactory.build(cfg, settings)

        call_kwargs = MockLLM.call_args
        assert call_kwargs.kwargs["model"] == "openai/gpt-4o" or call_kwargs.args[0] == "openai/gpt-4o"

    def test_build_passes_temperature(self) -> None:
        """build() must forward temperature from LLMOptions."""
        cfg = _make_llm_config(temperature=0.3)
        settings = _make_settings(openai_key="sk-test")

        with patch("src.agents.llm_factory.LLM") as MockLLM:
            LLMFactory.build(cfg, settings)

        kwargs = MockLLM.call_args.kwargs
        assert kwargs["temperature"] == 0.3

    def test_build_passes_max_tokens(self) -> None:
        """build() must forward max_tokens from LLMOptions."""
        cfg = _make_llm_config(max_tokens=2048)
        settings = _make_settings(openai_key="sk-test")

        with patch("src.agents.llm_factory.LLM") as MockLLM:
            LLMFactory.build(cfg, settings)

        kwargs = MockLLM.call_args.kwargs
        assert kwargs["max_tokens"] == 2048

    def test_build_passes_top_p(self) -> None:
        """build() must forward top_p from LLMOptions."""
        cfg = _make_llm_config(top_p=0.9)
        settings = _make_settings(openai_key="sk-test")

        with patch("src.agents.llm_factory.LLM") as MockLLM:
            LLMFactory.build(cfg, settings)

        kwargs = MockLLM.call_args.kwargs
        assert kwargs["top_p"] == 0.9

    def test_build_passes_timeout(self) -> None:
        """build() must forward timeout from LLMOptions."""
        cfg = _make_llm_config(timeout=60)
        settings = _make_settings(openai_key="sk-test")

        with patch("src.agents.llm_factory.LLM") as MockLLM:
            LLMFactory.build(cfg, settings)

        kwargs = MockLLM.call_args.kwargs
        assert kwargs["timeout"] == 60

    def test_build_injects_api_key_when_set(self) -> None:
        """build() must pass api_key when the secret is set in AppSettings."""
        cfg = _make_llm_config(provider="openai")
        settings = _make_settings(openai_key="sk-real-key")

        with patch("src.agents.llm_factory.LLM") as MockLLM:
            LLMFactory.build(cfg, settings)

        kwargs = MockLLM.call_args.kwargs
        assert kwargs.get("api_key") == "sk-real-key"

    def test_build_omits_api_key_when_not_set(self) -> None:
        """build() must not pass api_key when the secret is None."""
        cfg = _make_llm_config(provider="openai")
        settings = _make_settings()  # no key

        with patch("src.agents.llm_factory.LLM") as MockLLM:
            LLMFactory.build(cfg, settings)

        kwargs = MockLLM.call_args.kwargs
        assert "api_key" not in kwargs

    def test_build_anthropic_provider(self) -> None:
        """build() must construct the correct model string for anthropic."""
        cfg = _make_llm_config(provider="anthropic", model="claude-3-5-sonnet-latest")
        settings = _make_settings(anthropic_key="sk-ant-test")

        with patch("src.agents.llm_factory.LLM") as MockLLM:
            LLMFactory.build(cfg, settings)

        call_kwargs = MockLLM.call_args.kwargs
        assert call_kwargs["model"] == "anthropic/claude-3-5-sonnet-latest"
        assert call_kwargs.get("api_key") == "sk-ant-test"

