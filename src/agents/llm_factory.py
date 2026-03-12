"""
LLM factory for simple-ai-worker.

Provides :class:`LLMFactory` — a stateless factory that converts a
:class:`~src.config.agent_config.LLMConfig` plus the application
:class:`~src.config.settings.AppSettings` into a fully initialised
``crewai.LLM`` instance.

Model-string convention
-----------------------
CrewAI's ``LLM`` constructor expects a ``"<provider>/<model>"`` string
(e.g. ``"openai/gpt-4o"``, ``"anthropic/claude-3-5-sonnet-latest"``).
:meth:`LLMFactory.build` constructs this string automatically from
``llm_config.provider`` and ``llm_config.model``.

API-key injection
-----------------
``crewai.LLM`` reads credentials from environment variables at
construction time:

- ``OPENAI_API_KEY``    — picked up from the process environment.
- ``ANTHROPIC_API_KEY`` — picked up from the process environment.

``AppSettings`` stores these as :class:`pydantic.SecretStr`; this factory
calls ``get_secret_value()`` and injects them via ``api_key`` only when the
field is set — avoiding the accidental exposure of ``None`` as a string.

Usage::

    from src.agents.llm_factory import LLMFactory
    from src.config import get_settings
    from src.config.agent_config import LLMConfig

    llm_cfg = LLMConfig(provider="openai", model="gpt-4o")
    llm = LLMFactory.build(llm_cfg, get_settings())
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

from crewai import LLM

if TYPE_CHECKING:
    from src.config.agent_config import LLMConfig
    from src.config.settings import AppSettings

__all__: List[str] = ["LLMFactory"]

logger = logging.getLogger(__name__)

# Provider → environment-variable name (informational only; LLM reads env directly)
_PROVIDER_ENV_VAR: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


class LLMFactory:
    """Stateless factory that builds a ``crewai.LLM`` from Pydantic config.

    All methods are class methods so the factory never needs to be
    instantiated.
    """

    @classmethod
    def build(cls, llm_config: "LLMConfig", settings: "AppSettings") -> LLM:
        """Build and return a ``crewai.LLM`` instance.

        Constructs the ``"<provider>/<model>"`` model string required by
        CrewAI and forwards all ``LLMOptions`` fields as keyword arguments.
        The API key is injected from ``AppSettings`` when available.

        Args:
            llm_config: The :class:`~src.config.agent_config.LLMConfig`
                parsed from the agent YAML.
            settings: The application :class:`~src.config.settings.AppSettings`
                singleton, used to retrieve provider API keys.

        Returns:
            A fully initialised ``crewai.LLM`` instance ready to be passed
            to ``crewai.Agent(llm=...)``.

        Raises:
            ValueError: If the required API key for the chosen provider is
                not set in ``AppSettings`` / environment.
        """
        model_string = cls._model_string(llm_config.provider, llm_config.model)
        api_key: Optional[str] = cls._resolve_api_key(llm_config.provider, settings)

        opts = llm_config.options
        kwargs: dict[str, object] = {
            "temperature": opts.temperature,
            "max_tokens": opts.max_tokens,
            "top_p": opts.top_p,
            "timeout": opts.timeout,
        }
        if api_key is not None:
            kwargs["api_key"] = api_key

        logger.debug(
            "Building LLM: model=%s, temperature=%s, max_tokens=%s, timeout=%s.",
            model_string,
            opts.temperature,
            opts.max_tokens,
            opts.timeout,
        )
        return LLM(model=model_string, **kwargs)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _model_string(provider: str, model: str) -> str:
        """Return the ``"<provider>/<model>"`` string expected by ``crewai.LLM``.

        Args:
            provider: One of ``"openai"`` or ``"anthropic"``.
            model: The model identifier (e.g. ``"gpt-4o"``).

        Returns:
            Formatted model string, e.g. ``"openai/gpt-4o"``.
        """
        return f"{provider}/{model}"

    @staticmethod
    def _resolve_api_key(provider: str, settings: "AppSettings") -> Optional[str]:
        """Return the raw API-key string for *provider* from *settings*.

        Returns ``None`` if the field is not set (i.e. the value is ``None``),
        allowing CrewAI to fall back to the process environment variable.

        Args:
            provider: One of ``"openai"`` or ``"anthropic"``.
            settings: The application settings instance.

        Returns:
            The raw secret string, or ``None`` if not configured.
        """
        if provider == "openai":
            secret = settings.OPENAI_API_KEY
        elif provider == "anthropic":
            secret = settings.ANTHROPIC_API_KEY
        else:
            return None

        return secret.get_secret_value() if secret is not None else None

