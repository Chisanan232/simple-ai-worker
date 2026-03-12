"""
Configuration package for simple-ai-worker.

This package provides the centralised, Pydantic-based configuration layer for
the application.  All secrets and environment-driven options are loaded
**exclusively** via :class:`~src.config.settings.AppSettings` which is backed
by ``pydantic-settings`` ``BaseSettings``.

Public API
----------
Phase 2 — Settings
    - :func:`get_settings` — Return the (cached) :class:`AppSettings` singleton.
    - :class:`AppSettings` — The settings model itself (re-exported for convenience).

Phase 3 — Agent Config
    - :func:`load_agent_config` — Load & validate ``config/agents.yaml``.
    - :class:`AgentTeamConfig` — Top-level validated config model.
    - :class:`AgentConfigLoadError` — Raised on any load / parse / validation failure.

Usage::

    from src.config import get_settings, load_agent_config

    settings = get_settings()
    agent_team = load_agent_config(settings.AGENT_CONFIG_PATH)

Overriding in tests::

    from src.config import get_settings

    get_settings.cache_clear()          # bust the lru_cache
    # set environment variables before calling get_settings() again
    fresh = get_settings()
"""

from __future__ import annotations

from typing import List

from .agent_config import AgentTeamConfig
from .loader import AgentConfigLoadError, load_agent_config
from .settings import AppSettings, get_settings

__all__: List[str] = [
    # Phase 2
    "AppSettings",
    "get_settings",
    # Phase 3
    "AgentTeamConfig",
    "AgentConfigLoadError",
    "load_agent_config",
]
