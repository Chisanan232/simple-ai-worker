"""
Configuration package for simple-ai-worker.

This package provides the centralised, Pydantic-based configuration layer for
the application.  All secrets and environment-driven options are loaded
**exclusively** via :class:`~src.config.settings.AppSettings` which is backed
by ``pydantic-settings`` ``BaseSettings``.

Public API
----------
- :func:`get_settings` — Return the (cached) :class:`AppSettings` singleton.
- :class:`AppSettings` — The settings model itself (re-exported for convenience).

Usage::

    from src.config import get_settings

    settings = get_settings()
    print(settings.SCHEDULER_INTERVAL_SECONDS)

Overriding in tests::

    from src.config import get_settings

    get_settings.cache_clear()          # bust the lru_cache
    # set environment variables before calling get_settings() again
    fresh = get_settings()
"""

from __future__ import annotations

from typing import List

from .settings import AppSettings, get_settings

__all__: List[str] = [
    "AppSettings",
    "get_settings",
]
