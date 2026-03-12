"""
Slack Bolt application package for simple-ai-worker (Phase 6).

This package contains:

- :mod:`~src.slack_app.app`     — ``create_bolt_app()`` factory that builds
  and configures the single ``@ai-worker`` Slack Bolt app with Socket Mode.
- :mod:`~src.slack_app.router`  — ``role_router()`` that parses the
  ``[planner]`` / ``[dev lead]`` tag from an incoming Slack message and
  dispatches to the correct handler.
- :mod:`~src.slack_app.handlers` — per-role handler sub-package.

Exported names
--------------
``create_bolt_app``
    Build and return the configured :class:`slack_bolt.App` instance.
"""

from __future__ import annotations

from typing import List

from .app import create_bolt_app

__all__: List[str] = ["create_bolt_app"]

