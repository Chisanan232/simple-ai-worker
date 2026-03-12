"""
YAML loader for the agent team configuration file.

This module provides :func:`load_agent_config` — the single public entry
point for reading and validating ``config/agents.yaml`` (or whatever path
is set in ``AppSettings.AGENT_CONFIG_PATH``).

Design decisions
----------------
- Uses ``yaml.safe_load`` — never ``yaml.load`` — to prevent arbitrary
  Python object deserialisation from untrusted YAML.
- Returns a fully validated :class:`~src.config.agent_config.AgentTeamConfig`
  Pydantic model; callers receive a typed, IDE-navigable object rather than a
  raw ``dict``.
- Raises :class:`AgentConfigLoadError` (a thin wrapper around the underlying
  ``OSError`` / ``yaml.YAMLError`` / ``pydantic.ValidationError``) so callers
  only need to handle one exception type while still being able to inspect the
  original cause via ``__cause__``.
- Fails **fast** — the loader is called once at startup before any agent or
  crew is constructed, so a bad YAML file surfaces immediately rather than
  causing a cryptic error mid-run.

Usage::

    from src.config.loader import load_agent_config

    config = load_agent_config("config/agents.yaml")
    for agent in config.agents:
        print(agent.id, agent.llm.model)

Test override example::

    import textwrap, tempfile, pathlib
    from src.config.loader import load_agent_config

    yaml_text = textwrap.dedent(\"\"\"
        process: sequential
        agents:
          - id: planner
            role: Product Planner
            goal: Plan the product
            backstory: You are a planner.
            llm:
              provider: openai
              model: gpt-4o
    \"\"\")
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(yaml_text)
        path = f.name
    config = load_agent_config(path)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import yaml
from pydantic import ValidationError

from .agent_config import AgentTeamConfig

__all__: List[str] = [
    "AgentConfigLoadError",
    "load_agent_config",
]

logger = logging.getLogger(__name__)


class AgentConfigLoadError(Exception):
    """Raised when the agent configuration file cannot be loaded or is invalid.

    The original cause (``OSError``, ``yaml.YAMLError``, or
    ``pydantic.ValidationError``) is always chained via ``__cause__`` so
    that callers can inspect it when needed.

    Attributes:
        path: The filesystem path that was attempted.
        message: Human-readable description of the failure.
    """

    def __init__(self, path: str | Path, message: str) -> None:
        self.path = Path(path)
        self.message = message
        super().__init__(f"[{self.path}] {message}")


def load_agent_config(path: str | Path) -> AgentTeamConfig:
    """Load and validate the YAML agent configuration file.

    Reads the file at *path*, parses it with ``yaml.safe_load``, and
    validates the resulting dict against :class:`~src.config.agent_config.AgentTeamConfig`.

    Args:
        path: Filesystem path to the YAML agent configuration file.
            Relative paths are resolved from the current working directory
            (i.e. the project root when launched via ``uv run``).

    Returns:
        A fully validated :class:`~src.config.agent_config.AgentTeamConfig`
        instance.

    Raises:
        AgentConfigLoadError: If the file does not exist, cannot be read,
            contains invalid YAML syntax, or fails Pydantic schema
            validation.  The original exception is always available via
            ``__cause__``.

    Example::

        from src.config.loader import load_agent_config

        config = load_agent_config("config/agents.yaml")
        print(config.process)           # "sequential"
        print(config.agents[0].id)      # "planner"
    """
    resolved = Path(path)
    logger.debug("Loading agent config from: %s", resolved)

    # ------------------------------------------------------------------ #
    # 1. Read the file
    # ------------------------------------------------------------------ #
    try:
        raw_text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise AgentConfigLoadError(
            resolved,
            f"File not found or cannot be read: {exc}",
        ) from exc

    # ------------------------------------------------------------------ #
    # 2. Parse YAML (safe_load only — never yaml.load)
    # ------------------------------------------------------------------ #
    try:
        raw_data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise AgentConfigLoadError(
            resolved,
            f"YAML parse error: {exc}",
        ) from exc

    if not isinstance(raw_data, dict):
        raise AgentConfigLoadError(
            resolved,
            f"Expected a YAML mapping at the top level, got: {type(raw_data).__name__}",
        )

    # ------------------------------------------------------------------ #
    # 3. Validate against Pydantic schema
    # ------------------------------------------------------------------ #
    try:
        config = AgentTeamConfig.model_validate(raw_data)
    except ValidationError as exc:
        raise AgentConfigLoadError(
            resolved,
            f"Schema validation failed:\n{exc}",
        ) from exc

    logger.info(
        "Agent config loaded: process=%s, agents=%d (%s).",
        config.process,
        len(config.agents),
        ", ".join(a.id for a in config.agents),
    )
    return config
