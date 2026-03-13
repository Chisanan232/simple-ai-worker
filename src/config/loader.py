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
- ``${VAR_NAME}`` placeholders in ``mcp_servers[*].headers`` values are
  resolved against ``AppSettings`` (and the process environment as a
  fallback) *before* Pydantic validation.  This keeps secret tokens out of
  the YAML file while still providing a readable config format.

Usage::

    from src.config.loader import load_agent_config
    from src.config.settings import get_settings

    config = load_agent_config("config/agents.yaml", get_settings())
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
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import yaml
from pydantic import ValidationError

from .agent_config import AgentTeamConfig

if TYPE_CHECKING:
    from src.config.settings import AppSettings

__all__: List[str] = [
    "AgentConfigLoadError",
    "load_agent_config",
]

logger = logging.getLogger(__name__)

# Matches ${UPPER_CASE_VAR_NAME} placeholders in YAML string values.
_ENV_PLACEHOLDER_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


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


# ---------------------------------------------------------------------------
# Placeholder resolution helpers
# ---------------------------------------------------------------------------


def _resolve_placeholder(var_name: str, settings: Optional["AppSettings"]) -> Optional[str]:
    """Return the resolved value for *var_name* from settings or ``os.environ``.

    Checks ``AppSettings`` first (handles ``SecretStr`` fields transparently),
    then falls back to ``os.environ``.  Returns ``None`` if the variable is
    not found in either source.

    Args:
        var_name: The environment variable / settings field name to look up.
        settings: Optional :class:`~src.config.settings.AppSettings` instance.

    Returns:
        The resolved string value, or ``None`` if not found.
    """
    if settings is not None:
        field_val = getattr(settings, var_name, None)
        if field_val is not None:
            from pydantic import SecretStr  # local import — avoid circular dep
            return field_val.get_secret_value() if isinstance(field_val, SecretStr) else str(field_val)
    # Fall back to the process environment.
    return os.environ.get(var_name)


def _resolve_string_placeholders(value: str, settings: Optional["AppSettings"]) -> str:
    """Replace all ``${VAR_NAME}`` tokens in *value* with resolved values.

    Placeholders that cannot be resolved (variable not found in settings or
    environment) are left unchanged so that downstream validators can detect
    and warn about them.

    Args:
        value: The string that may contain ``${…}`` placeholders.
        settings: Optional :class:`~src.config.settings.AppSettings` used
            as the primary resolution source.

    Returns:
        The string with all resolvable placeholders substituted.
    """
    def _replace(match: re.Match[str]) -> str:
        resolved = _resolve_placeholder(match.group(1), settings)
        if resolved is None:
            logger.warning(
                "MCP header placeholder '${%s}' could not be resolved — "
                "token not found in AppSettings or environment.  "
                "The literal placeholder will be used as-is.",
                match.group(1),
            )
            return match.group(0)  # leave unchanged
        return resolved

    return _ENV_PLACEHOLDER_RE.sub(_replace, value)


def _apply_header_placeholders(raw_data: Dict[str, Any], settings: Optional["AppSettings"]) -> None:
    """Resolve ``${VAR}`` placeholders inside ``mcp_servers[*].headers`` in-place.

    Walks the ``mcp_servers`` section of the raw parsed YAML dict and replaces
    placeholder tokens in every header *value* string.  Keys are left
    unchanged.  Operates in-place — no return value.

    Args:
        raw_data: The top-level dict produced by ``yaml.safe_load``.
        settings: Optional :class:`~src.config.settings.AppSettings` used
            for resolution.  When ``None`` only ``os.environ`` is consulted.
    """
    mcp_servers = raw_data.get("mcp_servers")
    if not isinstance(mcp_servers, dict):
        return
    for server_id, server_cfg in mcp_servers.items():
        if not isinstance(server_cfg, dict):
            continue
        headers = server_cfg.get("headers")
        if not isinstance(headers, dict):
            continue
        resolved_headers: Dict[str, str] = {}
        for header_key, header_val in headers.items():
            if isinstance(header_val, str):
                resolved_headers[header_key] = _resolve_string_placeholders(header_val, settings)
            else:
                resolved_headers[header_key] = header_val
        server_cfg["headers"] = resolved_headers
        logger.debug(
            "Resolved header placeholders for MCP server '%s'.", server_id
        )


def load_agent_config(
    path: str | Path,
    settings: Optional["AppSettings"] = None,
) -> AgentTeamConfig:
    """Load and validate the YAML agent configuration file.

    Reads the file at *path*, parses it with ``yaml.safe_load``, resolves
    any ``${VAR_NAME}`` placeholders in ``mcp_servers[*].headers``, and
    validates the resulting dict against
    :class:`~src.config.agent_config.AgentTeamConfig`.

    Args:
        path: Filesystem path to the YAML agent configuration file.
            Relative paths are resolved from the current working directory
            (i.e. the project root when launched via ``uv run``).
        settings: Optional :class:`~src.config.settings.AppSettings` instance
            used to resolve ``${VAR_NAME}`` placeholders in MCP server header
            values.  When ``None``, only ``os.environ`` is consulted for
            placeholder resolution.

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
        from src.config.settings import get_settings

        config = load_agent_config("config/agents.yaml", get_settings())
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
    # 2b. Resolve ${VAR} placeholders in mcp_servers headers
    # ------------------------------------------------------------------ #
    _apply_header_placeholders(raw_data, settings)

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
        "Agent config loaded: process=%s, mcp_servers=%d (%s), agents=%d (%s).",
        config.process,
        len(config.mcp_servers),
        ", ".join(config.mcp_servers.keys()) if config.mcp_servers else "none",
        len(config.agents),
        ", ".join(a.id for a in config.agents),
    )
    return config
