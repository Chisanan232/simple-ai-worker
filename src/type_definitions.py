"""
Type definitions for the Python package.

This module provides centralized type aliases and type definitions following
PEP 561, PEP 484, PEP 585, and PEP 695 standards for static type checking with MyPy.

Type aliases use the modern `type` statement (PEP 695) introduced in Python 3.12,
which provides better type inference and cleaner syntax compared to TypeAlias.

Type Hierarchy:
    - JSON types: Basic JSON-compatible types
    - Slack types: Slack-specific type definitions
    - Event types: Event handling type definitions
    - Handler types: Handler function signatures
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

__all__ = [
]
