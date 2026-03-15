"""
Workflow operation abstraction (Phase 7 + Phase 9, v6 design).

The core design principle of this module: **the AI layer only knows about
operation names, never about status strings**.

Classes
-------
WorkflowOperation
    Enum of the eight stable, semantic operations the Dev / Dev Lead agents
    perform.  These never change; only the *status strings* they map to
    change per team.
OperationConfig
    Configuration for a single operation (status string + human_only flag).
WorkflowConfig
    Maps each WorkflowOperation to a team-specific ticket status string.
    Loaded from the ``workflow:`` block in ``agents.yaml``.  Enforces BR-1
    at the Python layer.

Business Rules Enforced
-----------------------
- **BR-1:** Any operation with ``human_only: true`` raises :exc:`PermissionError`
  when :meth:`WorkflowConfig.status_for_write` is called.  The AI may read
  (query) the status but never write (transition) to it.
- **BR-3:** ``SKIP_REJECTED`` is used by callers to build the exclusion filter.
  :meth:`WorkflowConfig.matches` performs case-insensitive comparison.
- **BR-10:** ``IN_PLANNING`` is ``human_only: true`` — only a human engineer
  sets a task to this status after receiving the Dev Agent's plan for review.

Backward Compatibility
----------------------
``WorkflowConfig.__init__`` accepts dicts with only the original six keys.
The two new keys (``open_for_dev``, ``in_planning``) default to empty-string
/ non-human-only when absent so that existing ``agents.yaml`` files continue
to work without modification.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

__all__: List[str] = [
    "WorkflowOperation",
    "OperationConfig",
    "WorkflowConfig",
]


class WorkflowOperation(str, Enum):
    """The named operations the Dev / Dev Lead agents perform in the workflow.

    These are **stable, semantic operation names** that never change.
    What changes between teams is the *ticket status string* each operation
    maps to — that is configurable in ``agents.yaml``.

    Values
    ------
    SCAN_FOR_WORK
        **Read** — find tickets whose status signals "ready to develop".
        Always ``human_only: true``; AI queries this status but never writes it.
    SKIP_REJECTED
        **Read** — identify cancelled/rejected tickets to silently skip.
    OPEN_FOR_DEV
        **Write (Dev Lead)** — status set by the Dev Lead AI when creating
        sub-tasks from a decomposed epic.  Signals "waiting for Dev Agent
        to produce a development plan".  Example status string: ``"OPEN"``.
    IN_PLANNING
        **Read (Dev Agent) / Write (Human only)** — status set by the human
        engineer after receiving the Dev Agent's development plan for review.
        ``human_only: true`` — the AI may query this status but must never
        transition a ticket to it.  Example: ``"IN PLANNING"``.
    START_DEVELOPMENT
        **Write** — transition when the Dev Agent starts coding.
    OPEN_FOR_REVIEW
        **Write** — transition when the Dev Agent opens a GitHub PR.
    MARK_COMPLETE
        **Write** — transition when the PR is merged.
    UPDATE_WITH_CONTEXT
        **Comment only** — no state transition; post discussion summary.
    """

    SCAN_FOR_WORK = "scan_for_work"
    SKIP_REJECTED = "skip_rejected"
    OPEN_FOR_DEV = "open_for_dev"
    IN_PLANNING = "in_planning"
    START_DEVELOPMENT = "start_development"
    OPEN_FOR_REVIEW = "open_for_review"
    MARK_COMPLETE = "mark_complete"
    UPDATE_WITH_CONTEXT = "update_with_context"


@dataclass(frozen=True)
class OperationConfig:
    """Configuration for a single workflow operation.

    Attributes:
        status_value: The exact status string used in the team's ticket tracker
            (e.g. ``"ACCEPTED"``, ``"IN PROGRESS"``, ``"Developing"``).
        human_only:   When ``True`` the AI may *read* this status (for queries)
            but must **never write** it.  Calling
            :meth:`WorkflowConfig.status_for_write` raises :exc:`PermissionError`.
            This enforces BR-1 at the Python layer without any hardcoded status string.
    """

    status_value: str
    human_only: bool = False


class WorkflowConfig:
    """Maps :class:`WorkflowOperation` values to team-specific status strings.

    Loaded once at startup from the ``workflow:`` YAML block inside an agent's
    config.  All status string lookups go through this object — no status
    string is ever hardcoded in application or scheduler code.

    Parameters
    ----------
    config:
        A dict shaped like the YAML ``workflow:`` block::

            {
                "scan_for_work":     {"status_value": "ACCEPTED", "human_only": True},
                "skip_rejected":     {"status_value": "REJECTED"},
                "start_development": {"status_value": "IN PROGRESS"},
                "open_for_review":   {"status_value": "IN REVIEW"},
                "mark_complete":     {"status_value": "COMPLETE"},
                "update_with_context": {"status_value": ""},
            }

    Raises
    ------
    ValueError
        If any :class:`WorkflowOperation` key is missing from *config*.

    Examples
    --------
    >>> cfg = WorkflowConfig({
    ...     "scan_for_work": {"status_value": "ACCEPTED", "human_only": True},
    ...     "skip_rejected": {"status_value": "REJECTED"},
    ...     "start_development": {"status_value": "IN PROGRESS"},
    ...     "open_for_review": {"status_value": "IN REVIEW"},
    ...     "mark_complete": {"status_value": "COMPLETE"},
    ...     "update_with_context": {"status_value": ""},
    ... })
    >>> cfg.status_for(WorkflowOperation.SCAN_FOR_WORK)
    'ACCEPTED'
    >>> cfg.status_for_write(WorkflowOperation.SCAN_FOR_WORK)  # raises PermissionError
    """

    # Original six operations — must be present in every config dict.
    _REQUIRED_OPS = frozenset(
        {
            WorkflowOperation.SCAN_FOR_WORK,
            WorkflowOperation.SKIP_REJECTED,
            WorkflowOperation.START_DEVELOPMENT,
            WorkflowOperation.OPEN_FOR_REVIEW,
            WorkflowOperation.MARK_COMPLETE,
            WorkflowOperation.UPDATE_WITH_CONTEXT,
        }
    )

    # Phase-9 additions — optional; default to empty / non-human-only so that
    # existing agents.yaml files that only define the original six keys continue
    # to work without modification.
    _OPTIONAL_OP_DEFAULTS: Dict[str, OperationConfig] = {
        WorkflowOperation.OPEN_FOR_DEV.value: OperationConfig(status_value="", human_only=False),
        WorkflowOperation.IN_PLANNING.value: OperationConfig(status_value="", human_only=True),
    }

    def __init__(self, config: Dict[str, dict]) -> None:
        self._ops: Dict[WorkflowOperation, OperationConfig] = {}
        for op in WorkflowOperation:
            raw = config.get(op.value)
            if raw is None:
                if op in self._REQUIRED_OPS:
                    raise ValueError(
                        f"Missing workflow operation config for '{op.value}'. "
                        "The original six operations must be defined in the "
                        "'workflow:' YAML block. "
                        f"Available keys: {sorted(config.keys())}."
                    )
                # Optional new op — use its default OperationConfig.
                self._ops[op] = self._OPTIONAL_OP_DEFAULTS[op.value]
            else:
                self._ops[op] = OperationConfig(
                    status_value=raw.get("status_value", ""),
                    human_only=bool(raw.get("human_only", False)),
                )

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def status_for(self, operation: WorkflowOperation) -> str:
        """Return the configured status string for *read* access.

        Safe to call for query/filter purposes.  Does **not** enforce the
        ``human_only`` flag — that guard only applies to write operations.

        Args:
            operation: The workflow operation whose configured status is needed.

        Returns:
            The status string exactly as configured in ``agents.yaml``.
        """
        return self._ops[operation].status_value

    def status_for_write(self, operation: WorkflowOperation) -> str:
        """Return the configured status string for a *write* (transition).

        Enforces BR-1: raises :exc:`PermissionError` if the operation is
        marked ``human_only: true``.  This prevents the AI from accidentally
        transitioning a ticket to a human-gated state even if a developer
        mistakenly calls this method.

        Args:
            operation: The workflow operation whose status will be written.

        Returns:
            The status string to use for the transition.

        Raises:
            PermissionError: If ``human_only: true`` is set for this operation.
        """
        op_cfg = self._ops[operation]
        if op_cfg.human_only:
            raise PermissionError(
                f"Operation '{operation.value}' (status: '{op_cfg.status_value}') "
                "is human-only — the AI may not write this status. (BR-1)"
            )
        return op_cfg.status_value

    def matches(self, operation: WorkflowOperation, raw_status: str) -> bool:
        """Check whether *raw_status* matches the configured status for *operation*.

        Comparison is case-insensitive and strips leading/trailing whitespace.

        Args:
            operation:  The workflow operation to compare against.
            raw_status: The raw status string from the ticket tracker.

        Returns:
            ``True`` if the status strings match (case-insensitive).
        """
        configured = self._ops[operation].status_value
        return configured.strip().lower() == raw_status.strip().lower()

    def is_human_only(self, operation: WorkflowOperation) -> bool:
        """Return ``True`` if this operation's target state is human-only.

        Args:
            operation: The workflow operation to check.

        Returns:
            ``True`` when ``human_only: true`` is set for this operation.
        """
        return self._ops[operation].human_only

    def __repr__(self) -> str:
        pairs = ", ".join(
            f"{op.value}={cfg.status_value!r}{'(human-only)' if cfg.human_only else ''}"
            for op, cfg in self._ops.items()
        )
        return f"WorkflowConfig({pairs})"
