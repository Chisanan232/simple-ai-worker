"""
Unit tests for :mod:`src.ticket.workflow` (UNIT-WF-01 through UNIT-WF-35).

Covers:
- WorkflowOperation enum values and membership
- OperationConfig dataclass construction, defaults, and immutability
- WorkflowConfig construction (happy-path and missing-key errors)
- status_for: returns correct value, never enforces human_only
- status_for_write: returns correct value, raises PermissionError on human_only (BR-1)
- matches: case-insensitive, whitespace-tolerant
- is_human_only: correct flag reflection
- __repr__: includes all operation names and status values
- Multi-team portability (Team A, Team B, Project C configs)
"""

from __future__ import annotations

import pytest

from src.ticket.workflow import OperationConfig, WorkflowConfig, WorkflowOperation


# ---------------------------------------------------------------------------
# Shared test config helpers
# ---------------------------------------------------------------------------

_TEAM_A = {
    "scan_for_work": {"status_value": "ACCEPTED", "human_only": True},
    "skip_rejected": {"status_value": "REJECTED"},
    "start_development": {"status_value": "IN PROGRESS"},
    "open_for_review": {"status_value": "IN REVIEW"},
    "mark_complete": {"status_value": "COMPLETE"},
    "update_with_context": {"status_value": ""},
}

_TEAM_B = {
    "scan_for_work": {"status_value": "Approved", "human_only": True},
    "skip_rejected": {"status_value": "Cancelled"},
    "start_development": {"status_value": "Developing"},
    "open_for_review": {"status_value": "PR Raised"},
    "mark_complete": {"status_value": "Finished"},
    "update_with_context": {"status_value": ""},
}

_PROJECT_C = {
    "scan_for_work": {"status_value": "Ready for Dev", "human_only": True},
    "skip_rejected": {"status_value": "Will Not Do"},
    "start_development": {"status_value": "Active Development"},
    "open_for_review": {"status_value": "Awaiting Code Review"},
    "mark_complete": {"status_value": "Development Complete"},
    "update_with_context": {"status_value": ""},
}


def _make_cfg(overrides: dict | None = None) -> WorkflowConfig:
    cfg = dict(_TEAM_A)
    if overrides:
        cfg.update(overrides)
    return WorkflowConfig(cfg)


# ===========================================================================
# WorkflowOperation enum
# ===========================================================================


class TestWorkflowOperationEnum:
    def test_all_six_operations_exist(self) -> None:
        """UNIT-WF-01: All six expected operations are members of the enum."""
        names = {op.value for op in WorkflowOperation}
        assert names == {
            "scan_for_work",
            "skip_rejected",
            "start_development",
            "open_for_review",
            "mark_complete",
            "update_with_context",
        }

    def test_enum_values_are_strings(self) -> None:
        """UNIT-WF-02: All operation values are plain strings (WorkflowOperation is str enum)."""
        for op in WorkflowOperation:
            assert isinstance(op.value, str)

    def test_enum_is_accessible_by_value(self) -> None:
        """UNIT-WF-03: Operations can be looked up by string value."""
        assert WorkflowOperation("scan_for_work") is WorkflowOperation.SCAN_FOR_WORK


# ===========================================================================
# OperationConfig dataclass
# ===========================================================================


class TestOperationConfig:
    def test_stores_status_value(self) -> None:
        """UNIT-WF-04: OperationConfig stores status_value correctly."""
        cfg = OperationConfig(status_value="ACCEPTED")
        assert cfg.status_value == "ACCEPTED"

    def test_human_only_defaults_to_false(self) -> None:
        """UNIT-WF-05: human_only defaults to False when not specified."""
        cfg = OperationConfig(status_value="IN PROGRESS")
        assert cfg.human_only is False

    def test_human_only_can_be_set_true(self) -> None:
        """UNIT-WF-06: human_only=True is stored correctly."""
        cfg = OperationConfig(status_value="ACCEPTED", human_only=True)
        assert cfg.human_only is True

    def test_operation_config_is_frozen(self) -> None:
        """UNIT-WF-07: OperationConfig is frozen — attribute assignment raises."""
        cfg = OperationConfig(status_value="ACCEPTED")
        with pytest.raises(Exception):
            cfg.status_value = "REJECTED"  # type: ignore[misc]

    def test_empty_status_value_is_valid(self) -> None:
        """UNIT-WF-08: status_value may be empty string (e.g. update_with_context)."""
        cfg = OperationConfig(status_value="")
        assert cfg.status_value == ""


# ===========================================================================
# WorkflowConfig construction
# ===========================================================================


class TestWorkflowConfigConstruction:
    def test_constructs_successfully_with_all_six_operations(self) -> None:
        """UNIT-WF-09: WorkflowConfig constructs without error when all keys are present."""
        cfg = WorkflowConfig(_TEAM_A)
        assert cfg is not None

    def test_raises_value_error_when_operation_missing(self) -> None:
        """UNIT-WF-10: Missing any one operation key raises ValueError."""
        incomplete = dict(_TEAM_A)
        del incomplete["mark_complete"]
        with pytest.raises(ValueError, match="mark_complete"):
            WorkflowConfig(incomplete)

    def test_raises_value_error_lists_available_keys(self) -> None:
        """UNIT-WF-11: ValueError message lists available keys to aid debugging."""
        incomplete = {k: v for k, v in _TEAM_A.items() if k != "skip_rejected"}
        with pytest.raises(ValueError) as exc_info:
            WorkflowConfig(incomplete)
        # Error message should mention what is available
        assert "skip_rejected" in str(exc_info.value)

    def test_raises_when_all_operations_missing(self) -> None:
        """UNIT-WF-12: Empty dict raises ValueError."""
        with pytest.raises(ValueError):
            WorkflowConfig({})

    def test_human_only_false_when_not_specified(self) -> None:
        """UNIT-WF-13: human_only defaults to False when the key is absent from a block."""
        cfg = WorkflowConfig(_TEAM_A)
        assert cfg.is_human_only(WorkflowOperation.START_DEVELOPMENT) is False

    def test_human_only_true_when_specified(self) -> None:
        """UNIT-WF-14: human_only=True is correctly parsed for scan_for_work."""
        cfg = WorkflowConfig(_TEAM_A)
        assert cfg.is_human_only(WorkflowOperation.SCAN_FOR_WORK) is True


# ===========================================================================
# status_for
# ===========================================================================


class TestStatusFor:
    def test_returns_correct_status_for_each_operation(self) -> None:
        """UNIT-WF-15: status_for returns the configured string for every operation."""
        cfg = WorkflowConfig(_TEAM_A)
        assert cfg.status_for(WorkflowOperation.SCAN_FOR_WORK) == "ACCEPTED"
        assert cfg.status_for(WorkflowOperation.SKIP_REJECTED) == "REJECTED"
        assert cfg.status_for(WorkflowOperation.START_DEVELOPMENT) == "IN PROGRESS"
        assert cfg.status_for(WorkflowOperation.OPEN_FOR_REVIEW) == "IN REVIEW"
        assert cfg.status_for(WorkflowOperation.MARK_COMPLETE) == "COMPLETE"
        assert cfg.status_for(WorkflowOperation.UPDATE_WITH_CONTEXT) == ""

    def test_status_for_does_not_raise_on_human_only(self) -> None:
        """UNIT-WF-16: status_for never raises PermissionError — even for human_only ops."""
        cfg = WorkflowConfig(_TEAM_A)
        # Must NOT raise — status_for is always safe for reads
        result = cfg.status_for(WorkflowOperation.SCAN_FOR_WORK)
        assert result == "ACCEPTED"

    def test_status_for_returns_team_b_values(self) -> None:
        """UNIT-WF-17: status_for returns Team B custom status names."""
        cfg = WorkflowConfig(_TEAM_B)
        assert cfg.status_for(WorkflowOperation.SCAN_FOR_WORK) == "Approved"
        assert cfg.status_for(WorkflowOperation.SKIP_REJECTED) == "Cancelled"
        assert cfg.status_for(WorkflowOperation.START_DEVELOPMENT) == "Developing"
        assert cfg.status_for(WorkflowOperation.OPEN_FOR_REVIEW) == "PR Raised"
        assert cfg.status_for(WorkflowOperation.MARK_COMPLETE) == "Finished"

    def test_status_for_returns_project_c_multi_word_values(self) -> None:
        """UNIT-WF-18: status_for handles multi-word status strings correctly."""
        cfg = WorkflowConfig(_PROJECT_C)
        assert cfg.status_for(WorkflowOperation.SCAN_FOR_WORK) == "Ready for Dev"
        assert cfg.status_for(WorkflowOperation.OPEN_FOR_REVIEW) == "Awaiting Code Review"
        assert cfg.status_for(WorkflowOperation.MARK_COMPLETE) == "Development Complete"


# ===========================================================================
# status_for_write
# ===========================================================================


class TestStatusForWrite:
    def test_returns_status_for_non_human_only_operation(self) -> None:
        """UNIT-WF-19: status_for_write returns the status for non-human-only ops."""
        cfg = WorkflowConfig(_TEAM_A)
        assert cfg.status_for_write(WorkflowOperation.START_DEVELOPMENT) == "IN PROGRESS"
        assert cfg.status_for_write(WorkflowOperation.OPEN_FOR_REVIEW) == "IN REVIEW"
        assert cfg.status_for_write(WorkflowOperation.MARK_COMPLETE) == "COMPLETE"
        assert cfg.status_for_write(WorkflowOperation.SKIP_REJECTED) == "REJECTED"

    def test_raises_permission_error_for_human_only_operation(self) -> None:
        """UNIT-WF-20: status_for_write raises PermissionError for human_only op (BR-1)."""
        cfg = WorkflowConfig(_TEAM_A)
        with pytest.raises(PermissionError):
            cfg.status_for_write(WorkflowOperation.SCAN_FOR_WORK)

    def test_permission_error_message_contains_operation_name(self) -> None:
        """UNIT-WF-21: PermissionError message identifies the blocked operation."""
        cfg = WorkflowConfig(_TEAM_A)
        with pytest.raises(PermissionError) as exc_info:
            cfg.status_for_write(WorkflowOperation.SCAN_FOR_WORK)
        assert "scan_for_work" in str(exc_info.value)

    def test_permission_error_message_contains_status_value(self) -> None:
        """UNIT-WF-22: PermissionError message includes the blocked status string."""
        cfg = WorkflowConfig(_TEAM_A)
        with pytest.raises(PermissionError) as exc_info:
            cfg.status_for_write(WorkflowOperation.SCAN_FOR_WORK)
        assert "ACCEPTED" in str(exc_info.value)

    def test_permission_error_message_references_br1(self) -> None:
        """UNIT-WF-23: PermissionError message references BR-1."""
        cfg = WorkflowConfig(_TEAM_A)
        with pytest.raises(PermissionError) as exc_info:
            cfg.status_for_write(WorkflowOperation.SCAN_FOR_WORK)
        assert "BR-1" in str(exc_info.value)

    def test_no_error_for_update_with_context_empty_status(self) -> None:
        """UNIT-WF-24: UPDATE_WITH_CONTEXT (empty status, not human_only) does not raise."""
        cfg = WorkflowConfig(_TEAM_A)
        result = cfg.status_for_write(WorkflowOperation.UPDATE_WITH_CONTEXT)
        assert result == ""

    def test_team_b_human_only_also_raises(self) -> None:
        """UNIT-WF-25: BR-1 is enforced regardless of status string vocabulary (Team B)."""
        cfg = WorkflowConfig(_TEAM_B)
        with pytest.raises(PermissionError):
            cfg.status_for_write(WorkflowOperation.SCAN_FOR_WORK)


# ===========================================================================
# matches
# ===========================================================================


class TestMatches:
    def test_exact_match_returns_true(self) -> None:
        """UNIT-WF-26: matches returns True when raw_status exactly equals configured value."""
        cfg = WorkflowConfig(_TEAM_A)
        assert cfg.matches(WorkflowOperation.SCAN_FOR_WORK, "ACCEPTED") is True

    def test_case_insensitive_match(self) -> None:
        """UNIT-WF-27: matches is case-insensitive."""
        cfg = WorkflowConfig(_TEAM_A)
        assert cfg.matches(WorkflowOperation.SCAN_FOR_WORK, "accepted") is True
        assert cfg.matches(WorkflowOperation.SCAN_FOR_WORK, "Accepted") is True
        assert cfg.matches(WorkflowOperation.SCAN_FOR_WORK, "ACCEPTED") is True

    def test_whitespace_stripped_in_comparison(self) -> None:
        """UNIT-WF-28: Leading/trailing whitespace is stripped before comparison."""
        cfg = WorkflowConfig(_TEAM_A)
        assert cfg.matches(WorkflowOperation.SKIP_REJECTED, "  REJECTED  ") is True

    def test_different_value_returns_false(self) -> None:
        """UNIT-WF-29: matches returns False for a different status string."""
        cfg = WorkflowConfig(_TEAM_A)
        assert cfg.matches(WorkflowOperation.SCAN_FOR_WORK, "IN PROGRESS") is False

    def test_matches_empty_status_value(self) -> None:
        """UNIT-WF-30: matches works correctly for empty status_value (update_with_context)."""
        cfg = WorkflowConfig(_TEAM_A)
        assert cfg.matches(WorkflowOperation.UPDATE_WITH_CONTEXT, "") is True
        assert cfg.matches(WorkflowOperation.UPDATE_WITH_CONTEXT, "   ") is True

    def test_matches_team_b_values(self) -> None:
        """UNIT-WF-31: matches works with Team B's custom status names."""
        cfg = WorkflowConfig(_TEAM_B)
        assert cfg.matches(WorkflowOperation.SCAN_FOR_WORK, "Approved") is True
        assert cfg.matches(WorkflowOperation.SKIP_REJECTED, "CANCELLED") is True  # case-insensitive


# ===========================================================================
# is_human_only
# ===========================================================================


class TestIsHumanOnly:
    def test_scan_for_work_is_human_only(self) -> None:
        """UNIT-WF-32: is_human_only returns True for scan_for_work (Team A)."""
        cfg = WorkflowConfig(_TEAM_A)
        assert cfg.is_human_only(WorkflowOperation.SCAN_FOR_WORK) is True

    def test_other_operations_are_not_human_only(self) -> None:
        """UNIT-WF-33: All write operations return False for is_human_only (Team A)."""
        cfg = WorkflowConfig(_TEAM_A)
        for op in WorkflowOperation:
            if op is not WorkflowOperation.SCAN_FOR_WORK:
                assert cfg.is_human_only(op) is False, f"Expected {op.value} to not be human_only"


# ===========================================================================
# __repr__
# ===========================================================================


class TestRepr:
    def test_repr_contains_all_operation_names(self) -> None:
        """UNIT-WF-34: __repr__ includes all six operation names."""
        cfg = WorkflowConfig(_TEAM_A)
        r = repr(cfg)
        for op in WorkflowOperation:
            assert op.value in r

    def test_repr_contains_status_values(self) -> None:
        """UNIT-WF-35: __repr__ includes configured status strings."""
        cfg = WorkflowConfig(_TEAM_A)
        r = repr(cfg)
        assert "ACCEPTED" in r
        assert "IN PROGRESS" in r
        assert "COMPLETE" in r

    def test_repr_marks_human_only_operations(self) -> None:
        """__repr__ marks human-only operations visibly."""
        cfg = WorkflowConfig(_TEAM_A)
        r = repr(cfg)
        assert "human-only" in r

