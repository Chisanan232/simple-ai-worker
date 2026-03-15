"""
Integration tests for WorkflowConfig (INT-WC-01 through INT-WC-10).

Verifies:
- Correct status string resolution (read and write)
- BR-1 enforcement: PermissionError on write to human_only operations
- BR-3: matches() for skip_rejected
- Missing operation raises ValueError at construction time
- Multi-team portability (Team B and Project C configs)
"""

from __future__ import annotations

import pytest

from src.ticket.workflow import WorkflowConfig, WorkflowOperation

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: dict) -> dict:
    """Build a complete workflow config dict with optional overrides."""
    base = {
        "scan_for_work": {"status_value": "ACCEPTED", "human_only": True},
        "skip_rejected": {"status_value": "REJECTED"},
        "start_development": {"status_value": "IN PROGRESS"},
        "open_for_review": {"status_value": "IN REVIEW"},
        "mark_complete": {"status_value": "COMPLETE"},
        "update_with_context": {"status_value": ""},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# INT-WC-01 — status_for returns configured value
# ---------------------------------------------------------------------------


class TestStatusFor:
    def test_status_for_returns_accepted_for_scan_for_work(self, default_workflow_config: WorkflowConfig) -> None:
        """INT-WC-01: status_for(SCAN_FOR_WORK) returns 'ACCEPTED'."""
        result = default_workflow_config.status_for(WorkflowOperation.SCAN_FOR_WORK)
        assert result == "ACCEPTED"

    def test_status_for_returns_all_operations(self, default_workflow_config: WorkflowConfig) -> None:
        """status_for works for all six operations."""
        assert default_workflow_config.status_for(WorkflowOperation.SKIP_REJECTED) == "REJECTED"
        assert default_workflow_config.status_for(WorkflowOperation.START_DEVELOPMENT) == "IN PROGRESS"
        assert default_workflow_config.status_for(WorkflowOperation.OPEN_FOR_REVIEW) == "IN REVIEW"
        assert default_workflow_config.status_for(WorkflowOperation.MARK_COMPLETE) == "COMPLETE"
        assert default_workflow_config.status_for(WorkflowOperation.UPDATE_WITH_CONTEXT) == ""


# ---------------------------------------------------------------------------
# INT-WC-02 — status_for_write raises PermissionError when human_only (BR-1)
# ---------------------------------------------------------------------------


class TestStatusForWrite:
    def test_status_for_write_raises_for_human_only_scan_for_work(
        self, default_workflow_config: WorkflowConfig
    ) -> None:
        """INT-WC-02: status_for_write(SCAN_FOR_WORK) raises PermissionError (BR-1)."""
        with pytest.raises(PermissionError) as exc_info:
            default_workflow_config.status_for_write(WorkflowOperation.SCAN_FOR_WORK)
        assert "human-only" in str(exc_info.value).lower() or "BR-1" in str(exc_info.value)

    def test_status_for_write_succeeds_for_start_development(self, default_workflow_config: WorkflowConfig) -> None:
        """INT-WC-03: status_for_write(START_DEVELOPMENT) returns 'IN PROGRESS'."""
        result = default_workflow_config.status_for_write(WorkflowOperation.START_DEVELOPMENT)
        assert result == "IN PROGRESS"

    def test_status_for_write_succeeds_for_mark_complete(self, default_workflow_config: WorkflowConfig) -> None:
        """INT-WC-04: status_for_write(MARK_COMPLETE) returns 'COMPLETE'."""
        result = default_workflow_config.status_for_write(WorkflowOperation.MARK_COMPLETE)
        assert result == "COMPLETE"

    def test_status_for_write_succeeds_for_open_for_review(self, default_workflow_config: WorkflowConfig) -> None:
        """status_for_write(OPEN_FOR_REVIEW) returns 'IN REVIEW'."""
        result = default_workflow_config.status_for_write(WorkflowOperation.OPEN_FOR_REVIEW)
        assert result == "IN REVIEW"

    def test_status_for_write_succeeds_for_skip_rejected(self, default_workflow_config: WorkflowConfig) -> None:
        """status_for_write(SKIP_REJECTED) returns 'REJECTED' (not human_only)."""
        result = default_workflow_config.status_for_write(WorkflowOperation.SKIP_REJECTED)
        assert result == "REJECTED"


# ---------------------------------------------------------------------------
# INT-WC-05 — matches() is case-insensitive
# ---------------------------------------------------------------------------


class TestMatches:
    def test_matches_is_case_insensitive(self, default_workflow_config: WorkflowConfig) -> None:
        """INT-WC-05: matches(SKIP_REJECTED, 'rejected') returns True (case-insensitive)."""
        assert default_workflow_config.matches(WorkflowOperation.SKIP_REJECTED, "rejected") is True
        assert default_workflow_config.matches(WorkflowOperation.SKIP_REJECTED, "REJECTED") is True
        assert default_workflow_config.matches(WorkflowOperation.SKIP_REJECTED, "Rejected") is True

    def test_matches_returns_false_for_different_status(self, default_workflow_config: WorkflowConfig) -> None:
        """INT-WC-06: matches(SKIP_REJECTED, 'IN PROGRESS') returns False."""
        assert default_workflow_config.matches(WorkflowOperation.SKIP_REJECTED, "IN PROGRESS") is False

    def test_matches_strips_whitespace(self, default_workflow_config: WorkflowConfig) -> None:
        """matches() strips leading/trailing whitespace before comparing."""
        assert default_workflow_config.matches(WorkflowOperation.SKIP_REJECTED, "  REJECTED  ") is True

    def test_matches_for_scan_for_work(self, default_workflow_config: WorkflowConfig) -> None:
        """matches(SCAN_FOR_WORK, 'accepted') returns True."""
        assert default_workflow_config.matches(WorkflowOperation.SCAN_FOR_WORK, "accepted") is True

    def test_matches_returns_false_for_unrelated(self, default_workflow_config: WorkflowConfig) -> None:
        assert default_workflow_config.matches(WorkflowOperation.SCAN_FOR_WORK, "IN PROGRESS") is False


# ---------------------------------------------------------------------------
# INT-WC-07 — raises ValueError on missing operation
# ---------------------------------------------------------------------------


class TestMissingOperation:
    def test_raises_on_missing_operation_in_config(self) -> None:
        """INT-WC-07: Config missing 'start_development' key raises ValueError."""
        incomplete = {
            "scan_for_work": {"status_value": "ACCEPTED", "human_only": True},
            "skip_rejected": {"status_value": "REJECTED"},
            # start_development MISSING
            "open_for_review": {"status_value": "IN REVIEW"},
            "mark_complete": {"status_value": "COMPLETE"},
            "update_with_context": {"status_value": ""},
        }
        with pytest.raises(ValueError, match="start_development"):
            WorkflowConfig(incomplete)

    def test_raises_on_completely_empty_config(self) -> None:
        """Empty config dict raises ValueError for first missing operation."""
        with pytest.raises(ValueError):
            WorkflowConfig({})


# ---------------------------------------------------------------------------
# INT-WC-08 — Team B config portability
# ---------------------------------------------------------------------------


class TestTeamBConfig:
    def test_team_b_config_different_status_names(self, team_b_workflow_config: WorkflowConfig) -> None:
        """INT-WC-08: Team B config returns 'Approved' for scan_for_work."""
        assert team_b_workflow_config.status_for(WorkflowOperation.SCAN_FOR_WORK) == "Approved"

    def test_team_b_start_development_returns_developing(self, team_b_workflow_config: WorkflowConfig) -> None:
        """status_for_write(START_DEVELOPMENT) returns 'Developing' for Team B."""
        result = team_b_workflow_config.status_for_write(WorkflowOperation.START_DEVELOPMENT)
        assert result == "Developing"

    def test_team_b_mark_complete_returns_finished(self, team_b_workflow_config: WorkflowConfig) -> None:
        assert team_b_workflow_config.status_for_write(WorkflowOperation.MARK_COMPLETE) == "Finished"

    def test_team_b_scan_for_work_still_human_only(self, team_b_workflow_config: WorkflowConfig) -> None:
        """BR-1 still enforced for Team B — scan_for_work is human_only."""
        with pytest.raises(PermissionError):
            team_b_workflow_config.status_for_write(WorkflowOperation.SCAN_FOR_WORK)


# ---------------------------------------------------------------------------
# INT-WC-09 — Project C multi-word status names
# ---------------------------------------------------------------------------


class TestTeamCConfig:
    def test_team_c_multi_word_status_names(self, team_c_workflow_config: WorkflowConfig) -> None:
        """INT-WC-09: Multi-word status string returned correctly."""
        result = team_c_workflow_config.status_for_write(WorkflowOperation.START_DEVELOPMENT)
        assert result == "Active Development"

    def test_team_c_open_for_review_multi_word(self, team_c_workflow_config: WorkflowConfig) -> None:
        assert team_c_workflow_config.status_for_write(WorkflowOperation.OPEN_FOR_REVIEW) == "Awaiting Code Review"


# ---------------------------------------------------------------------------
# INT-WC-10 — is_human_only reflects config
# ---------------------------------------------------------------------------


class TestIsHumanOnly:
    def test_is_human_only_true_for_scan_for_work(self, default_workflow_config: WorkflowConfig) -> None:
        """INT-WC-10: is_human_only(SCAN_FOR_WORK) returns True."""
        assert default_workflow_config.is_human_only(WorkflowOperation.SCAN_FOR_WORK) is True

    def test_is_human_only_false_for_start_development(self, default_workflow_config: WorkflowConfig) -> None:
        assert default_workflow_config.is_human_only(WorkflowOperation.START_DEVELOPMENT) is False

    def test_is_human_only_false_for_all_write_operations(self, default_workflow_config: WorkflowConfig) -> None:
        for op in [
            WorkflowOperation.SKIP_REJECTED,
            WorkflowOperation.START_DEVELOPMENT,
            WorkflowOperation.OPEN_FOR_REVIEW,
            WorkflowOperation.MARK_COMPLETE,
            WorkflowOperation.UPDATE_WITH_CONTEXT,
        ]:
            assert default_workflow_config.is_human_only(op) is False, f"Expected {op} to not be human_only"
