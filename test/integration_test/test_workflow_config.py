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


# ---------------------------------------------------------------------------
# INT-WF-01 through INT-WF-04 — Phase 9: new operations (OPEN_FOR_DEV, IN_PLANNING)
# ---------------------------------------------------------------------------


class TestPhase9WorkflowOperations:
    """Tests for the two Phase-9 additions: OPEN_FOR_DEV and IN_PLANNING."""

    def _make_full_config(self) -> dict:
        """Config including all eight operations."""
        return {
            "scan_for_work": {"status_value": "ACCEPTED", "human_only": True},
            "skip_rejected": {"status_value": "REJECTED"},
            "open_for_dev": {"status_value": "OPEN", "human_only": False},
            "in_planning": {"status_value": "IN PLANNING", "human_only": True},
            "start_development": {"status_value": "IN PROGRESS"},
            "open_for_review": {"status_value": "IN REVIEW"},
            "mark_complete": {"status_value": "COMPLETE"},
            "update_with_context": {"status_value": ""},
        }

    def test_int_wf_01_full_eight_key_config_accepted(self) -> None:
        """INT-WF-01: WorkflowConfig accepts open_for_dev and in_planning in the config dict."""
        cfg = WorkflowConfig(self._make_full_config())
        assert cfg.status_for(WorkflowOperation.OPEN_FOR_DEV) == "OPEN"
        assert cfg.status_for(WorkflowOperation.IN_PLANNING) == "IN PLANNING"

    def test_int_wf_02_backward_compat_six_key_config(self) -> None:
        """INT-WF-02: Old six-key dict still works; new ops default to empty string."""
        six_key = {
            "scan_for_work": {"status_value": "ACCEPTED", "human_only": True},
            "skip_rejected": {"status_value": "REJECTED"},
            "start_development": {"status_value": "IN PROGRESS"},
            "open_for_review": {"status_value": "IN REVIEW"},
            "mark_complete": {"status_value": "COMPLETE"},
            "update_with_context": {"status_value": ""},
        }
        cfg = WorkflowConfig(six_key)  # must not raise
        # New optional ops default to empty string
        assert cfg.status_for(WorkflowOperation.OPEN_FOR_DEV) == ""
        assert cfg.status_for(WorkflowOperation.IN_PLANNING) == ""

    def test_int_wf_03_in_planning_is_human_only(self) -> None:
        """INT-WF-03: status_for_write(IN_PLANNING) raises PermissionError (human_only)."""
        cfg = WorkflowConfig(self._make_full_config())
        with pytest.raises(PermissionError, match="human-only"):
            cfg.status_for_write(WorkflowOperation.IN_PLANNING)

    def test_int_wf_03b_in_planning_default_is_human_only(self) -> None:
        """INT-WF-03b: Even when using the six-key default, IN_PLANNING defaults to human_only=True."""
        six_key = {
            "scan_for_work": {"status_value": "ACCEPTED", "human_only": True},
            "skip_rejected": {"status_value": "REJECTED"},
            "start_development": {"status_value": "IN PROGRESS"},
            "open_for_review": {"status_value": "IN REVIEW"},
            "mark_complete": {"status_value": "COMPLETE"},
            "update_with_context": {"status_value": ""},
        }
        cfg = WorkflowConfig(six_key)
        with pytest.raises(PermissionError):
            cfg.status_for_write(WorkflowOperation.IN_PLANNING)

    def test_int_wf_03c_open_for_dev_is_writable(self) -> None:
        """INT-WF-03c: status_for_write(OPEN_FOR_DEV) succeeds (not human_only)."""
        cfg = WorkflowConfig(self._make_full_config())
        result = cfg.status_for_write(WorkflowOperation.OPEN_FOR_DEV)
        assert result == "OPEN"

    def test_int_wf_04_workflow_config_model_includes_new_ops(self) -> None:
        """INT-WF-04: WorkflowConfigModel.to_workflow_config() includes open_for_dev and in_planning."""
        from src.config.agent_config import WorkflowConfigModel, WorkflowOperationConfig

        model = WorkflowConfigModel(
            scan_for_work=WorkflowOperationConfig(status_value="ACCEPTED", human_only=True),
            skip_rejected=WorkflowOperationConfig(status_value="REJECTED"),
            open_for_dev=WorkflowOperationConfig(status_value="OPEN", human_only=False),
            in_planning=WorkflowOperationConfig(status_value="IN PLANNING", human_only=True),
            start_development=WorkflowOperationConfig(status_value="IN PROGRESS"),
            open_for_review=WorkflowOperationConfig(status_value="IN REVIEW"),
            mark_complete=WorkflowOperationConfig(status_value="COMPLETE"),
            update_with_context=WorkflowOperationConfig(status_value=""),
        )
        wf = model.to_workflow_config()
        assert wf.status_for(WorkflowOperation.OPEN_FOR_DEV) == "OPEN"
        assert wf.status_for(WorkflowOperation.IN_PLANNING) == "IN PLANNING"
        assert wf.is_human_only(WorkflowOperation.IN_PLANNING) is True
        assert wf.is_human_only(WorkflowOperation.OPEN_FOR_DEV) is False

    def test_int_wf_04b_workflow_config_model_defaults(self) -> None:
        """INT-WF-04b: WorkflowConfigModel defaults for open_for_dev and in_planning are empty."""
        from src.config.agent_config import WorkflowConfigModel, WorkflowOperationConfig

        model = WorkflowConfigModel(
            scan_for_work=WorkflowOperationConfig(status_value="ACCEPTED", human_only=True),
            skip_rejected=WorkflowOperationConfig(status_value="REJECTED"),
            start_development=WorkflowOperationConfig(status_value="IN PROGRESS"),
            open_for_review=WorkflowOperationConfig(status_value="IN REVIEW"),
            mark_complete=WorkflowOperationConfig(status_value="COMPLETE"),
            update_with_context=WorkflowOperationConfig(status_value=""),
        )
        # Defaults must be empty strings
        assert model.open_for_dev.status_value == ""
        assert model.in_planning.status_value == ""
        assert model.in_planning.human_only is True

    def test_int_wf_matches_open_for_dev(self) -> None:
        """INT-WF: matches() works case-insensitively for OPEN_FOR_DEV."""
        cfg = WorkflowConfig(self._make_full_config())
        assert cfg.matches(WorkflowOperation.OPEN_FOR_DEV, "OPEN") is True
        assert cfg.matches(WorkflowOperation.OPEN_FOR_DEV, "open") is True
        assert cfg.matches(WorkflowOperation.OPEN_FOR_DEV, "IN PROGRESS") is False

    def test_int_wf_matches_in_planning(self) -> None:
        """INT-WF: matches() works for IN_PLANNING."""
        cfg = WorkflowConfig(self._make_full_config())
        assert cfg.matches(WorkflowOperation.IN_PLANNING, "IN PLANNING") is True
        assert cfg.matches(WorkflowOperation.IN_PLANNING, "in planning") is True
        assert cfg.matches(WorkflowOperation.IN_PLANNING, "ACCEPTED") is False
