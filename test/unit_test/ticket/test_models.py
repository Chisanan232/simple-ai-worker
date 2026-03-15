"""
Unit tests for :mod:`src.ticket.models` (UNIT-TM-01 through UNIT-TM-12).

Covers:
- TicketRecord construction, field access, immutability (frozen=True)
- TicketRecord default values
- PRRecord construction, mutability, and field defaults
"""

from __future__ import annotations

import time

import pytest

from src.ticket.models import PRRecord, TicketRecord


# ===========================================================================
# TicketRecord
# ===========================================================================


class TestTicketRecordConstruction:
    def test_all_fields_stored(self) -> None:
        """UNIT-TM-01: All explicitly provided fields are stored correctly."""
        rec = TicketRecord(
            id="PROJ-1",
            source="jira",
            title="Implement login",
            url="https://jira.example.com/PROJ-1",
            raw_status="ACCEPTED",
        )
        assert rec.id == "PROJ-1"
        assert rec.source == "jira"
        assert rec.title == "Implement login"
        assert rec.url == "https://jira.example.com/PROJ-1"
        assert rec.raw_status == "ACCEPTED"

    def test_raw_status_defaults_to_empty_string(self) -> None:
        """UNIT-TM-02: raw_status defaults to '' when not supplied."""
        rec = TicketRecord(id="T-1", source="clickup", title="Task", url="")
        assert rec.raw_status == ""

    def test_url_may_be_empty_string(self) -> None:
        """UNIT-TM-03: url field accepts empty string (ticket URL is optional)."""
        rec = TicketRecord(id="T-2", source="jira", title="No URL", url="")
        assert rec.url == ""

    def test_clickup_source_stored(self) -> None:
        """UNIT-TM-04: source='clickup' is accepted and stored as-is."""
        rec = TicketRecord(id="cu-99", source="clickup", title="CU task", url="")
        assert rec.source == "clickup"


class TestTicketRecordImmutability:
    def test_frozen_rejects_attribute_assignment(self) -> None:
        """UNIT-TM-05: TicketRecord is frozen — assignment raises FrozenInstanceError."""
        rec = TicketRecord(id="P-1", source="jira", title="T", url="")
        with pytest.raises(Exception):  # FrozenInstanceError (dataclasses.FrozenInstanceError)
            rec.id = "P-2"  # type: ignore[misc]

    def test_two_identical_records_are_equal(self) -> None:
        """UNIT-TM-06: Two TicketRecords with identical fields compare equal."""
        r1 = TicketRecord(id="P-1", source="jira", title="T", url="u", raw_status="A")
        r2 = TicketRecord(id="P-1", source="jira", title="T", url="u", raw_status="A")
        assert r1 == r2

    def test_two_records_with_different_id_are_not_equal(self) -> None:
        """UNIT-TM-07: Records with different id values are not equal."""
        r1 = TicketRecord(id="P-1", source="jira", title="T", url="", raw_status="")
        r2 = TicketRecord(id="P-2", source="jira", title="T", url="", raw_status="")
        assert r1 != r2

    def test_frozen_record_is_hashable(self) -> None:
        """UNIT-TM-08: A frozen TicketRecord can be stored in a set / used as dict key."""
        rec = TicketRecord(id="P-1", source="jira", title="T", url="")
        s = {rec}
        assert rec in s


# ===========================================================================
# PRRecord
# ===========================================================================


class TestPRRecordConstruction:
    def test_all_required_fields_stored(self) -> None:
        """UNIT-TM-09: ticket_id, pr_url, and opened_at_utc are stored correctly."""
        ts = time.time()
        rec = PRRecord(ticket_id="PROJ-5", pr_url="https://github.com/r/pull/5", opened_at_utc=ts)
        assert rec.ticket_id == "PROJ-5"
        assert rec.pr_url == "https://github.com/r/pull/5"
        assert rec.opened_at_utc == ts

    def test_approval_count_defaults_to_zero(self) -> None:
        """UNIT-TM-10: approval_count defaults to 0 when not supplied."""
        rec = PRRecord(ticket_id="T-1", pr_url="https://github.com/r/pull/1", opened_at_utc=0.0)
        assert rec.approval_count == 0

    def test_is_merged_defaults_to_false(self) -> None:
        """UNIT-TM-11: is_merged defaults to False when not supplied."""
        rec = PRRecord(ticket_id="T-1", pr_url="https://github.com/r/pull/1", opened_at_utc=0.0)
        assert rec.is_merged is False

    def test_pr_record_is_mutable(self) -> None:
        """UNIT-TM-12: PRRecord is mutable — fields can be updated in-place."""
        rec = PRRecord(ticket_id="T-1", pr_url="https://github.com/r/pull/1", opened_at_utc=0.0)
        rec.approval_count = 2
        rec.is_merged = True
        assert rec.approval_count == 2
        assert rec.is_merged is True

    def test_explicit_approval_count_and_is_merged(self) -> None:
        """PRRecord accepts explicit approval_count and is_merged values."""
        rec = PRRecord(
            ticket_id="T-2",
            pr_url="https://github.com/r/pull/2",
            opened_at_utc=100.0,
            approval_count=3,
            is_merged=True,
        )
        assert rec.approval_count == 3
        assert rec.is_merged is True

