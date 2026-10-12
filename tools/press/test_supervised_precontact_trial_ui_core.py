from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from press.supervised_precontact_trial_ui_core import (
    SnapshotError,
    accept_prepare,
    consume_move,
    read_canonical_snapshot,
)


def test_snapshot_binds_exact_canonical_bytes(tmp_path):
    content = json.dumps({"a": 1, "z": [2]}, sort_keys=True, separators=(",", ":"))
    path = tmp_path / "frozen.json"
    path.write_text(content, encoding="utf-8")
    snapshot = read_canonical_snapshot(str(path))
    assert snapshot.text == content
    assert snapshot.sha256 == hashlib.sha256(content.encode()).hexdigest()


def test_snapshot_never_reformats_evidence(tmp_path):
    path = tmp_path / "frozen.json"
    path.write_text('{"z": [2], "a": 1}\n', encoding="utf-8")
    with pytest.raises(SnapshotError, match="not canonical"):
        read_canonical_snapshot(str(path))


def test_prepare_ticket_is_bounded_and_move_requires_it():
    ticket = accept_prepare(accepted=True, authorization_id="ticket", ttl_ms=2000, planned_target_sha256="a" * 64, route_sha256="b" * 64, now=10.0)
    assert ticket is not None and ticket.remaining_ms(10.0) == 2000
    assert consume_move(ticket, 11.99) is ticket
    assert consume_move(ticket, 12.0) is None


def test_prepare_never_accepts_long_or_incomplete_ticket():
    assert accept_prepare(accepted=True, authorization_id="x", ttl_ms=2001, planned_target_sha256="a", route_sha256="b", now=0.0) is None
    assert accept_prepare(accepted=True, authorization_id="", ttl_ms=1, planned_target_sha256="a", route_sha256="b", now=0.0) is None


def test_panel_has_only_explicit_prepare_and_move_controls():
    source = (Path(__file__).parents[1] / "supervised_precontact_trial_panel.py").read_text(encoding="utf-8")
    assert '"PREPARE (NO MOTION)"' in source
    assert '"MOVE TO PRE-CONTACT"' in source
    assert 'confirmation_text = "MOVE_TO_PRECONTACT_ONLY"' in source
    assert "self.ticket = None  # consumed/expired before any goal can be sent" in source
    for forbidden in ("Torque", "standby", "teleop", "press", "return", "retry"):
        # The prohibition sentence may name it; no matching control label may.
        assert f'"{forbidden.title()}' not in source
