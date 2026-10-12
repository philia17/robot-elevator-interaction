#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from press.t5_observe_preflight import (
    LEGACY_T5_AUTHORITY_RETIRED,
    LEGACY_T5_UNRECOGNIZED_ARTIFACT,
    PreflightError,
    authorize_plan,
    build_report,
    check_static_gates,
    load_plan,
    main,
)

REPO = Path(__file__).resolve().parents[2]
PLAN = (
    REPO / "config" / "press" / "pilots" / "t5_visual_observe_auth_plan_20260805.json"
)
ARCHIVE = (
    REPO
    / "config"
    / "press"
    / "pilots"
    / "archive"
    / "t5_visual_observe_authorization_20260807.json"
)
CONFIRMATION = "AUTHORIZE T5 OBSERVE 80-100MM NO PRESS 20260805"


def _report_for(path: Path) -> dict:
    plan = load_plan(path)
    return build_report(plan, plan_path=path, ros_report=None)


def test_default_plan_is_fail_closed_and_has_no_historical_auth_fields():
    plan = load_plan(PLAN)
    assert plan["motion_authorized"] is False
    assert plan["user_approved"] is False
    assert plan["capture_enabled"] is False
    assert plan["execution_ready"] is False
    assert plan["press_authorized"] is False
    assert plan["contact_authorized"] is False
    assert plan["vendor_initialize_authorized"] is False
    assert plan["publish_tf"] is False
    assert "authorization_id" not in plan
    assert "authorized_at_utc" not in plan
    assert plan["plan_status"] == "awaiting_authorization"
    assert check_static_gates(plan) == []


def test_default_report_is_audit_only_and_fail_closed():
    report = _report_for(PLAN)
    assert report["report_semantics"] == "legacy_t5_audit_only"
    assert report["artifact_role"] == "inert_default_template"
    assert report["stored_motion_authorized"] is False
    assert report["stored_user_approved"] is False
    assert report["stored_capture_enabled"] is False
    assert report["stored_plan_id"] == "t5_visual_observe_20260805_r1"
    assert report["stored_plan_status"] == "awaiting_authorization"
    assert "stored_authorization_id" not in report
    assert "stored_authorized_at_utc" not in report
    assert report["effective_motion_authority"] is False
    assert report["authority_creation_supported"] is False
    assert report["execution_ready"] is False
    assert report["motion_commanded"] is False
    assert report["audit_passed"] is True
    assert report["passed"] is True
    assert "motion_authorized" not in report
    assert "authorization_id" not in report
    json.dumps(report)


def test_archive_report_keeps_historical_facts_but_effective_false():
    plan = load_plan(ARCHIVE)
    assert plan["motion_authorized"] is True
    assert plan["user_approved"] is True
    assert plan["capture_enabled"] is True
    assert plan["authorization_id"] == "t5_observe_20260807_philia"
    assert plan["authorized_at_utc"].startswith("2026-08-07")

    report = build_report(plan, plan_path=ARCHIVE, ros_report=None)
    assert report["artifact_role"] == "historical_legacy_authorization"
    assert report["stored_motion_authorized"] is True
    assert report["stored_user_approved"] is True
    assert report["stored_capture_enabled"] is True
    assert report["stored_authorization_id"] == "t5_observe_20260807_philia"
    assert report["stored_authorized_at_utc"] == plan["authorized_at_utc"]
    assert report["stored_plan_id"] == "t5_visual_observe_20260805_r1"
    assert report["stored_plan_status"] == "authorized_awaiting_operator_move"
    assert report["effective_motion_authority"] is False
    assert report["authority_creation_supported"] is False
    assert report["execution_ready"] is False
    assert report["motion_commanded"] is False
    assert report["audit_passed"] is True
    assert report["passed"] is True
    assert "motion_authorized" not in report
    assert "authorization_id" not in report


@pytest.mark.parametrize("source", [PLAN, ARCHIVE])
def test_authorize_plan_rejects_known_plans_and_leaves_input_unchanged(source):
    plan = load_plan(source)
    snapshot = copy.deepcopy(plan)
    with pytest.raises(PreflightError) as exc:
        authorize_plan(plan, CONFIRMATION, "t5_observe_new")
    assert LEGACY_T5_AUTHORITY_RETIRED in str(exc.value)
    assert plan == snapshot


def test_authorize_plan_rejects_arbitrary_input_without_inspecting_it():
    arbitrary = {
        "motion_authorized": True,
        "required_confirmation_text": CONFIRMATION,
        "authorization_id": "should_never_be_read",
    }
    snapshot = copy.deepcopy(arbitrary)
    with pytest.raises(PreflightError, match=LEGACY_T5_AUTHORITY_RETIRED):
        authorize_plan(arbitrary, CONFIRMATION, "arbitrary-id")
    assert arbitrary == snapshot

    sentinel = object()
    with pytest.raises(PreflightError, match=LEGACY_T5_AUTHORITY_RETIRED):
        authorize_plan(sentinel, "anything", "anything")


def test_confirm_rejects_before_any_disk_write(tmp_path: Path):
    out = tmp_path / "nested" / "report.json"
    before = PLAN.read_bytes()
    with pytest.raises(SystemExit, match=LEGACY_T5_AUTHORITY_RETIRED):
        main(["--confirm", CONFIRMATION, "--output", str(out)])
    assert not out.exists()
    assert not (tmp_path / "nested").exists()
    assert PLAN.read_bytes() == before


def test_confirm_with_write_plan_rejects_before_any_disk_write(tmp_path: Path):
    out = tmp_path / "nested" / "report.json"
    before = PLAN.read_bytes()
    with pytest.raises(SystemExit, match=LEGACY_T5_AUTHORITY_RETIRED):
        main(
            [
                "--confirm",
                CONFIRMATION,
                "--write-plan",
                "--output",
                str(out),
            ]
        )
    assert not out.exists()
    assert not (tmp_path / "nested").exists()
    assert PLAN.read_bytes() == before


def test_auth_id_alone_is_also_rejected(tmp_path: Path):
    out = tmp_path / "report.json"
    with pytest.raises(SystemExit, match=LEGACY_T5_AUTHORITY_RETIRED):
        main(["--auth-id", "t5_observe_new", "--output", str(out)])
    assert not out.exists()


def test_empty_confirm_flag_is_still_rejected(tmp_path: Path):
    out = tmp_path / "report.json"
    with pytest.raises(SystemExit, match=LEGACY_T5_AUTHORITY_RETIRED):
        main(["--confirm", "", "--output", str(out)])
    assert not out.exists()


def test_write_plan_rejects_missing_and_existing_targets_without_writes(
    tmp_path: Path,
):
    missing = tmp_path / "missing" / "plan.json"
    out = tmp_path / "out" / "report.json"
    with pytest.raises(SystemExit, match=LEGACY_T5_AUTHORITY_RETIRED):
        main(["--write-plan", "--plan", str(missing), "--output", str(out)])
    assert not missing.exists()
    assert not (tmp_path / "missing").exists()
    assert not out.exists()
    assert not (tmp_path / "out").exists()

    existing = tmp_path / "existing.json"
    existing.write_text("sentinel", encoding="utf-8")
    before = existing.read_bytes()
    with pytest.raises(SystemExit, match=LEGACY_T5_AUTHORITY_RETIRED):
        main(["--write-plan", "--plan", str(existing)])
    assert existing.read_bytes() == before


def test_unknown_explicit_plan_is_rejected_fail_closed(tmp_path: Path):
    copy_path = tmp_path / "copy.json"
    copy_path.write_bytes(PLAN.read_bytes())
    out = tmp_path / "out" / "report.json"
    with pytest.raises(SystemExit, match=LEGACY_T5_UNRECOGNIZED_ARTIFACT):
        main(["--plan", str(copy_path), "--output", str(out)])
    assert not out.exists()
    assert not (tmp_path / "out").exists()

    with pytest.raises(SystemExit, match=LEGACY_T5_UNRECOGNIZED_ARTIFACT):
        main(["--plan", str(tmp_path / "does_not_exist.json")])


def test_output_only_report_is_allowed_and_audit_only(tmp_path: Path):
    out = tmp_path / "reports" / "audit.json"
    assert main(["--output", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["report_semantics"] == "legacy_t5_audit_only"
    assert report["artifact_role"] == "inert_default_template"
    assert report["effective_motion_authority"] is False
    assert report["authority_creation_supported"] is False
    assert report["execution_ready"] is False
    assert report["motion_commanded"] is False
    assert "motion_authorized" not in report
    assert "authorization_id" not in report


def test_output_report_for_historical_archive_is_audit_only(tmp_path: Path):
    out = tmp_path / "archive_audit.json"
    assert main(["--plan", str(ARCHIVE), "--output", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["artifact_role"] == "historical_legacy_authorization"
    assert report["stored_motion_authorized"] is True
    assert report["stored_authorization_id"] == "t5_observe_20260807_philia"
    assert report["effective_motion_authority"] is False
    assert report["execution_ready"] is False
    assert report["motion_commanded"] is False


def test_artifact_hash_equals_exact_bytes_and_plan_sha256_is_alias(tmp_path: Path):
    plan_bytes = PLAN.read_bytes()
    copy_path = tmp_path / "copy.json"
    copy_path.write_bytes(plan_bytes)
    plan = load_plan(copy_path)
    report = build_report(plan, plan_path=copy_path, ros_report=None)
    expected = hashlib.sha256(plan_bytes).hexdigest()
    assert report["artifact_sha256"] == expected
    assert report["plan_sha256"] == expected
    assert report["plan_sha256"] == report["artifact_sha256"]
    assert report["artifact_role"] == "unrecognized_artifact"
    assert report["audit_passed"] is False
    assert report["effective_motion_authority"] is False
    assert report["stored_motion_authorized"] is False


def test_report_uses_exact_artifact_bytes_not_the_mutated_in_memory_dict(
    tmp_path: Path,
):
    copy_path = tmp_path / "copy.json"
    copy_path.write_bytes(PLAN.read_bytes())
    plan = load_plan(copy_path)
    plan["motion_authorized"] = True
    plan["authorization_id"] = "in_memory_only"
    report = build_report(plan, plan_path=copy_path, ros_report=None)
    assert report["stored_motion_authorized"] is False
    assert "stored_authorization_id" not in report
    assert report["effective_motion_authority"] is False


def test_passed_and_audit_passed_mean_audit_validity_not_authority():
    default_report = _report_for(PLAN)
    archive_report = _report_for(ARCHIVE)
    assert default_report["passed"] is default_report["audit_passed"] is True
    assert archive_report["passed"] is archive_report["audit_passed"] is True
    assert archive_report["stored_motion_authorized"] is True
    assert archive_report["effective_motion_authority"] is False
    assert archive_report["authority_creation_supported"] is False


def test_no_receipt_or_ticket_fields_and_execution_flags_stay_false():
    for path in (PLAN, ARCHIVE):
        report = _report_for(path)
        assert not any(
            "receipt" in key.lower() or "ticket" in key.lower() for key in report
        )
        assert report["execution_ready"] is False
        assert report["motion_commanded"] is False
        assert report["serial_used"] is False


def test_next_operator_step_never_tells_operator_to_authorize():
    for path in (PLAN, ARCHIVE):
        step = _report_for(path)["next_operator_step"].lower()
        assert "paste" not in step
        assert "to authorize" not in step
        assert "retired" in step


def test_static_gates_accept_archive_historical_motion_fact():
    assert check_static_gates(load_plan(PLAN)) == []
    assert check_static_gates(load_plan(ARCHIVE)) == []
