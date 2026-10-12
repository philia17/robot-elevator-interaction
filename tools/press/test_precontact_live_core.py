import pytest
from types import SimpleNamespace

from press.precontact_live_core import (
    BUTTON1_DIAGNOSTIC_SNAPSHOT_SCHEMA,
    DIAGNOSTIC_FALLBACK_DELAY_S,
    DIAGNOSTIC_TARGET_ARTIFACT_SCHEMA,
    DisplayCadenceLatch,
    DIAGNOSTIC_RAW_PAIR_MAX_AGE_S,
    DiagnosticSnapshot,
    EvidenceHeader,
    LivePhase,
    ProjectionEvidence,
    RAW_PAIR_DROP_S,
    SAFE_FEEDBACK_MAX_AGE_S,
    TorqueDiagnosticLatch,
    diagnostic_fallback_due,
    diagnostic_fallback_remaining_s,
    diagnostic_capture_from_raw_pair,
    evaluate_precontact_gate,
    freeze_button1,
    m0_is_fresh_ready,
    m0_summary,
    raw_button1_pair,
    target_projection_evidence,
    torque_diagnostic_state,
    wrist_center_evidence_state,
)


def _summary():
    return {
        "header": _header(100),
        "detection_id": "button_004",
        "label": "1",
        "ready": True,
        "depth_valid": True,
        "normal_valid": True,
        "camera_xyz": (0.03, -0.02, 0.52),
        "normal": (0.0, 0.0, -1.0),
        "bbox": (10, 20, 30, 40),
        "status": "READY",
        "confirmations": 5,
    }


def _header(stamp_ns: int, frame_id: str = "camera_optical_frame") -> EvidenceHeader:
    return EvidenceHeader(stamp_ns=stamp_ns, frame_id=frame_id)


def _raw_inputs():
    return (
        [{"detection_id": "button_004", "label": "1", "pressable": True}],
        [
            {
                "detection_id": "button_004",
                "depth_valid": True,
                "x_min": 10.0,
                "y_min": 20.0,
                "x_max": 30.0,
                "y_max": 40.0,
                "x_m": 0.03,
                "y_m": -0.02,
                "z_m": 0.52,
                "surface_geometry": {
                    "estimate_valid": True,
                    "panel_outward_normal_x": 0.0,
                    "panel_outward_normal_y": 0.0,
                    "panel_outward_normal_z": -1.0,
                },
            }
        ],
    )


def _m0_message(*, stamp_ns: int = 100, normal_valid: bool = True):
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000),
            frame_id="camera_optical_frame",
        ),
        effective_label="1",
        target_label="1",
        detection_id="button_004",
        perception_ready=True,
        confirmation_count=5,
        observation=SimpleNamespace(
            x_min=10.0,
            y_min=20.0,
            x_max=30.0,
            y_max=40.0,
            x_m=0.03,
            y_m=-0.02,
            z_m=0.52,
            depth_valid=True,
            surface_geometry=SimpleNamespace(
                estimate_valid=normal_valid,
                panel_outward_normal_x=0.0,
                panel_outward_normal_y=0.0,
                panel_outward_normal_z=-1.0,
            ),
        ),
        status="READY",
    )


def test_freeze_is_deterministic_and_non_executable():
    frozen = freeze_button1(_summary(), now=100.0, m0_receive_age_s=0.0)
    assert frozen.snapshot_id.startswith("button1-")
    assert len(frozen.snapshot_hash) == 64
    assert frozen.locked.label == "1"
    assert frozen.artifact_schema == DIAGNOSTIC_TARGET_ARTIFACT_SCHEMA
    assert frozen.target_id == "button1:camera_optical_frame:100:button_004"
    assert frozen.detection_id == "button_004"
    assert frozen.source_stamp_ns == 100
    assert frozen.source_frame_id == "camera_optical_frame"
    assert frozen.provenance_hash == frozen.snapshot_hash


def test_freeze_rejects_missing_diagnostic_target_provenance_without_relaxing_m0_gate():
    missing_detection = dict(_summary())
    missing_detection["detection_id"] = ""
    with pytest.raises(ValueError, match="detection_id"):
        freeze_button1(missing_detection, now=100.0, m0_receive_age_s=0.0)
    missing_header = dict(_summary())
    missing_header["header"] = _header(0, "")
    with pytest.raises(ValueError, match="provenance header"):
        freeze_button1(missing_header, now=100.0, m0_receive_age_s=0.0)


def test_m0_summary_uses_tuple_geometry_accepted_by_freeze():
    summary = m0_summary(_m0_message())
    assert isinstance(summary["camera_xyz"], tuple)
    assert isinstance(summary["normal"], tuple)
    assert summary["header"] == _header(100)
    frozen = freeze_button1(summary, now=100.0, m0_receive_age_s=0.0)
    assert frozen.locked.camera_xyz_m == pytest.approx((0.03, -0.02, 0.52))


@pytest.mark.parametrize("label", ["2", "open", "close", "up", "down", "b1"])
def test_selected_label_freezes_current_track_not_old_frame_detector_id(label):
    summary = {
        **_summary(), "label": label, "detection_id": "current-frame-id",
        "semantic_track_id": "chosen-track", "pressable": True,
    }
    frozen = freeze_button1(
        summary, now=100.0, m0_receive_age_s=.1, target_label=label,
        detection_id="old-frame-id", semantic_track_id="chosen-track",
    )
    assert frozen.locked.label == label
    assert frozen.detection_id == "current-frame-id"
    assert frozen.locked.camera_xyz_m == summary["camera_xyz"]
    with pytest.raises(ValueError, match="identity"):
        freeze_button1(
            summary, now=100.0, m0_receive_age_s=.1, target_label=label,
            semantic_track_id="another-track",
        )


def test_selected_raw_pair_joins_semantic_track_to_its_same_frame_observation():
    semantics, observations = _raw_inputs()
    semantics[0].update({"label": "open", "raw_response": '{"track_id":"chosen-track"}'})
    semantics.append({"label": "open", "detection_id": "neighbor", "track_id": "other-track", "pressable": True})
    observations.append({**observations[0], "detection_id": "neighbor", "x_m": .25})
    pair = raw_button1_pair(
        semantics, observations, semantics_header=_header(100), observations_header=_header(100),
        target_label="open", detection_id="from-earlier-frame", semantic_track_id="chosen-track",
    )
    assert pair["matched"]
    assert pair["label"] == "open"
    assert pair["semantic_track_id"] == "chosen-track"
    assert pair["detection_id"] == "button_004"
    assert pair["observation_3d"][0] == .03
    diagnostic = diagnostic_capture_from_raw_pair(
        pair, now=100.0, raw_pair_receive_age_s=.1, target_label="open",
        semantic_track_id="chosen-track",
    )
    assert diagnostic.locked.label == "open"
    assert diagnostic.diagnostic_only is True
    mismatched = raw_button1_pair(
        semantics, observations, semantics_header=_header(100), observations_header=_header(101),
        target_label="open", semantic_track_id="chosen-track",
    )
    assert mismatched["error"] == "HEADER_STAMP_MISMATCH"


@pytest.mark.parametrize("label,pressable", [("unknown", True), ("2", False)])
def test_selected_unknown_or_nonpressable_never_freezes(label, pressable):
    with pytest.raises(ValueError):
        freeze_button1(
            {**_summary(), "label": label, "pressable": pressable, "semantic_track_id": "track"},
            target_label=label, semantic_track_id="track", now=100., m0_receive_age_s=.1,
        )


def test_m0_summary_exposes_current_track_and_verified_pressable_state():
    message = _m0_message()
    message.semantic_track_id = "track-2"
    message.effective_label = "2"
    summary = m0_summary(message)
    assert summary["semantic_track_id"] == "track-2"
    assert summary["pressable"] is True
    message.perception_ready = False
    message.status = "INVALID_TARGET"
    assert m0_summary(message)["pressable"] is False


def test_diagnostic_fallback_delay_boundary_and_documentation():
    assert DIAGNOSTIC_FALLBACK_DELAY_S == 2.5
    assert diagnostic_fallback_due(None, now=0.0) is True
    assert diagnostic_fallback_due(100.0, now=102.49, delay_s=2.5) is False
    assert diagnostic_fallback_due(100.0, now=102.5, delay_s=2.5) is True
    assert diagnostic_fallback_remaining_s(100.0, now=101.0, delay_s=2.5) == pytest.approx(1.5)
    assert diagnostic_fallback_remaining_s(100.0, now=103.0, delay_s=2.5) == 0.0


def test_current_execution_gate_is_hard_disabled():
    frozen = freeze_button1(_summary(), now=100.0, m0_receive_age_s=0.0)
    result = evaluate_precontact_gate(
        frozen,
        execution_ready=False,
        action_server_ready=False,
        feedback_fresh=True,
        robot_at_standby=True,
        foreign_joint_target_publishers=0,
        now=100.5,
    )
    assert result.phase is LivePhase.DISABLED
    assert result.enabled is False
    assert result.reason == "execution_ready=false"


def test_execution_requires_fresh_frozen_target_and_single_owner():
    frozen = freeze_button1(_summary(), now=100.0, m0_receive_age_s=0.0)
    stale = evaluate_precontact_gate(
        frozen,
        execution_ready=True,
        action_server_ready=True,
        feedback_fresh=True,
        robot_at_standby=True,
        foreign_joint_target_publishers=0,
        now=102.1,
    )
    assert stale.enabled is False
    assert "expired" in stale.reason
    busy = evaluate_precontact_gate(
        frozen,
        execution_ready=True,
        action_server_ready=True,
        feedback_fresh=True,
        robot_at_standby=True,
        foreign_joint_target_publishers=1,
        now=100.5,
    )
    assert busy.enabled is False
    assert "publisher" in busy.reason


def test_raw_pair_rejects_cross_stamp_or_frame_before_detection_id_join():
    semantics, observations = _raw_inputs()
    cross_stamp = raw_button1_pair(
        semantics,
        observations,
        semantics_header=_header(100),
        observations_header=_header(101),
    )
    assert cross_stamp["matched"] is False
    assert cross_stamp["error"] == "HEADER_STAMP_MISMATCH"
    cross_frame = raw_button1_pair(
        semantics,
        observations,
        semantics_header=_header(100, "semantic_frame"),
        observations_header=_header(100, "camera_frame"),
    )
    assert cross_frame["matched"] is False
    assert cross_frame["error"] == "HEADER_FRAME_MISMATCH"
    paired = raw_button1_pair(
        semantics,
        observations,
        semantics_header=_header(100),
        observations_header=_header(100),
    )
    assert paired["matched"] is True
    assert paired["observation_3d"] == pytest.approx((0.03, -0.02, 0.52))
    assert paired["depth_valid"] is True
    assert paired["normal_valid"] is True


def _qualified_raw_pair(**observation_overrides):
    semantics, observations = _raw_inputs()
    if observation_overrides:
        observations[0].update(observation_overrides)
    return raw_button1_pair(
        semantics,
        observations,
        semantics_header=_header(100),
        observations_header=_header(100),
    )


def test_diagnostic_capture_seals_qualified_same_header_raw_pair():
    pair = _qualified_raw_pair()
    snap = diagnostic_capture_from_raw_pair(
        pair,
        now=100.0,
        raw_pair_receive_age_s=0.4,
        m0_status="STABILIZING",
    )
    assert isinstance(snap, DiagnosticSnapshot)
    assert snap.diagnostic_only is True
    assert snap.artifact_schema == BUTTON1_DIAGNOSTIC_SNAPSHOT_SCHEMA
    assert snap.snapshot_id.startswith("button1-diag-")
    assert len(snap.snapshot_hash) == 64
    assert snap.provenance_hash == snap.snapshot_hash
    assert snap.target_id == "button1:camera_optical_frame:100:button_004"
    assert snap.detection_id == "button_004"
    assert snap.source_stamp_ns == 100
    assert snap.source_frame_id == "camera_optical_frame"
    assert snap.locked.camera_xyz_m == pytest.approx((0.03, -0.02, 0.52))
    assert snap.locked.normal_camera == pytest.approx((0.0, 0.0, -1.0))
    assert snap.locked.bbox == pytest.approx((10.0, 20.0, 30.0, 40.0))
    assert snap.m0_status == "STABILIZING"
    # Deterministic hash: same evidence seals identically.
    again = diagnostic_capture_from_raw_pair(
        pair,
        now=100.0,
        raw_pair_receive_age_s=0.4,
        m0_status="STABILIZING",
    )
    assert again.snapshot_hash == snap.snapshot_hash


def test_diagnostic_capture_rejects_invalid_or_stale_evidence():
    semantics, observations = _raw_inputs()

    def _capture(pair, *, age: float = 0.4):
        return diagnostic_capture_from_raw_pair(
            pair,
            now=100.0,
            raw_pair_receive_age_s=age,
        )

    # Header stamp/frame mismatch is rejected before any identity join.
    for label_headers in (
        (_header(100), _header(101)),
        (_header(100, "semantic_frame"), _header(100, "camera_frame")),
    ):
        with pytest.raises(ValueError):
            _capture(
                raw_button1_pair(
                    semantics,
                    observations,
                    semantics_header=label_headers[0],
                    observations_header=label_headers[1],
                )
            )
    # Missing detection_id.
    no_id = ([{"detection_id": "", "label": "1"}], observations)
    with pytest.raises(ValueError, match="LABEL_1_NO_DETECTION_ID"):
        _capture(
            raw_button1_pair(
                no_id[0],
                no_id[1],
                semantics_header=_header(100),
                observations_header=_header(100),
            )
        )
    # Ambiguous label 1.
    ambiguous = (
        [
            {"detection_id": "button_004", "label": "1"},
            {"detection_id": "button_099", "label": "1"},
        ],
        observations,
    )
    with pytest.raises(ValueError, match="AMBIGUOUS_LABEL_1_2"):
        _capture(
            raw_button1_pair(
                ambiguous[0],
                ambiguous[1],
                semantics_header=_header(100),
                observations_header=_header(100),
            )
        )
    # Invalid depth, invalid normal estimate, nonfinite xyz, nonfinite bbox.
    with pytest.raises(ValueError, match="depth_valid"):
        _capture(_qualified_raw_pair(depth_valid=False))
    with pytest.raises(ValueError, match="estimate_valid"):
        _capture(_qualified_raw_pair(surface_geometry={
            "estimate_valid": False,
            "panel_outward_normal_x": 0.0,
            "panel_outward_normal_y": 0.0,
            "panel_outward_normal_z": -1.0,
        }))
    with pytest.raises(ValueError, match="camera_xyz"):
        _capture(_qualified_raw_pair(x_m=float("nan")))
    with pytest.raises(ValueError, match="INVALID_BBOX"):
        _capture(_qualified_raw_pair(x_max=float("nan")))
    with pytest.raises(ValueError, match="INVALID_BBOX"):
        _capture(_qualified_raw_pair(x_max=5.0))
    # Stale receive age never becomes the current pair.
    with pytest.raises(ValueError, match="stale"):
        _capture(_qualified_raw_pair(), age=RAW_PAIR_DROP_S + 0.01)
    with pytest.raises(ValueError, match="stale"):
        _capture(_qualified_raw_pair(), age=DIAGNOSTIC_RAW_PAIR_MAX_AGE_S + 0.01)
    with pytest.raises(ValueError, match="receive age"):
        _capture(_qualified_raw_pair(), age=None)
    # A non-matched pair is never accepted.
    with pytest.raises(ValueError, match="NO_LABEL_1"):
        diagnostic_capture_from_raw_pair(
            {"matched": False, "error": "NO_LABEL_1"},
            now=100.0,
            raw_pair_receive_age_s=0.1,
        )


def test_diagnostic_snapshot_never_passes_precontact_gate():
    snap = diagnostic_capture_from_raw_pair(
        _qualified_raw_pair(),
        now=100.0,
        raw_pair_receive_age_s=0.1,
        m0_status="STABILIZING",
    )
    result = evaluate_precontact_gate(
        snap,
        execution_ready=True,
        action_server_ready=True,
        feedback_fresh=True,
        robot_at_standby=True,
        foreign_joint_target_publishers=0,
        now=100.2,
    )
    assert result.phase is LivePhase.DISABLED
    assert result.enabled is False
    assert "NO MOTION TICKET" in result.reason


def test_display_cadence_gap_boundary_and_sticky_recovery():
    latch = DisplayCadenceLatch(gap_s=1.5, recovery_matched_pairs=3)
    assert latch.observe_matched_pair(_header(1), received_at=0.0).state == "DEGRADED"
    assert latch.observe_matched_pair(_header(2), received_at=0.0).state == "DEGRADED"
    assert latch.observe_matched_pair(_header(3), received_at=0.0).state == "VALID"
    assert latch.tick(now=1.49).state == "VALID"
    assert latch.tick(now=1.51).state == "DEGRADED"
    assert latch.observe_matched_pair(_header(4), received_at=1.52).state == "DEGRADED"
    assert latch.observe_matched_pair(_header(5), received_at=1.62).state == "DEGRADED"
    assert latch.observe_matched_pair(_header(6), received_at=1.72).state == "VALID"


def test_projection_hides_expired_pair_at_drop_boundary():
    semantics, observations = _raw_inputs()
    pair = raw_button1_pair(
        semantics,
        observations,
        semantics_header=_header(100),
        observations_header=_header(100),
    )
    cadence = DisplayCadenceLatch(recovery_matched_pairs=1)
    cadence.observe_matched_pair(_header(100), received_at=0.0)
    visible = target_projection_evidence(
        m0=None,
        m0_receive_age_s=None,
        raw_pair=pair,
        raw_pair_receive_age_s=11.99,
        cadence=cadence.snapshot(),
    )
    assert visible.source == "OBS_A3D_LAST_KNOWN"
    assert visible.camera_xyz_m == pytest.approx((0.03, -0.02, 0.52))
    hidden = target_projection_evidence(
        m0=None,
        m0_receive_age_s=None,
        raw_pair=pair,
        raw_pair_receive_age_s=12.01,
        cadence=cadence.snapshot(),
    )
    assert hidden.source == "NO_TARGET"
    assert hidden.camera_xyz_m is None


def test_m0_receive_age_fails_closed_for_freeze_and_projection():
    accepted = freeze_button1(_summary(), now=100.0, m0_receive_age_s=1.49)
    assert accepted.locked.label == "1"
    with pytest.raises(ValueError, match="not fresh READY"):
        freeze_button1(_summary(), now=100.0, m0_receive_age_s=1.51)
    fresh = target_projection_evidence(
        m0=_summary(),
        m0_receive_age_s=1.49,
        raw_pair=None,
        raw_pair_receive_age_s=None,
        cadence=None,
    )
    assert fresh.source == "M0_A3D"
    stale = target_projection_evidence(
        m0=_summary(),
        m0_receive_age_s=1.51,
        raw_pair=None,
        raw_pair_receive_age_s=None,
        cadence=None,
    )
    assert stale.source == "NO_TARGET"


def test_invalid_m0_normal_is_neither_fresh_ready_nor_freezable():
    invalid_normal = m0_summary(_m0_message(normal_valid=False))
    assert m0_is_fresh_ready(invalid_normal, receive_age_s=0.0) is False
    with pytest.raises(ValueError, match="not fresh READY"):
        freeze_button1(invalid_normal, now=100.0, m0_receive_age_s=0.0)
    projection = target_projection_evidence(
        m0=invalid_normal,
        m0_receive_age_s=0.0,
        raw_pair=None,
        raw_pair_receive_age_s=None,
        cadence=None,
    )
    assert projection.source == "NO_TARGET"


def test_fresh_m0_carries_a2d_only_for_fresh_valid_same_source_raw_pair():
    semantics, observations = _raw_inputs()
    pair = raw_button1_pair(
        semantics,
        observations,
        semantics_header=_header(100),
        observations_header=_header(100),
    )
    cadence = DisplayCadenceLatch(recovery_matched_pairs=1)
    cadence.observe_matched_pair(_header(100), received_at=0.0)
    same_source = target_projection_evidence(
        m0=m0_summary(_m0_message(stamp_ns=100)),
        m0_receive_age_s=0.0,
        raw_pair=pair,
        raw_pair_receive_age_s=0.1,
        cadence=cadence.snapshot(),
    )
    assert same_source.source == "M0_A3D"
    assert same_source.a2d == pytest.approx((20.0, 30.0))
    stale_raw = target_projection_evidence(
        m0=m0_summary(_m0_message(stamp_ns=100)),
        m0_receive_age_s=0.0,
        raw_pair=pair,
        raw_pair_receive_age_s=1.51,
        cadence=cadence.snapshot(),
    )
    assert stale_raw.source == "M0_A3D"
    assert stale_raw.a2d is None
    cross_source = target_projection_evidence(
        m0=m0_summary(_m0_message(stamp_ns=101)),
        m0_receive_age_s=0.0,
        raw_pair=pair,
        raw_pair_receive_age_s=0.1,
        cadence=cadence.snapshot(),
    )
    assert cross_source.source == "M0_A3D"
    assert cross_source.a2d is None


def test_projection_evidence_carries_source_frame_and_calibration_provenance():
    summary = _summary()
    summary["source_frame_id"] = "gemini335_color_optical_frame"
    summary["section_b_calibration_id"] = "calibration-id-a"
    summary["provenance_complete"] = True
    summary["raw_to_canonical_alias_status"] = "software_identity_verified_runtime_bound_physical_axes_pending"
    evidence = target_projection_evidence(
        m0=summary,
        m0_receive_age_s=0.0,
        raw_pair=None,
        raw_pair_receive_age_s=None,
        cadence=None,
    )
    assert evidence.source == "M0_A3D"
    assert evidence.source_frame_id == "gemini335_color_optical_frame"
    assert evidence.source_stamp_ns == 100
    assert evidence.calibration_id == "calibration-id-a"
    assert evidence.provenance_complete is True
    assert "pending" in evidence.raw_to_canonical_alias_status


def test_raw_pair_and_projection_preserve_observation_provenance():
    semantics, observations = _raw_inputs()
    observations[0]["surface_geometry"]["source_frame_id"] = "gemini335_color_optical_frame"
    observations[0]["surface_geometry"]["canonical_frame_id"] = "camera_optical_frame"
    observations[0]["surface_geometry"]["raw_to_canonical_alias_id"] = (
        "gemini335_color_optical_frame_to_camera_optical_frame_identity_v1"
    )
    observations[0]["surface_geometry"]["raw_to_canonical_alias_status"] = (
        "software_identity_verified_runtime_bound_physical_axes_pending"
    )
    observations[0]["surface_geometry"]["section_b_calibration_id"] = "calibration-id-a"
    observations[0]["surface_geometry"]["provenance_complete"] = True
    pair = raw_button1_pair(
        semantics,
        observations,
        semantics_header=_header(100),
        observations_header=_header(100),
    )
    assert pair["source_frame_id"] == "gemini335_color_optical_frame"
    assert pair["canonical_frame_id"] == "camera_optical_frame"
    assert pair["section_b_calibration_id"] == "calibration-id-a"
    assert pair["provenance_complete"] is True
    cadence = DisplayCadenceLatch(recovery_matched_pairs=1)
    cadence.observe_matched_pair(_header(100), received_at=0.0)
    evidence = target_projection_evidence(
        m0=None,
        m0_receive_age_s=None,
        raw_pair=pair,
        raw_pair_receive_age_s=0.1,
        cadence=cadence.snapshot(),
    )
    assert evidence.source == "OBS_A3D"
    assert evidence.source_frame_id == "gemini335_color_optical_frame"
    assert evidence.calibration_id == "calibration-id-a"
    assert evidence.provenance_complete is True


def test_wrist_center_gate_requires_fresh_canonical_contract_and_matching_provenance():
    summary = _summary()
    summary["source_frame_id"] = "camera_optical_frame"
    summary["section_b_calibration_id"] = "calibration-id-a"
    summary["provenance_complete"] = True
    summary["raw_to_canonical_alias_status"] = ""
    evidence = target_projection_evidence(
        m0=summary,
        m0_receive_age_s=0.0,
        raw_pair=None,
        raw_pair_receive_age_s=None,
        cadence=None,
    )
    blocked, reason = wrist_center_evidence_state(
        evidence,
        expected_calibration_id="calibration-id-a",
        wrist_calibration_contract_ok=False,
        bundle_contract_ok=True,
    )
    assert blocked is False
    assert reason == "NO_CENTER_CONTRACT"

    ok, reason = wrist_center_evidence_state(
        evidence,
        expected_calibration_id="calibration-id-a",
        wrist_calibration_contract_ok=True,
        bundle_contract_ok=True,
    )
    assert ok is True
    assert reason == "OK"


def test_wrist_center_gate_rejects_raw_frame_last_known_and_calibration_mismatch():
    summary = _summary()
    summary["source_frame_id"] = "gemini335_color_optical_frame"
    summary["section_b_calibration_id"] = "calibration-id-a"
    summary["provenance_complete"] = True
    evidence = target_projection_evidence(
        m0=summary,
        m0_receive_age_s=0.0,
        raw_pair=None,
        raw_pair_receive_age_s=None,
        cadence=None,
    )
    ok, reason = wrist_center_evidence_state(
        evidence,
        expected_calibration_id="calibration-id-a",
        wrist_calibration_contract_ok=True,
        bundle_contract_ok=True,
    )
    assert ok is False
    assert reason == "RAW_FRAME_NO_CENTER_CONTRACT"

    last_known = ProjectionEvidence(
        camera_xyz_m=(0.03, -0.02, 0.52),
        normal_camera=(0.0, 0.0, -1.0),
        source="OBS_A3D_LAST_KNOWN",
        a2d=(20.0, 30.0),
        source_frame_id="camera_optical_frame",
        calibration_id="calibration-id-a",
        provenance_complete=True,
    )
    ok, reason = wrist_center_evidence_state(
        last_known,
        expected_calibration_id="calibration-id-a",
        wrist_calibration_contract_ok=True,
        bundle_contract_ok=True,
    )
    assert ok is False
    assert reason == "LAST_KNOWN_ONLY"

    mismatch = ProjectionEvidence(
        camera_xyz_m=(0.03, -0.02, 0.52),
        normal_camera=(0.0, 0.0, -1.0),
        source="OBS_A3D",
        a2d=(20.0, 30.0),
        source_frame_id="camera_optical_frame",
        calibration_id="calibration-id-b",
        provenance_complete=True,
    )
    ok, reason = wrist_center_evidence_state(
        mismatch,
        expected_calibration_id="calibration-id-a",
        wrist_calibration_contract_ok=True,
        bundle_contract_ok=True,
    )
    assert ok is False
    assert reason == "CALIBRATION_ID_MISMATCH"


def test_torque_diagnostic_state_healthy_and_reports_off_flags():
    now = 1_000.0
    healthy = {"torque_enabled": [True] * 6, "fault": ""}
    state, joints, detail = torque_diagnostic_state(healthy, now - 0.5, now)
    assert state == "OK"
    assert joints == []
    assert "6/6" in detail

    off = {"torque_enabled": [True, True, False, True, True, False], "fault": ""}
    state, joints, detail = torque_diagnostic_state(off, now - 0.5, now)
    assert state == "TORQUE_FLAGS_OFF"
    assert joints == ["E", "G"]
    assert "not a recognition failure" not in detail  # label lives in the latch snapshot

    unknown = {"fault": ""}
    state, _joints, _detail = torque_diagnostic_state(unknown, now - 0.5, now)
    assert state == "TORQUE_FLAGS_UNKNOWN"

    assert torque_diagnostic_state(None, None, now)[0] == "UNKNOWN"
    assert torque_diagnostic_state({"fault": ""}, None, now)[0] == "UNKNOWN"


def test_torque_diagnostic_state_stale_and_faulted_are_diagnostic_only():
    now = 1_000.0
    sample = {"torque_enabled": [True] * 6, "fault": ""}
    state, _joints, detail = torque_diagnostic_state(sample, now - SAFE_FEEDBACK_MAX_AGE_S - 1.0, now)
    assert state == "STALE"
    assert "stale" in detail

    faulted = {"torque_enabled": [True] * 6, "fault": "one or more joint torque switches are off"}
    state, _joints, detail = torque_diagnostic_state(faulted, now, now)
    assert state == "FAULTED"
    assert "torque switches" in detail


def test_torque_diagnostic_latch_never_autoclears_a_problem():
    now = 1_000.0
    latch = TorqueDiagnosticLatch()
    snap = latch.observe({"torque_enabled": [True] * 6, "fault": ""}, now - 0.5, now)
    assert snap["state"] == "OK"
    assert snap["diagnostic"] is False
    assert snap["latched"] is False

    snap = latch.observe(
        {"torque_enabled": [True, False, True, True, True, True], "fault": ""},
        now,
        now,
    )
    assert snap["state"] == "TORQUE_FLAGS_OFF"
    assert snap["off_joints"] == ["S"]
    assert snap["latched"] is True
    assert snap["diagnostic"] is True
    assert "not a recognition failure" in snap["label"]

    # A later healthy sample never silently clears the latched diagnostic.
    snap = latch.observe({"torque_enabled": [True] * 6, "fault": ""}, now + 1.0, now + 1.0)
    assert snap["state"] == "TORQUE_FLAGS_OFF"
    assert snap["latched"] is True
    assert "LATCHED" in snap["detail"]

    # A stale sample latches too and never auto-recovers.
    latch2 = TorqueDiagnosticLatch()
    snap = latch2.observe({"torque_enabled": [True] * 6, "fault": ""}, now - 100.0, now)
    assert snap["state"] == "STALE"
    assert snap["latched"] is True
    snap = latch2.observe({"torque_enabled": [True] * 6, "fault": ""}, now + 1.0, now + 1.0)
    assert snap["state"] == "STALE"
    assert snap["latched"] is True
