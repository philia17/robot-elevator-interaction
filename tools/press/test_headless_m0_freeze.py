"""Focused tests for the headless M0 freeze utility (formal + diagnostic).

The utility keeps its production ROS imports (rclpy / elevator_interfaces),
so this file installs minimal import-time stubs before loading it.  All logic
under test is the same code path the AGX tool executes.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import types
from types import SimpleNamespace

import pytest


_INJECTED: list = []


class _FakeNode:
    def __init__(self, *args, **kwargs):
        pass

    def create_subscription(self, *args, **kwargs):
        pass

    def destroy_node(self):
        pass


def _install_ros_stubs():
    if "rclpy" in sys.modules:
        return
    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda *args, **kwargs: None
    rclpy.shutdown = lambda *args, **kwargs: None
    rclpy.ok = lambda: True

    def spin_once(node, timeout_sec=0.2):
        while _INJECTED:
            message = _INJECTED.pop(0)
            if hasattr(message, "semantics") and not hasattr(
                message, "observations"
            ):
                node._on_semantics(message)
            elif hasattr(message, "observations") and not hasattr(
                message, "semantics"
            ):
                node._on_observations(message)
            else:
                node._on_target(message)
            break

    rclpy.spin_once = spin_once

    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_node.Node = _FakeNode
    rclpy_qos = types.ModuleType("rclpy.qos")
    rclpy_qos.QoSProfile = lambda **kwargs: SimpleNamespace(depth=10)
    rclpy_qos.ReliabilityPolicy = SimpleNamespace(RELIABLE="RELIABLE")

    elevator = types.ModuleType("elevator_interfaces")
    elevator_msg = types.ModuleType("elevator_interfaces.msg")
    elevator_msg.M0Target3D = object
    elevator_msg.ButtonObservation3DArray = object
    elevator_msg.ButtonSemanticArray = object
    elevator.msg = elevator_msg

    sys.modules["rclpy"] = rclpy
    sys.modules["rclpy.node"] = rclpy_node
    sys.modules["rclpy.qos"] = rclpy_qos
    sys.modules["elevator_interfaces"] = elevator
    sys.modules["elevator_interfaces.msg"] = elevator_msg


_install_ros_stubs()

from press.headless_m0_freeze import (  # noqa: E402
    FAIL_HEADER_MISMATCH,
    FAIL_INVALID_NORMAL,
    FAIL_NO_M0_READY,
    FAIL_OUTPUT_EXISTS,
    FAIL_STALE_TARGET,
    FAIL_TARGET_IDENTITY_MISMATCH,
    HeadlessFreezeNode,
    _evidence_header,
    _observation_dict,
    _semantic_dict,
    _ready,
    canonical_bytes,
    diagnostic_pair_failure,
    diagnostic_target_dict,
    main,
    parse_args,
    reject_reason,
    target_dict,
    with_canonical_hash,
)


def make_message(**overrides):
    msg = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=1_786_452_000, nanosec=0),
            frame_id="camera_optical_frame",
        ),
        status="READY",
        failure_reason="",
        perception_ready=True,
        target_label="1",
        effective_label="1",
        matching_label_count=1,
        detection_id="det_001",
        semantic_track_id="track_001",
        semantic_status="FRESH",
        semantic_age_sec=0.05,
        confirmation_count=5,
        required_confirmations=5,
        position_spread_m=0.002,
        motion_authorized=False,
    )
    msg.observation = SimpleNamespace(
        x_min=100.0,
        y_min=200.0,
        x_max=140.0,
        y_max=240.0,
        x_m=0.1,
        y_m=0.0,
        z_m=0.5,
        detector_confidence=0.93,
        depth_valid_ratio=0.98,
        surface_geometry=SimpleNamespace(
            estimate_valid=True,
            provenance_complete=True,
            failure_reason="",
            surface_point_x_m=0.1,
            surface_point_y_m=0.0,
            surface_point_z_m=0.5,
            panel_outward_normal_x=0.0,
            panel_outward_normal_y=0.0,
            panel_outward_normal_z=-1.0,
            normal_angular_uncertainty_rad=0.05,
            plane_rms_residual_m=0.001,
            plane_sample_count=80,
            geometry_method_id="ransac",
            section_b_calibration_id="cal_20260803",
            source_frame_id="camera_optical_frame",
            canonical_frame_id="camera_optical_frame",
        ),
    )
    observation_fields = {
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "x_m",
        "y_m",
        "z_m",
        "detector_confidence",
        "depth_valid_ratio",
    }
    for key, value in overrides.items():
        if key in observation_fields:
            setattr(msg.observation, key, value)
        else:
            setattr(msg, key, value)
    return msg


def surface(msg, **overrides):
    for key, value in overrides.items():
        setattr(msg.observation.surface_geometry, key, value)
    return msg


def observation(msg, **overrides):
    for key, value in overrides.items():
        setattr(msg.observation, key, value)
    return msg


def target(msg):
    return target_dict(msg)


STAMP_NS = 1_786_452_000_000_000_000


def make_observations_message(**overrides):
    stamp_ns = overrides.pop("stamp_ns", STAMP_NS)
    frame_id = overrides.pop("frame_id", "camera_optical_frame")
    detection_id = overrides.pop("detection_id", "det_001")
    obs = SimpleNamespace(
        detection_id=detection_id,
        x_min=100.0,
        y_min=200.0,
        x_max=140.0,
        y_max=240.0,
        detector_confidence=0.93,
        depth_valid=True,
        depth_valid_ratio=0.98,
        x_m=0.1,
        y_m=0.0,
        z_m=0.5,
        surface_geometry=SimpleNamespace(
            estimate_valid=True,
            provenance_complete=True,
            failure_reason="",
            surface_point_x_m=0.1,
            surface_point_y_m=0.0,
            surface_point_z_m=0.5,
            panel_outward_normal_x=0.0,
            panel_outward_normal_y=0.0,
            panel_outward_normal_z=-1.0,
            normal_angular_uncertainty_rad=0.05,
            plane_rms_residual_m=0.001,
            plane_sample_count=80,
            geometry_method_id="ransac",
            section_b_calibration_id="cal_20260803",
            source_frame_id=frame_id,
            canonical_frame_id=frame_id,
        ),
    )
    if "surface_geometry" in overrides:
        for key, value in overrides.pop("surface_geometry").items():
            setattr(obs.surface_geometry, key, value)
    for key, value in overrides.items():
        setattr(obs, key, value)
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(
                sec=stamp_ns // 1_000_000_000,
                nanosec=stamp_ns % 1_000_000_000,
            ),
            frame_id=frame_id,
        ),
        observations=[obs],
    )


def make_semantics_message(**overrides):
    stamp_ns = overrides.pop("stamp_ns", STAMP_NS)
    frame_id = overrides.pop("frame_id", "camera_optical_frame")
    detection_id = overrides.pop("detection_id", "det_001")
    label = overrides.pop("label", "1")
    metadata = {
        "track_id": "track_001",
        "semantic_status": "FRESH",
        "semantic_age_sec": 0.05,
        "association_valid": True,
    }
    metadata.update(overrides.pop("metadata", {}))
    for key in ("track_id", "semantic_status", "semantic_age_sec"):
        if key in overrides:
            metadata[key] = overrides.pop(key)
    semantic = SimpleNamespace(
        detection_id=detection_id,
        label=label,
        pressable=True,
        raw_response=json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    for key, value in overrides.items():
        setattr(semantic, key, value)
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(
                sec=stamp_ns // 1_000_000_000,
                nanosec=stamp_ns % 1_000_000_000,
            ),
            frame_id=frame_id,
        ),
        semantics=[semantic],
    )


def raw_pair(observations_msg, semantics_msg):
    from press.precontact_live_core import raw_button1_pair

    semantic_items = [
        _semantic_dict(item) for item in semantics_msg.semantics
    ]
    pair = raw_button1_pair(
        semantic_items,
        [_observation_dict(item) for item in observations_msg.observations],
        semantics_header=_evidence_header(semantics_msg),
        observations_header=_evidence_header(observations_msg),
    )
    if semantic_items:
        pair["semantic_age_sec"] = semantic_items[0].get(
            "semantic_age_sec", math.nan
        )
        pair["association_valid"] = semantic_items[0].get(
            "association_valid", False
        )
        pair["pressable"] = semantic_items[0].get("pressable", False)
    return pair


@pytest.fixture(autouse=True)
def _reset_injected():
    _INJECTED.clear()
    yield
    _INJECTED.clear()


# --- CLI surface ----------------------------------------------------------


def test_parse_args_preserves_formal_defaults():
    args = parse_args([])
    assert args.stable_frames == 5
    assert args.single_frame_diagnostic is False


def test_parse_args_exposes_diagnostic_flag():
    args = parse_args(["--single-frame-diagnostic", "--stable-frames", "3"])
    assert args.single_frame_diagnostic is True
    assert args.stable_frames == 3


# --- Formal 5-frame behavior ----------------------------------------------


def test_formal_ready_gate_still_strict():
    assert _ready(target(make_message()), "1") is True
    for field, value in (
        ("status", "STABILIZING"),
        ("perception_ready", False),
        ("target_label", "2"),
        ("effective_label", "other"),
        ("motion_authorized", True),
    ):
        assert _ready(target(make_message(**{field: value})), "1") is False, field
    zero_normal = surface(
        make_message(),
        panel_outward_normal_x=0.0,
        panel_outward_normal_y=0.0,
        panel_outward_normal_z=0.0,
    )
    assert _ready(target(zero_normal), "1") is False


def test_formal_node_requires_consecutive_ready_frames():
    node = HeadlessFreezeNode("/topic", "1")
    ready = make_message()
    for _ in range(4):
        node._on_target(ready)
    assert node.ready_count == 4
    node._on_target(make_message(status="STABILIZING"))
    assert node.ready_count == 0
    for _ in range(5):
        node._on_target(ready)
    assert node.ready_count == 5
    assert node.last_reject_code is None


def test_formal_main_freezes_after_five_readies(tmp_path):
    out = tmp_path / "formal.json"
    for _ in range(5):
        _INJECTED.append(make_message())
    rc = main(["--output", str(out), "--timeout-sec", "2.0"])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["schema"] == "m0_demo_freeze_v1"
    assert payload["mode"] == "headless_one_shot"
    assert payload["motion_authorized"] is False
    assert "execution_ready" not in payload


def test_formal_main_timeout_insufficient_frames(tmp_path, capsys):
    out = tmp_path / "formal.json"
    for _ in range(4):
        _INJECTED.append(make_message())
    rc = main(["--output", str(out), "--timeout-sec", "0.2"])
    assert rc == 2
    assert not out.exists()
    captured = capsys.readouterr().out
    assert "freeze_failure=NO_M0_READY" in captured
    assert "insufficient_stable_frames=4/5" in captured


# --- Diagnostic one-shot gate ---------------------------------------------


def _pair_failure(
    pair,
    *,
    now=1.0,
    received_at=0.5,
    depth_valid_ratio=0.98,
    source_frame_id="camera_optical_frame",
    canonical_frame_id="camera_optical_frame",
):
    return diagnostic_pair_failure(
        pair,
        now=now,
        received_at=received_at,
        depth_valid_ratio=depth_valid_ratio,
        source_frame_id=source_frame_id,
        canonical_frame_id=canonical_frame_id,
    )


def test_diagnostic_node_accepts_raw_pair_with_cached_semantic_without_m0_ready():
    node = HeadlessFreezeNode("/topic", "1", diagnostic=True)
    node._on_observations(make_observations_message())
    node._on_semantics(make_semantics_message(semantic_status="CACHED"))
    assert node.diagnostic_accepted is not None
    assert node.diagnostic_accepted["detection_id"] == "det_001"
    assert node.diagnostic_accepted["semantic_status"] == "CACHED"
    assert node.diagnostic_accepted["confirmation_count"] == 1
    assert node.diagnostic_accepted["motion_authorized"] is False
    assert node.last_reject_code is None


def test_diagnostic_pair_accepts_current_cached_semantic_pair():
    pair = raw_pair(
        make_observations_message(),
        make_semantics_message(
            semantic_status="CACHED",
            semantic_age_sec=1.5,
        ),
    )
    assert _pair_failure(pair) is None


@pytest.mark.parametrize(
    ("obs_overrides", "sem_overrides", "expected_code"),
    [
        (
            {"stamp_ns": STAMP_NS + 1},
            {},
            FAIL_HEADER_MISMATCH,
        ),
        (
            {"frame_id": "camera_frame"},
            {"frame_id": "semantic_frame"},
            FAIL_HEADER_MISMATCH,
        ),
        (
            {},
            {"label": "2"},
            FAIL_TARGET_IDENTITY_MISMATCH,
        ),
        (
            {},
            {"detection_id": ""},
            FAIL_TARGET_IDENTITY_MISMATCH,
        ),
        (
            {"depth_valid": False},
            {},
            FAIL_NO_M0_READY,
        ),
        (
            {"surface_geometry": {"estimate_valid": False}},
            {},
            FAIL_INVALID_NORMAL,
        ),
        (
            {
                "surface_geometry": {
                    "panel_outward_normal_x": 0.0,
                    "panel_outward_normal_y": 0.0,
                    "panel_outward_normal_z": 0.0,
                }
            },
            {},
            FAIL_INVALID_NORMAL,
        ),
        (
            {"x_m": float("nan")},
            {},
            FAIL_NO_M0_READY,
        ),
        (
            {"x_min": 150.0},
            {},
            FAIL_NO_M0_READY,
        ),
    ],
)
def test_diagnostic_pair_rejections(
    obs_overrides,
    sem_overrides,
    expected_code,
):
    pair = raw_pair(
        make_observations_message(**obs_overrides),
        make_semantics_message(**sem_overrides),
    )
    failure = _pair_failure(pair)
    assert failure is not None
    assert failure[0] == expected_code


def test_diagnostic_pair_rejects_stale_age_low_depth_ratio_and_empty_frames():
    pair = raw_pair(make_observations_message(), make_semantics_message())
    assert _pair_failure(pair, now=3.0, received_at=0.0)[0] == FAIL_STALE_TARGET
    zero_stamp = raw_pair(
        make_observations_message(stamp_ns=0),
        make_semantics_message(stamp_ns=0),
    )
    assert _pair_failure(zero_stamp)[0] == FAIL_STALE_TARGET
    assert (
        _pair_failure(pair, depth_valid_ratio=0.2)[0] == FAIL_NO_M0_READY
    )
    assert (
        _pair_failure(
            pair,
            source_frame_id="",
            canonical_frame_id="",
        )[0]
        == FAIL_STALE_TARGET
    )
    stale_semantic = raw_pair(
        make_observations_message(),
        make_semantics_message(semantic_age_sec=3.0),
    )
    assert _pair_failure(stale_semantic)[0] == FAIL_STALE_TARGET
    unassociated = raw_pair(
        make_observations_message(),
        make_semantics_message(metadata={"association_valid": False}),
    )
    assert _pair_failure(unassociated)[0] == FAIL_TARGET_IDENTITY_MISMATCH
    not_pressable = raw_pair(
        make_observations_message(),
        make_semantics_message(pressable=False),
    )
    assert _pair_failure(not_pressable)[0] == FAIL_TARGET_IDENTITY_MISMATCH


def test_diagnostic_pair_rejects_ambiguous_label_one():
    semantics_msg = make_semantics_message()
    semantics_msg.semantics.append(
        SimpleNamespace(
            detection_id="det_002",
            label="1",
            raw_response="{}",
        )
    )
    pair = raw_pair(make_observations_message(), semantics_msg)
    assert _pair_failure(pair)[0] == FAIL_TARGET_IDENTITY_MISMATCH


def test_formal_reject_reason_uses_only_formal_gates():
    assert reject_reason(target(make_message()), "1", diagnostic=False) is None
    # Formal mode does not gate on semantic staleness.
    stale = target(make_message(semantic_status="CACHED"))
    assert reject_reason(stale, "1", diagnostic=False) is None
    wrong = target(make_message(target_label="2"))
    assert (
        reject_reason(wrong, "1", diagnostic=False)[0]
        == FAIL_TARGET_IDENTITY_MISMATCH
    )
    msg = surface(make_message(), estimate_valid=False)
    assert (
        reject_reason(target(msg), "1", diagnostic=False)[0] == FAIL_NO_M0_READY
    )
    assert (
        reject_reason(
            target(msg), "1", diagnostic=False, allow_invalid_surface=True
        )
        is None
    )


# --- Diagnostic main-loop end to end --------------------------------------


def test_diagnostic_main_one_shot_success(tmp_path):
    out = tmp_path / "diag.json"
    _INJECTED.append(make_message(status="SEMANTIC_SET_INCOMPLETE"))
    _INJECTED.append(make_observations_message())
    _INJECTED.append(make_semantics_message(semantic_status="CACHED"))
    rc = main(
        [
            "--single-frame-diagnostic",
            "--output",
            str(out),
            "--timeout-sec",
            "2.0",
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["schema"] == "m0_single_frame_diagnostic_v1"
    assert payload["mode"] == "single_frame_diagnostic"
    assert payload["diagnostic_only"] is True
    assert payload["no_motion"] is True
    assert payload["execution_ready"] is False
    assert payload["motion_authorized"] is False
    assert payload["source_stamp_ns"] == STAMP_NS
    assert payload["source_frame_id"] == "camera_optical_frame"
    assert payload["detection_id"] == "det_001"
    assert payload["semantic_track_id"] == "track_001"
    assert payload["semantic_status"] == "CACHED"
    assert payload["m0_status"] == "SEMANTIC_SET_INCOMPLETE"
    assert payload["frozen_target"]["perception_ready"] is False
    assert payload["frozen_target"]["surface_geometry"]["provenance_complete"] is True
    assert payload["canonical_payload_sha256"]
    for forbidden in (
        "route",
        "trajectory",
        "command",
        "joint_target",
        "action_goal",
    ):
        assert forbidden not in payload


def test_diagnostic_main_accepts_current_raw_pair_without_m0_message(tmp_path):
    out = tmp_path / "diag.json"
    _INJECTED.append(make_observations_message())
    _INJECTED.append(make_semantics_message())
    rc = main(
        [
            "--single-frame-diagnostic",
            "--output",
            str(out),
            "--timeout-sec",
            "2.0",
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["diagnostic_only"] is True
    assert payload["m0_status"] == "NO_M0_MESSAGE"


@pytest.mark.parametrize(
    ("obs_overrides", "sem_overrides", "code"),
    [
        (
            {"stamp_ns": STAMP_NS + 1},
            {},
            FAIL_HEADER_MISMATCH,
        ),
        (
            {},
            {"label": "2"},
            FAIL_TARGET_IDENTITY_MISMATCH,
        ),
        (
            {"depth_valid": False},
            {},
            FAIL_NO_M0_READY,
        ),
        (
            {
                "surface_geometry": {
                    "panel_outward_normal_x": 0.0,
                    "panel_outward_normal_y": 0.0,
                    "panel_outward_normal_z": 0.0,
                }
            },
            {},
            FAIL_INVALID_NORMAL,
        ),
        (
            {"depth_valid_ratio": 0.2},
            {},
            FAIL_NO_M0_READY,
        ),
        (
            {
                "surface_geometry": {
                    "source_frame_id": "",
                    "canonical_frame_id": "",
                }
            },
            {},
            FAIL_STALE_TARGET,
        ),
        (
            {},
            {"semantic_age_sec": 3.0},
            FAIL_STALE_TARGET,
        ),
        (
            {},
            {"metadata": {"association_valid": False}},
            FAIL_TARGET_IDENTITY_MISMATCH,
        ),
        (
            {},
            {"pressable": False},
            FAIL_TARGET_IDENTITY_MISMATCH,
        ),
    ],
)
def test_diagnostic_main_timeout_failure_codes(
    tmp_path,
    capsys,
    obs_overrides,
    sem_overrides,
    code,
):
    out = tmp_path / "diag.json"
    _INJECTED.append(make_observations_message(**obs_overrides))
    _INJECTED.append(make_semantics_message(**sem_overrides))
    rc = main(
        [
            "--single-frame-diagnostic",
            "--output",
            str(out),
            "--timeout-sec",
            "0.2",
        ]
    )
    assert rc == 2
    assert not out.exists()
    captured = capsys.readouterr().out
    assert f"freeze_failure={code}" in captured


def test_diagnostic_main_timeout_without_streams(tmp_path, capsys):
    out = tmp_path / "diag.json"
    rc = main(
        [
            "--single-frame-diagnostic",
            "--output",
            str(out),
            "--timeout-sec",
            "0.2",
        ]
    )
    assert rc == 2
    captured = capsys.readouterr().out
    assert "freeze_failure=NO_M0_READY" in captured
    assert "no_m0_or_raw_pair" in captured


# --- Hash / tamper / overwrite --------------------------------------------


def test_canonical_hash_and_tamper_detection():
    payload = {"mode": "test", "value": 1, "nested": {"b": 2, "a": 1}}
    stamped = with_canonical_hash(payload)
    digest = stamped["canonical_payload_sha256"]
    assert digest == hashlib.sha256(canonical_bytes(payload)).hexdigest()
    tampered = {
        key: value
        for key, value in stamped.items()
        if key != "canonical_payload_sha256"
    }
    tampered["value"] = 2
    assert hashlib.sha256(canonical_bytes(tampered)).hexdigest() != digest


def test_saved_artifact_hash_verifies(tmp_path):
    out = tmp_path / "diag.json"
    _INJECTED.append(make_observations_message())
    _INJECTED.append(make_semantics_message())
    rc = main(
        [
            "--single-frame-diagnostic",
            "--output",
            str(out),
            "--timeout-sec",
            "2.0",
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    digest = payload["canonical_payload_sha256"]
    body = {
        key: value
        for key, value in payload.items()
        if key != "canonical_payload_sha256"
    }
    assert digest == hashlib.sha256(canonical_bytes(body)).hexdigest()


def test_refuses_to_overwrite(tmp_path, capsys):
    out = tmp_path / "diag.json"
    _INJECTED.append(make_observations_message())
    _INJECTED.append(make_semantics_message())
    rc = main(
        [
            "--single-frame-diagnostic",
            "--output",
            str(out),
            "--timeout-sec",
            "2.0",
        ]
    )
    assert rc == 0
    original = out.read_text()
    _INJECTED.append(make_observations_message())
    _INJECTED.append(make_semantics_message())
    rc = main(
        [
            "--single-frame-diagnostic",
            "--output",
            str(out),
            "--timeout-sec",
            "2.0",
        ]
    )
    assert rc == 3
    captured = capsys.readouterr().out
    assert f"freeze_failure={FAIL_OUTPUT_EXISTS}" in captured
    assert out.read_text() == original
