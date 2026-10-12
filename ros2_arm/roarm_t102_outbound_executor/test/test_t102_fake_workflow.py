"""Pure fake integration; creates no ROS node, publisher, client, or motion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

from press.precontact_qc_core import ActionReceipt, LiveIntent, PrecontactLeaseCore, PrepareReceipt, canonical_json
from press.t102_discrete_xy_core import DiscreteXYCore
from press.t102_session_template_core import bind_session, load_policy
from press.t102_v2_execution_core import V2ExecutionMachine, load_v2_package


class FakeResources:
    publisher_count = 0
    lease_active = False
    armed = False

    def open(self):
        self.lease_active = True
        self.publisher_count = 1
        self.armed = True

    def close(self):
        self.publisher_count = 0
        self.armed = False
        self.lease_active = False


def _snapshot(policy):
    base = json.loads((ROOT / "temp/t102_build_execution_release_deploy_and_move_20260815/base_target_v5.json").read_text())
    value = {
        "schema": "m0_demo_freeze_v1", "motion_authorized": False, "target_label": "BUTTON 1",
        "frozen_target": {
            "target_label": "BUTTON 1", "detection_id": "button_004", "source_stamp_ns": 1_000_000_000,
            "frame_id": "camera_optical_frame", "confidence": 0.99, "status": "READY",
            "capture_nonce": "capture-a", "snapshot_id": "capture-a", "boot_epoch": "boot-a", "cart_pose_epoch": "cart-a",
            "camera_point_m": dict(zip("xyz", base["camera_point_m"])), "motion_authorized": False,
            "surface_geometry": {"estimate_valid": True, "provenance_complete": True,
                "section_b_calibration_id": policy["base_transform"]["calibration_id"],
                "panel_outward_normal": dict(zip("xyz", base["panel_outward_normal_camera"]))},
        },
    }
    text = canonical_json(value)
    return value, text, hashlib.sha256(text.encode()).hexdigest()


def test_fake_freeze_prepare_coarse_then_one_xy_and_teardown():
    policy = load_policy()
    package = load_v2_package()
    snapshot, text, digest = _snapshot(policy)
    session = bind_session(snapshot, digest, package["start"]["positions_rad"], policy=policy, now_unix_ns=1_500_000_000)
    core = PrecontactLeaseCore()
    prepare_id = "prepare-1"
    assert core.begin_prepare(LiveIntent(prepare_id, "PREPARE_PRECONTACT", 1.0, text, digest), now_unix_s=1.0).accepted
    assert core.record_prepare(PrepareReceipt(True, "ticket", 10000, package["package_sha256"], package["package_sha256"], digest,
        session_id=session.session_id, executor_instance_nonce="executor-boot-a", template_code="TEMPLATE_MATCH"), now_monotonic=1.0).accepted
    assert core.begin_confirm(LiveIntent("move-1", "CONFIRM_PRECONTACT", 1.1, text, digest, prepare_id), now_unix_s=1.1, now_monotonic=1.1).accepted

    resources = FakeResources()
    resources.open()
    machine = V2ExecutionMachine(package, package["start"]["positions_rad"], first_published_monotonic=0.0)
    segment = list(package["start"]["positions_rad"])
    for index in range(3):
        target = package["outbound_targets_rad"][index]
        actual = [a + 0.9 * (b-a) for a, b in zip(segment, target)]
        assert machine.consider_feedback(actual, feedback_age_s=0.01, now_monotonic=index+1.0) is not None
        segment = actual
    terminal = machine.validate_terminal("CONVERGED_DISARMED", package["endpoint"]["joint_positions_rad"])
    resources.close()
    assert core.record_action_result(ActionReceipt(True, "DYNAMIC_PRECONTACT_STOPPED_DISARMED", digest,
        package["package_sha256"], package["package_sha256"], digest, session_id=session.session_id,
        endpoint_max_joint_error_rad=terminal["maximum_final_joint_error_rad"], terminal_joint_target_publishers=resources.publisher_count,
        terminal_precontact_lease_active=resources.lease_active, terminal_armed=resources.armed)).accepted

    xy_policy = dict(policy["xy"])
    xy_policy["hard_joint_limits_rad"] = package["command_limits"]["hard_joint_limits_rad"]
    xy = DiscreteXYCore(session.session_id, xy_policy, runtime_enabled=True)
    q = tuple(package["endpoint"]["joint_positions_rad"])
    assert xy.measure(session_id=session.session_id, frame_stamp_ns=100, frame_age_s=0.01, button_available=True,
        tool_center_available=True, error_px=(30, 0), feedback_age_s=0.01, current_q=q).accepted
    candidate = list(q); candidate[0] += 0.06
    assert xy.apply_one(session_id=session.session_id, candidate_target_rad=candidate,
        predicted_cross_axis_drift_px=0, predicted_normal_drift_m=0).accepted
    resources.open()
    resources.close()
    assert xy.record_terminal(measured_q=candidate, new_frame_stamp_ns=101, new_error_px=(20, 0)).accepted
    assert (resources.publisher_count, resources.lease_active, resources.armed) == (0, False, False)


def test_fake_failure_always_tears_resources_down():
    resources = FakeResources()
    try:
        resources.open()
        raise RuntimeError("injected")
    except RuntimeError:
        resources.close()
    assert (resources.publisher_count, resources.lease_active, resources.armed) == (0, False, False)
