import ast
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE / "roarm_safe_driver" / "calibration_recorder.py"
NODE_SOURCE = PACKAGE / "roarm_safe_driver" / "node.py"
PACKAGE_XML = PACKAGE / "package.xml"
SETUP = PACKAGE / "setup.py"


def test_recorder_is_read_only_and_uses_timestamped_tf():
    source = SOURCE.read_text()
    assert "JointState" in source
    assert '"/joint_states"' in source
    assert '"/calibration/apriltag/detections"' in source
    assert "target_time = Time(nanoseconds=joint.stamp_ns)" in source
    assert "lookup_transform(" in source
    for forbidden in ("create_publisher", "create_service", "create_client", ".write(", "ReadOnlySerialTransport"):
        assert forbidden not in source


def test_recorder_requires_an_explicit_status_schema_and_transient_status_qos():
    source = SOURCE.read_text()
    assert 'MONITOR_STATUS_SOURCE: "/roarm_monitor/status"' in source
    assert 'SAFE_DRIVER_STATUS_SOURCE: "/roarm_safe/feedback_status"' in source
    assert 'parser.add_argument("--capture-plan", required=True, type=Path)' in source
    assert "parsed.status_source = SAFE_DRIVER_STATUS_SOURCE" in source
    assert "DurabilityPolicy.TRANSIENT_LOCAL" in source
    assert "ReliabilityPolicy.RELIABLE" in source
    assert "get_publishers_info_by_topic" in source
    assert '"/roarm_safe/joint_target"' in source
    assert "safe-driver capture requires zero joint-target publishers" in source
    assert "validate_safe_status_window(" in source
    assert "validate_tag_window(" in source
    assert '"tag_static_window": {' in source
    assert '"window_duration_ns_minimum": TAG_WINDOW_NS' in source
    assert '"sample_count_minimum": MIN_TAG_SAMPLES' in source
    assert '"sample_gap_ns_maximum": MAX_TAG_SAMPLE_GAP_NS' in source
    assert '"capture_plan_sha256": self.parsed.capture_plan_sha256' in source
    assert '"recorder_timeout_s": self.parsed.timeout_s' in source
    assert '"frozen_source_hashes_sha256": (' in source
    assert '"driver_versions": driver_versions' in source
    assert "validate_raw_integrity(" in source
    assert "validate_preappend_integrity_chain(" in source
    assert '"preappend_raw_integrity": {' in source
    assert '"existing_sample_hashes_sha256": existing_raw_hashes' in source
    assert '"joint_safe_window": {' in source
    assert '"maximum_gap_ns": max(joint_gaps_ns)' in source
    assert '"per_joint_span_rad": joint_spans_rad' in source
    assert '"observations": safe_observations' in source
    assert '"translation_spread_m": tag_window.translation_spread_m' in source
    assert '"rotation_spread_deg": tag_window.rotation_spread_deg' in source
    assert '"aggregation_method": tag_window.aggregation_method' in source
    assert '"observations": [' in source
    assert '"torque_enabled": evidence["torque_enabled"]' in source
    assert '"torque_evidence": evidence["torque_evidence"]' in source
    assert "Buffer(cache_time=Duration(seconds=5.0))" in source
    assert "Buffer(cache_time=Duration(seconds=5.0), node=self)" not in source
    assert "parsed.timeout_s = 15.0" in source
    assert 'parser.add_argument("--capture-plan-sha256", required=True)' in source
    assert "--capture-plan bytes changed or do not match" in source
    assert "strict JSON without duplicate keys" in source
    assert "validate_active_source_hashes(" in source
    assert "active source validation failed" in source
    assert "operator-authorization-id is not bound" in source
    assert "requested sample is not the next formal sample ID" in source
    assert "append_integrity_indexes(" in source
    assert "raw_directory_lock(" in source


def test_safe_driver_status_and_joint_owner_keys_have_identical_shape():
    tree = ast.parse(SOURCE.read_text())
    assignments = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"status_owner", "joint_owner"}
        ):
            assignments[node.targets[0].id] = node.value
    assert assignments.keys() == {"status_owner", "joint_owner"}
    for value in assignments.values():
        assert isinstance(value, ast.Tuple)
        assert [
            element.attr for element in value.elts if isinstance(element, ast.Attribute)
        ] == ["node_namespace", "node_name"]


def test_safe_driver_publishes_status_aligned_to_the_joint_state_stamp():
    source = NODE_SOURCE.read_text()
    assert '"/roarm_safe/feedback_status"' in source
    assert "feedback_time = self.get_clock().now()" in source
    assert '"feedback_stamp_ns": feedback_time.nanoseconds' in source
    assert "msg.header.stamp = feedback_time.to_msg()" in source
    assert '"torque_enabled": list(feedback.torque_enabled)' in source
    assert "self.feedback_status_pub.publish(feedback_status_msg)" in source


def test_recorder_is_packaged_with_its_message_dependencies():
    package_xml = PACKAGE_XML.read_text()
    for dependency in ("apriltag_msgs", "geometry_msgs", "tf2_msgs"):
        assert f"<exec_depend>{dependency}</exec_depend>" in package_xml
    assert "calibration_recorder = roarm_safe_driver.calibration_recorder:main" in SETUP.read_text()
