from pathlib import Path
import json

import pytest

from press.m0_target_to_base import TransformError
from press.precontact_preview import build_precontact_preview


REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = REPO_ROOT / "config/calibration/accepted/gemini335_roarm_20260803_r6_eye_on_base_relative_bundle.json"


NOW_NS = 1_750_000_000_000_000_000


def _target(*, normal=(0.0, 0.0, -1.0), complete=True, stamp_ns=NOW_NS - 100_000_000):
    frozen = {
        "target_label": "1",
        "frame_id": "camera_color_optical_frame",
        "source_stamp_ns": stamp_ns,
        "header": {"frame_id": "camera_color_optical_frame", "stamp_ns": stamp_ns},
        "camera_point_m": [0.0, 0.0, 0.55],
        "surface_geometry": {
            "panel_outward_normal": list(normal),
            "estimate_valid": complete,
            "provenance_complete": complete,
        },
    }
    package = {
        "schema": "m0_demo_freeze_v1",
        "trial_id": "test-freeze",
        "mode": "headless_one_shot",
        "motion_authorized": False,
        "target_label": "1",
        "frozen_target": frozen,
    }
    import hashlib
    import json

    package["canonical_payload_sha256"] = hashlib.sha256(
        json.dumps(package, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return package


def test_preview_is_endpoint_only_and_keeps_fifty_mm_standoff():
    preview = build_precontact_preview(_target(), bundle_path=BUNDLE, now_unix_ns=NOW_NS)

    assert preview["schema"] == "precontact_preview_v1"
    assert preview["mode"] == "DISPLAY_ONLY"
    assert preview["decision"] == "DIAGNOSTIC_ACCEPTED_NO_MOTION"
    assert preview["target_identity"] == "Button 1"
    assert preview["source_header"] == {
        "frame_id": "camera_color_optical_frame",
        "stamp_ns": NOW_NS - 100_000_000,
    }
    assert preview["precontact_distance_m"] == pytest.approx(0.05)
    assert preview["execution_ready"] is False
    assert preview["motion_authorized"] is False
    assert preview["trajectory_generated"] is False
    assert preview["ik_run"] is False
    assert preview["action_server_required"] is True
    assert preview["execution_interface"] == "NOT_IMPLEMENTED"
    assert preview["required_owner"] == "SAFE_ADAPTER_ACTION_SERVER"
    assert preview["preview_id"].startswith("precontact-preview-")
    assert len(preview["preview_hash"]) == 64
    assert preview["canonical_payload_sha256"] == preview["preview_hash"]
    assert set(preview["provenance"]) == {
        "section_b_relative_bundle",
        "fn_active_pointer",
        "fn_diagnostic_candidate",
    }


def test_preview_uses_explicit_outward_normal_without_flipping():
    preview = build_precontact_preview(_target(), bundle_path=BUNDLE, now_unix_ns=NOW_NS)
    assert preview["normal_direction_status"] == "VALID_OUTWARD"
    target = preview["target_point_base_m"]
    normal = preview["panel_outward_normal_base"]
    endpoint = preview["precontact_fingertip_base_m"]
    assert [endpoint[i] - target[i] for i in range(3)] == pytest.approx(
        [0.05 * normal[i] for i in range(3)]
    )


def test_preview_rejects_incomplete_geometry():
    with pytest.raises(TransformError, match="estimate_valid|provenance_complete"):
        build_precontact_preview(_target(complete=False), bundle_path=BUNDLE, now_unix_ns=NOW_NS)


def test_preview_rejects_unreasonable_distance():
    with pytest.raises(TransformError, match="d_pre_mm"):
        build_precontact_preview(_target(), bundle_path=BUNDLE, d_pre_mm=150.0, now_unix_ns=NOW_NS)


def test_preview_rejects_stale_package_before_geometry():
    with pytest.raises(TransformError, match="header is stale"):
        build_precontact_preview(
            _target(stamp_ns=NOW_NS - 4_000_000_000),
            bundle_path=BUNDLE,
            now_unix_ns=NOW_NS,
        )


def test_preview_rejects_missing_or_tampered_package_provenance():
    package = _target()
    del package["canonical_payload_sha256"]
    with pytest.raises(TransformError, match="canonical_payload_sha256"):
        build_precontact_preview(package, bundle_path=BUNDLE, now_unix_ns=NOW_NS)


def test_preview_rejects_missing_header_before_transform():
    package = _target()
    del package["frozen_target"]["source_stamp_ns"]
    package.pop("canonical_payload_sha256")
    import hashlib

    package["canonical_payload_sha256"] = hashlib.sha256(
        json.dumps(package, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    with pytest.raises(TransformError, match="source_stamp_ns"):
        build_precontact_preview(package, bundle_path=BUNDLE, now_unix_ns=NOW_NS)


def test_preview_rejects_tampered_fn_provenance(tmp_path):
    registry = json.loads((REPO_ROOT / "config/calibration/precontact_candidate_registry_v1.json").read_text())
    for record in registry["provenance"]:
        if record["role"] == "fn_active_pointer":
            record["sha256"] = "0" * 64
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry))
    with pytest.raises(TransformError, match="provenance hash mismatch for fn_active_pointer"):
        build_precontact_preview(
            _target(),
            bundle_path=BUNDLE,
            registry_path=registry_path,
            now_unix_ns=NOW_NS,
        )


def test_preview_rejects_inward_normal_without_silent_flip():
    with pytest.raises(TransformError, match="inward or unverified"):
        build_precontact_preview(
            _target(normal=(0.0, 0.0, 1.0)), bundle_path=BUNDLE, now_unix_ns=NOW_NS
        )
