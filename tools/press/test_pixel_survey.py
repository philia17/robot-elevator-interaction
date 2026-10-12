from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from press.pixel_survey import PixelSurveyError, build_record, validate_record


ROOT = Path(__file__).parents[2]
IMAGE = ROOT / "temp/t102_smooth_precontact_measured_handoff_20260815/wrist_camera_precontact_20260815_205426.png"


def record():
    return build_record(IMAGE, width=852, height=638, button_u=510, button_v=501,
                        tool_u=404, tool_v=540, timestamp_utc="2026-08-15T20:54:26Z",
                        target_identity="button_1", final_q_identity="NOT_PROVIDED_IN_SOURCE_ARTIFACT",
                        final_pose_identity="NOT_PROVIDED_IN_SOURCE_ARTIFACT",
                        annotation_method="MANUAL_APPROXIMATE_SINGLE_FRAME", confidence="LOW")


def test_sign_convention_and_no_control_status():
    value = record()
    assert value["signed_error_px"] == {"du": 106.0, "dv": -39.0}
    assert value["control_status"] == "NO_CONTROL_NO_CONTACT"


def test_bounds_and_missing_or_stale_metadata_fail_closed():
    with pytest.raises(PixelSurveyError, match="outside image bounds"):
        build_record(IMAGE, width=852, height=638, button_u=852, button_v=1, tool_u=1, tool_v=1,
                     timestamp_utc="2026-08-15T20:54:26Z", target_identity="button_1",
                     final_q_identity="q", final_pose_identity="pose", annotation_method="MANUAL_REVIEWED", confidence="MEDIUM")
    value = record()
    del value["timestamp_utc"]
    with pytest.raises(PixelSurveyError, match="record fields must be exact"):
        validate_record(value)
    stale = record()
    stale["metadata_state"] = "MISSING_OR_STALE"
    with pytest.raises(PixelSurveyError, match="missing or stale"):
        validate_record(stale)


def test_image_hash_and_deterministic_output():
    first, second = record(), record()
    assert first == second
    assert first["image"]["sha256"] == "44a6801d1facf4bad53cb15affcc5ef7bef7d328131787e8961a29842f0bd84a"
    tampered = copy.deepcopy(first)
    tampered["signed_error_px"]["du"] = 0
    with pytest.raises(PixelSurveyError, match="signed error"):
        validate_record(tampered)


def test_cli_writes_canonical_json_without_runtime_interfaces(tmp_path):
    output = tmp_path / "survey.json"
    subprocess.run([sys.executable, str(ROOT / "tools/press/pixel_survey.py"), str(IMAGE), "--width", "852", "--height", "638",
                    "--button-u", "510", "--button-v", "501", "--tool-u", "404", "--tool-v", "540",
                    "--timestamp-utc", "2026-08-15T20:54:26Z", "--target-identity", "button_1",
                    "--final-q-identity", "NOT_PROVIDED_IN_SOURCE_ARTIFACT", "--final-pose-identity", "NOT_PROVIDED_IN_SOURCE_ARTIFACT",
                    "--output", str(output)], cwd=ROOT, check=True)
    value = json.loads(output.read_text(encoding="utf-8"))
    validate_record(value)
    source = (ROOT / "tools/press/pixel_survey.py").read_text(encoding="utf-8")
    assert all(token not in source for token in ("import rclpy", "serial.Serial", "create_publisher", "subprocess"))
