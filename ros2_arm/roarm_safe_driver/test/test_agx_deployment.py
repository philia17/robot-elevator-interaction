from pathlib import Path

import yaml


PACKAGE = Path(__file__).resolve().parents[1]
CONFIG = PACKAGE / "config" / "agx_unified_safe.yaml"
SAFE_DEFAULTS_CONFIG = PACKAGE / "config" / "safe_defaults.yaml"
PHASE5_BASE90_CONFIG = PACKAGE / "config" / "phase5_base90_calibration.yaml"
LAUNCH = PACKAGE / "launch" / "agx_unified_safe.launch.py"


def test_agx_profile_is_hardware_active_and_uses_stable_device_identity():
    params = yaml.safe_load(CONFIG.read_text())["roarm_safe_driver"]["ros__parameters"]
    assert params["transport"] == "serial"
    assert params["dry_run"] is False
    assert params["fake_feedback"] is False
    assert params["serial_port"].startswith("/dev/serial/by-id/")
    assert "/dev/ttyUSB" not in params["serial_port"]
    assert params["joint_max_delta_deg"] == [5.0, 2.0, 2.0, 3.0, 5.0, 2.0]
    assert params["joint_acc_registers"] == [1, 1, 1, 1, 1, 1]


def test_repository_safe_defaults_are_inert():
    params = yaml.safe_load(SAFE_DEFAULTS_CONFIG.read_text())["roarm_safe_driver"][
        "ros__parameters"
    ]
    assert params["transport"] == "none"
    assert params["serial_port"] == ""
    assert params["dry_run"] is True
    assert params["fake_feedback"] is False


def test_agx_launch_contains_only_project_safe_driver():
    text = LAUNCH.read_text()
    assert 'package="roarm_safe_driver"' in text
    for forbidden in ("roarm_driver", "move_group", "joint_state_publisher", "rviz2"):
        assert forbidden not in text


def test_phase5_base90_profile_is_inert_and_only_expands_reviewed_axes():
    params = yaml.safe_load(PHASE5_BASE90_CONFIG.read_text())[
        "roarm_safe_driver"
    ]["ros__parameters"]
    assert params["transport"] == "none"
    assert params["serial_port"] == ""
    assert params["dry_run"] is True
    assert params["fake_feedback"] is False
    assert params["joint_speed_steps_s"] == [34, 17, 17, 23, 34, 17]
    assert params["joint_acc_registers"] == [1, 1, 1, 1, 1, 1]
    assert params["joint_max_delta_deg"] == [90.0, 10.0, 10.0, 15.0, 30.0, 15.0]
    assert params["tolerance_deg"] == 0.5
    assert params["convergence_samples"] == 20
