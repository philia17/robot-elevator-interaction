from pathlib import Path

import yaml


PACKAGE = Path(__file__).resolve().parents[1]
MONITOR = PACKAGE / "roarm_safe_driver" / "monitor_node.py"
CONFIG = PACKAGE / "config" / "webui_monitor.yaml"
LAUNCH = PACKAGE / "launch" / "webui_monitor.launch.py"


def test_monitor_source_exposes_no_control_or_serial_write_path():
    source = MONITOR.read_text()
    assert "create_subscription" not in source
    assert "create_service" not in source
    assert ".write(" not in source
    assert "FEEDBACK_REQUEST" not in source


def test_monitor_uses_stable_serial_identity():
    params = yaml.safe_load(CONFIG.read_text())["roarm_feedback_monitor"]["ros__parameters"]
    assert params["serial_port"].startswith("/dev/serial/by-id/")
    assert "/dev/ttyUSB" not in params["serial_port"]
    assert params["baud_rate"] == 115200


def test_monitor_launch_contains_only_read_only_executable():
    source = LAUNCH.read_text()
    assert 'executable="feedback_monitor"' in source
    for forbidden in ("safe_driver", "roarm_driver", "move_group", "rviz2"):
        assert f'package="{forbidden}"' not in source
