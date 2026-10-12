"""Pure Phase D1 regression tests; no ROS graph or camera is required."""

import subprocess

from elevator_vision.qos import (
    ImageQosSettings,
    compatible_subscriber_qos_for_publisher,
    is_compatible_image_qos,
    parse_image_qos,
)

from test_phase_c_logic import _project_root


def test_source_setup_safely_handles_generated_setup_under_nounset(tmp_path):
    setup_file = tmp_path / 'setup.bash'
    setup_file.write_text(
        '\n'.join([
            'if [ -z "$COLCON_TRACE" ]; then',
            '  :',
            'fi',
            'PHASE_D1_SAFE_SOURCE=ok',
            'export PHASE_D1_SAFE_SOURCE',
        ]),
        encoding='utf-8',
    )
    helper = _project_root() / 'scripts/_ros_env.sh'
    script = '\n'.join([
        'set -euo pipefail',
        'unset COLCON_TRACE',
        'source "{}"'.format(helper),
        'source_setup_safely "{}"'.format(setup_file),
        'test "${PHASE_D1_SAFE_SOURCE}" = ok',
        'case "$-" in *u*) exit 0 ;; *) exit 3 ;; esac',
    ])
    subprocess.run(['bash', '-lc', script], check=True)


def test_source_setup_safely_reports_missing_setup(tmp_path):
    helper = _project_root() / 'scripts/_ros_env.sh'
    missing = tmp_path / 'missing_setup.bash'
    script = '\n'.join([
        'set -euo pipefail',
        'source "{}"'.format(helper),
        'source_setup_safely "{}"'.format(missing),
    ])
    result = subprocess.run(
        ['bash', '-lc', script],
        check=False,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode != 0
    assert 'Setup file not found:' in result.stderr


def test_send_goal_script_uses_humble_compatible_action_cli():
    project_root = _project_root()
    script = (project_root / 'scripts/send_goal.sh').read_text(encoding='utf-8')
    action = (
        project_root
        / 'ros2_ws/src/elevator_interfaces/action/RecognizeTargetButton.action'
    ).read_text(encoding='utf-8')

    assert '--feedback' in script
    assert '\n  -f' not in script
    unsupported_spin_option = '--' + 'spin' + '-time'
    assert unsupported_spin_option not in script
    assert 'building_id' in script
    assert 'requested_floor' in script
    assert 'timeout_sec' in script
    assert 'string building_id' in action
    assert 'string requested_floor' in action
    assert 'float32 timeout_sec' in action


def test_image_qos_compatibility_for_camera_publishers():
    best_effort_volatile = ImageQosSettings('best_effort', 'volatile', 5)
    reliable_volatile = ImageQosSettings('reliable', 'volatile', 10)
    reliable_transient = ImageQosSettings('reliable', 'transient_local', 10)

    assert is_compatible_image_qos(best_effort_volatile, best_effort_volatile)
    assert not is_compatible_image_qos(
        best_effort_volatile,
        reliable_volatile,
    )
    assert is_compatible_image_qos(reliable_volatile, best_effort_volatile)
    assert not is_compatible_image_qos(
        reliable_volatile,
        reliable_transient,
    )
    assert is_compatible_image_qos(reliable_transient, reliable_volatile)

    selected = compatible_subscriber_qos_for_publisher(
        'reliable',
        'transient_local',
        12,
    )
    assert selected == ImageQosSettings('reliable', 'volatile', 12)
    assert is_compatible_image_qos(reliable_transient, selected)


def test_live_camera_launch_exposes_required_qos_and_mock_backend_arguments():
    launch_file = (
        _project_root()
        / 'ros2_ws/src/elevator_bringup/launch/live_camera.launch.py'
    ).read_text(encoding='utf-8')

    for argument in (
        'image_topic',
        'qos_reliability',
        'qos_durability',
        'queue_depth',
        'detector_backend',
        'vlm_backend',
    ):
        assert "'{}'".format(argument) in launch_file
        assert 'DeclareLaunchArgument' in launch_file

    assert "default_value='/gemini335/color/image_raw'" in launch_file
    assert "default_value='best_effort'" in launch_file
    assert "default_value='volatile'" in launch_file
    assert "default_value='10'" in launch_file
    assert "'backend': detector_backend" in launch_file
    assert "{'backend': vlm_backend}" in launch_file


def test_live_rgbd_preview_keeps_frames_and_enables_paper_semantics():
    launch_file = (
        _project_root()
        / 'ros2_ws/src/elevator_bringup/launch/live_rgbd_3d.launch.py'
    ).read_text(encoding='utf-8')

    # The quality gate is appropriate for offline selection, but a live
    # overlay must continue publishing even when the lens is covered, the
    # scene is dark, or the operator is moving the camera. The maintained
    # paper path also keeps display-only Qwen semantics enabled.
    for override in (
        "'min_sharpness': 0.0",
        "'min_brightness': 0.0",
        "'max_brightness': 255.0",
        "'max_motion': 1.0e9",
        "'semantic_enabled': True",
    ):
        assert override in launch_file


def test_parse_image_qos_accepts_supported_live_camera_overrides():
    assert parse_image_qos('best_effort', 'volatile', '10') == ImageQosSettings(
        'best_effort',
        'volatile',
        10,
    )
    assert parse_image_qos('reliable', 'transient_local', 3) == ImageQosSettings(
        'reliable',
        'transient_local',
        3,
    )
