#!/usr/bin/env bash
# Pure/offline checks only: no ROS launch, serial access, camera, microphone or model download.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo"
python_bin=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo:$repo/tools:$repo/scripts/sprint3:$repo/voice_control/src:$repo/ros2_arm/roarm_safe_driver:$repo/ros2_arm/roarm_precontact_adapter:$repo/ros2_perception/ros2_ws/src/elevator_vision:$repo/ros2_perception/ros2_ws/src/lift_robot_camera_adapter"
test_tmp=${PYTEST_TMPDIR:-"$repo/.pytest_tmp"}
mkdir -p "$test_tmp"
"$python_bin" -m pytest -q -p no:cacheprovider --import-mode=importlib --basetemp="$test_tmp/core" \
  scripts/coordinate_contract/test \
  scripts/sprint3/tests/test_detector_contract.py \
  ros2_perception/ros2_ws/src/elevator_vision/test/test_m0_target_core.py \
  ros2_perception/ros2_ws/src/elevator_vision/test/test_rgbd_geometry.py \
  ros2_perception/ros2_ws/src/lift_robot_camera_adapter/test/test_core.py \
  ros2_perception/ros2_ws/src/lift_robot_camera_adapter/test/test_apriltag_axis_core.py \
  ros2_arm/roarm_safe_driver/test/test_core.py \
  ros2_arm/roarm_safe_driver/test/test_torque_transaction.py \
  tools/press/test_alignment_guidance.py \
  tools/press/test_button1_xy_align.py \
  tools/press/test_installed_tool_envelope.py \
  tools/press/test_static_installation_measurement_bundle.py \
  tools/press/test_p_depth_observer.py \
  tools/press/test_precontact_status_core.py
(
  cd voice_control
  "$python_bin" -m pytest -q -p no:cacheprovider --basetemp="$test_tmp/voice" tests
)
