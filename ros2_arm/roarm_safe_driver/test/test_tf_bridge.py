from pathlib import Path

import pytest

from roarm_safe_driver.tf_bridge import DescriptionError, build_phase4_robot_description


PACKAGE = Path(__file__).resolve().parents[1]
LAUNCH = PACKAGE / "launch" / "phase4_tf_bridge.launch.py"


def _vendor_description(*, world_joint: str = "world_to_base_link", hand_parent: str = "link5"):
    links = "".join(f'<link name="{name}"/>' for name in (
        "world", "base_link", "link1", "link2", "link3", "link4", "link5", "hand_tcp", "gripper_link"
    ))
    joints = [
        f'<joint name="{world_joint}" type="fixed"><parent link="world"/><child link="base_link"/></joint>',
        '<joint name="base_link_to_link1" type="revolute"><parent link="base_link"/><child link="link1"/></joint>',
        '<joint name="link1_to_link2" type="revolute"><parent link="link1"/><child link="link2"/></joint>',
        '<joint name="link2_to_link3" type="revolute"><parent link="link2"/><child link="link3"/></joint>',
        '<joint name="link3_to_link4" type="revolute"><parent link="link3"/><child link="link4"/></joint>',
        '<joint name="link4_to_link5" type="revolute"><parent link="link4"/><child link="link5"/></joint>',
        f'<joint name="link5_to_hand_tcp" type="fixed"><parent link="{hand_parent}"/><child link="hand_tcp"/></joint>',
        '<joint name="link5_to_gripper_link" type="revolute"><parent link="link5"/><child link="gripper_link"/></joint>',
    ]
    return f'<robot name="roarm_m3">{links}{"".join(joints)}</robot>'


def test_bridge_removes_only_vendor_demo_root_and_keeps_fk_chain():
    description = build_phase4_robot_description(_vendor_description())
    assert 'name="world"' not in description
    assert "world_to_base_link" not in description
    assert 'name="link5_to_hand_tcp"' in description
    assert 'parent link="link5"' in description


@pytest.mark.parametrize(
    "xml",
    [
        _vendor_description(world_joint="different_root_joint"),
        _vendor_description(hand_parent="gripper_link"),
        '<robot name="not_roarm_m3"/>',
    ],
)
def test_bridge_rejects_unexpected_vendor_model_or_fk_chain(xml):
    with pytest.raises(DescriptionError):
        build_phase4_robot_description(xml)


def test_phase4_launch_is_tf_only_and_uses_the_safe_builder():
    source = LAUNCH.read_text()
    assert "build_phase4_robot_description" in source
    assert "import xacro" not in source
    assert 'package="robot_state_publisher"' in source
    assert 'executable="static_transform_publisher"' in source
    for required_arg in (
        '"--x", "0"', '"--y", "0"', '"--z", "0"',
        '"--qx", "0"', '"--qy", "0"', '"--qz", "0"', '"--qw", "1"',
        '"--frame-id", "roarm_base_link"',
        '"--child-frame-id", "base_link"',
    ):
        assert required_arg in source
    for forbidden in (
        'executable="feedback_monitor"',
        'executable="safe_driver"',
        'package="roarm_driver"',
        'package="joint_state_publisher"',
        'package="moveit_ros_move_group"',
        'executable="rviz2"',
    ):
        assert forbidden not in source
