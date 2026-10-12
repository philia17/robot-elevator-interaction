"""Build the contract-safe RoArm description used by the Phase-4 TF bridge.

The vendor M3 xacro includes a demo-only ``world -> base_link`` root.  The
project contract instead makes ``base_link`` a child of ``roarm_base_link``
through a separate identity static transform.  This module derives a
kinematic-only URDF from the vendor xacro's direct link/joint definitions,
removes that demo root, and validates the exact FK chain needed for
calibration.  It deliberately does not touch serial hardware or publish ROS
messages.
"""

from __future__ import annotations

from copy import deepcopy
from xml.etree import ElementTree as ET


class DescriptionError(ValueError):
    """Raised when the vendor robot description is not the expected M3 model."""


MAIN_CHAIN = (
    ("base_link_to_link1", "revolute", "base_link", "link1"),
    ("link1_to_link2", "revolute", "link1", "link2"),
    ("link2_to_link3", "revolute", "link2", "link3"),
    ("link3_to_link4", "revolute", "link3", "link4"),
    ("link4_to_link5", "revolute", "link4", "link5"),
)
HAND_TCP_JOINT = ("link5_to_hand_tcp", "fixed", "link5", "hand_tcp")
WORLD_ROOT_JOINT = "world_to_base_link"
KINEMATIC_JOINT_CHILDREN = ("origin", "parent", "child", "axis", "limit")


def _only(elements: list[ET.Element], description: str) -> ET.Element:
    if len(elements) != 1:
        raise DescriptionError(f"expected exactly one {description}, found {len(elements)}")
    return elements[0]


def _joint_endpoints(joint: ET.Element) -> tuple[str | None, str | None]:
    parent = joint.find("parent")
    child = joint.find("child")
    return (
        None if parent is None else parent.get("link"),
        None if child is None else child.get("link"),
    )


def _validate_main_chain(root: ET.Element) -> None:
    links = [link.get("name") for link in root.findall("link")]
    required_links = {"base_link", "link1", "link2", "link3", "link4", "link5", "hand_tcp"}
    missing_links = sorted(required_links - set(links))
    if missing_links:
        raise DescriptionError(f"missing required links: {', '.join(missing_links)}")
    if "world" in links:
        raise DescriptionError("vendor world link was not removed")

    joints = root.findall("joint")
    for name, joint_type, parent_link, child_link in MAIN_CHAIN + (HAND_TCP_JOINT,):
        joint = _only([item for item in joints if item.get("name") == name], f"joint {name}")
        if joint.get("type") != joint_type:
            raise DescriptionError(f"joint {name} must be {joint_type}")
        if _joint_endpoints(joint) != (parent_link, child_link):
            raise DescriptionError(
                f"joint {name} must connect {parent_link} to {child_link}"
            )

    if any(joint.get("name") == WORLD_ROOT_JOINT for joint in joints):
        raise DescriptionError("vendor world root joint was not removed")
    if any("world" in _joint_endpoints(joint) for joint in joints):
        raise DescriptionError("a remaining joint references the vendor world frame")
    if any(_joint_endpoints(joint)[1] == "base_link" for joint in joints):
        raise DescriptionError("base_link must be the root after removing vendor world")


def build_phase4_robot_description(vendor_xacro_xml: str) -> str:
    """Build a kinematic-only, validated URDF from the vendor M3 xacro.

    The FK definitions are direct xacro children; material, mesh, Gazebo and
    transmission includes are intentionally excluded because they do not
    affect kinematics and are unavailable on the current AGX runtime.
    """

    try:
        root = ET.fromstring(vendor_xacro_xml)
    except ET.ParseError as exc:
        raise DescriptionError(f"invalid vendor xacro XML: {exc}") from exc
    if root.tag != "robot" or root.get("name") != "roarm_m3":
        raise DescriptionError("expected vendor robot named roarm_m3")

    world_link = _only(
        [link for link in root.findall("link") if link.get("name") == "world"],
        "vendor world link",
    )
    world_joint = _only(
        [joint for joint in root.findall("joint") if joint.get("name") == WORLD_ROOT_JOINT],
        "vendor world root joint",
    )
    if world_joint.get("type") != "fixed" or _joint_endpoints(world_joint) != ("world", "base_link"):
        raise DescriptionError("vendor world root joint must be fixed world -> base_link")

    kinematic_root = ET.Element("robot", {"name": "roarm_m3"})
    for link in root.findall("link"):
        name = link.get("name")
        if name is None:
            raise DescriptionError("vendor link without a name")
        if link is not world_link:
            ET.SubElement(kinematic_root, "link", {"name": name})
    for joint in root.findall("joint"):
        if joint is world_joint:
            continue
        name = joint.get("name")
        joint_type = joint.get("type")
        if name is None or joint_type is None:
            raise DescriptionError("vendor joint without name or type")
        kinematic_joint = ET.SubElement(
            kinematic_root, "joint", {"name": name, "type": joint_type}
        )
        for child_name in KINEMATIC_JOINT_CHILDREN:
            child = joint.find(child_name)
            if child is not None:
                kinematic_joint.append(deepcopy(child))

    _validate_main_chain(kinematic_root)
    return ET.tostring(kinematic_root, encoding="unicode")
