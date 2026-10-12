#!/usr/bin/env python3
"""Independently verify M1 endpoint FK with PyKDL (no ROS, no hardware)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET

import PyKDL
import yaml

from tools.press.m1_no_contact_route import (
    FK_CHAIN,
    JOINT_NAMES,
    REPO_ROOT,
    URDF_PATH,
    sha256_file,
    write_artifact_exclusive,
)


def _vector(text, default):
    return list(default) if not text else [float(value) for value in text.split()]


def kdl_fk(positions, joints):
    total = PyKDL.Frame.Identity()
    by_name = dict(zip(JOINT_NAMES, positions))
    for name in FK_CHAIN:
        joint = joints[name]
        origin = joint.find("origin")
        xyz = _vector(None if origin is None else origin.get("xyz"), [0, 0, 0])
        rpy = _vector(None if origin is None else origin.get("rpy"), [0, 0, 0])
        total = total * PyKDL.Frame(
            PyKDL.Rotation.RPY(*rpy), PyKDL.Vector(*xyz)
        )
        if joint.get("type") in {"revolute", "continuous"}:
            axis_node = joint.find("axis")
            axis = _vector(
                None if axis_node is None else axis_node.get("xyz"), [1, 0, 0]
            )
            total = total * PyKDL.Frame(
                PyKDL.Rotation.Rot(PyKDL.Vector(*axis), by_name[name])
            )
    return [total.p[index] for index in range(3)], list(total.M.GetQuaternion())


def distance(left, right):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def quaternion_error_deg(left, right):
    dot = abs(sum(a * b for a, b in zip(left, right)))
    return math.degrees(2 * math.acos(max(-1.0, min(1.0, dot))))


def verify(artifact_path: Path):
    artifact = json.loads(artifact_path.read_text())
    joints = {
        joint.get("name"): joint
        for joint in ET.fromstring(URDF_PATH.read_text()).findall("joint")
    }
    endpoints = []
    for role in ("press_standby", "pre_contact"):
        source = artifact["sources"][role]
        pose_path = REPO_ROOT / source["path"]
        pose = yaml.safe_load(pose_path.read_text())
        translation, quaternion = kdl_fk(pose["joint_positions_rad"], joints)
        stored = pose["T_roarm_base_link_hand_tcp"]
        translation_error = distance(translation, stored["translation_m"])
        orientation_error = quaternion_error_deg(
            quaternion, stored["rotation_xyzw"]
        )
        endpoints.append(
            {
                "role": role,
                "pose_id": pose["pose_id"],
                "pose_path": source["path"],
                "pose_sha256": sha256_file(pose_path),
                "kdl_translation_m": translation,
                "stored_translation_m": stored["translation_m"],
                "translation_error_m": translation_error,
                "kdl_rotation_xyzw": quaternion,
                "stored_rotation_xyzw": stored["rotation_xyzw"],
                "orientation_error_deg": orientation_error,
                "passed": translation_error <= 1e-12
                and orientation_error <= 1e-4,
            }
        )
    report = {
        "schema": "m1_no_contact_kdl_fk_verification_v1",
        "status": "passed" if all(item["passed"] for item in endpoints) else "failed",
        "scope": "independent PyKDL endpoint FK; no ROS and no hardware",
        "transform_contract": {
            "matrix": "T_parent_child = T_origin * R_axis(q)",
            "quaternion_order": "XYZW",
            "parent": "roarm_base_link_identity_to_base_link",
            "child": "hand_tcp",
        },
        "route_artifact": {
            "path": str(artifact_path.resolve()),
            "sha256": sha256_file(artifact_path),
            "canonical_payload_sha256": artifact["canonical_payload_sha256"],
        },
        "urdf": {
            "path": str(URDF_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_file(URDF_PATH),
        },
        "thresholds": {
            "translation_error_max_m": 1e-12,
            "orientation_error_max_deg": 1e-4,
        },
        "endpoints": endpoints,
        "ros_used": False,
        "serial_used": False,
        "execution_ready": False,
        "motion_authorized": False,
        "contact_authorized": False,
        "press_authorized": False,
    }
    canonical = json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    report["canonical_payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    if report["status"] != "passed":
        raise RuntimeError("PyKDL endpoint verification failed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = verify(Path(args.artifact).expanduser().resolve())
    print(
        json.dumps(
            write_artifact_exclusive(report, args.output),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
