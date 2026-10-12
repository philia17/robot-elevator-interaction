"""OpenCV AprilTag helpers for wrist RGB eye-in-hand capture (tag36h11)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class TagDetection:
    tag_id: int
    corners_px: np.ndarray  # (4, 2) float64, TL TR BR BL in image pixels
    decision_margin: float
    hamming: int
    t_cam_tag: np.ndarray | None  # 4x4 when pose solved
    reprojection_error_px: float | None


def april_tag_dictionary():
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def april_tag_detector_parameters():
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        params = cv2.aruco.DetectorParameters_create()
    else:
        params = cv2.aruco.DetectorParameters()
    # Prefer strict detections for calibration samples.
    if hasattr(params, "aprilTagMaxHamming"):
        params.aprilTagMaxHamming = 0
    return params


def make_detector():
    dictionary = april_tag_dictionary()
    params = april_tag_detector_parameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        return ("new", cv2.aruco.ArucoDetector(dictionary, params), dictionary, params)
    return ("legacy", None, dictionary, params)


def detect_markers(gray: np.ndarray, detector_bundle):
    kind, detector, dictionary, params = detector_bundle
    if kind == "new":
        return detector.detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)


def tag_object_corners_m(edge_m: float) -> np.ndarray:
    """Black-border square corners in tag frame (z=0), TL TR BR BL."""
    half = float(edge_m) * 0.5
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=float,
    )


def provisional_vga_intrinsics(
    width: int = 640,
    height: int = 480,
    *,
    hfov_deg: float = 90.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rough K for wide-FOV VGA modules (TYS-CLB30W / GC0308 class).

    Default 90° HFOV matches the strong barrel distortion seen on AGX wrist RGB.
    Solver may refine distortion with this K held or softly constrained.
    """
    hfov = float(hfov_deg)
    fx = (width * 0.5) / np.tan(np.deg2rad(hfov * 0.5))
    fy = fx
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
    dist = np.zeros(5, dtype=float)
    return camera_matrix, dist


def transform(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, :3] = np.asarray(rotation, dtype=float)
    result[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return result


def transform_dict(value: np.ndarray) -> dict[str, list[float]]:
    quat = Rotation.from_matrix(value[:3, :3]).as_quat()
    if quat[3] < 0.0:
        quat = -quat
    return {
        "translation_m": value[:3, 3].tolist(),
        "rotation_xyzw": quat.tolist(),
    }


def from_xyzw(data: dict) -> np.ndarray:
    rotation = Rotation.from_quat(np.asarray(data["rotation_xyzw"], dtype=float)).as_matrix()
    return transform(rotation, data["translation_m"])


def detect_tag36h11(
    frame_bgr: np.ndarray,
    *,
    detector,
    tag_id: int,
    edge_m: float,
    camera_matrix: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
) -> TagDetection | None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners_list, ids, _rejected = detect_markers(gray, detector)
    if ids is None or len(ids) == 0:
        return None
    chosen = None
    for corners, marker_id in zip(corners_list, ids.flatten().tolist()):
        if int(marker_id) != int(tag_id):
            continue
        # New API: corners shape (1,4,2); legacy same.
        chosen = np.asarray(corners, dtype=float).reshape(4, 2)
        break
    if chosen is None:
        return None

    # OpenCV ArUco does not always expose aprilTag decision margin; use 0 as unknown.
    detection = TagDetection(
        tag_id=int(tag_id),
        corners_px=chosen,
        decision_margin=0.0,
        hamming=0,
        t_cam_tag=None,
        reprojection_error_px=None,
    )
    if camera_matrix is None or dist_coeffs is None:
        return detection

    object_points = tag_object_corners_m(edge_m)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        chosen,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            chosen,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    if not ok:
        return detection
    rotation, _ = cv2.Rodrigues(rvec)
    t_cam_tag = transform(rotation, tvec.reshape(3))
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    err = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - chosen, axis=1)))
    return TagDetection(
        tag_id=detection.tag_id,
        corners_px=detection.corners_px,
        decision_margin=detection.decision_margin,
        hamming=detection.hamming,
        t_cam_tag=t_cam_tag,
        reprojection_error_px=err,
    )


def draw_detection(frame: np.ndarray, detection: TagDetection | None) -> None:
    if detection is None:
        return
    pts = np.round(detection.corners_px).astype(int)
    for i in range(4):
        cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 220, 0), 2)
    center = tuple(np.round(detection.corners_px.mean(axis=0)).astype(int))
    cv2.putText(
        frame,
        f"ID{detection.tag_id}",
        (center[0] - 20, center[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 220, 0),
        2,
        cv2.LINE_AA,
    )
