import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.annotation_web.distance import (
    DistanceEstimationError,
    calculate_panel_distance,
    estimate_panel_distance,
)
from scripts.annotation_web.import_capture import (
    SessionValidationError,
    build_rsync_command,
    normalize_date,
    validate_session,
)


def _write_session(root: Path, records: list[dict], *, extra_rgb: bool = False) -> Path:
    session = root / "lib_20260722_1958_session"
    (session / "images").mkdir(parents=True)
    (session / "depth").mkdir()
    for index, record in enumerate(records):
        (session / record["filename"]).write_bytes(b"jpeg")
        depth = np.full((8, 8), 1000 + index, dtype=np.uint16)
        assert cv2.imwrite(str(session / record["depth_filename"]), depth)
    if extra_rgb:
        (session / "images" / "extra.jpg").write_bytes(b"jpeg")
    text = "".join(json.dumps(record) + "\n" for record in records)
    (session / "metadata.jsonl").write_text(text, encoding="utf-8")
    return session


def test_validate_session_requires_exact_rgb_depth_manifest(tmp_path: Path) -> None:
    records = [
        {"filename": "images/a.jpg", "depth_filename": "depth/a_depth.png"},
        {"filename": "images/b.jpg", "depth_filename": "depth/b_depth.png"},
    ]
    report = validate_session(_write_session(tmp_path, records))
    assert report.pair_count == report.rgb_count == report.depth_count == 2


def test_validate_session_rejects_invalid_json(tmp_path: Path) -> None:
    session = _write_session(
        tmp_path,
        [{"filename": "images/a.jpg", "depth_filename": "depth/a_depth.png"}],
    )
    with (session / "metadata.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{not json}\n")
    with pytest.raises(SessionValidationError, match="invalid JSON"):
        validate_session(session)


def test_validate_session_rejects_extra_or_missing_pair(tmp_path: Path) -> None:
    records = [{"filename": "images/a.jpg", "depth_filename": "depth/a_depth.png"}]
    session = _write_session(tmp_path, records, extra_rgb=True)
    (session / "depth" / "a_depth.png").unlink()
    with pytest.raises(SessionValidationError, match="unreferenced RGB.*missing depth"):
        validate_session(session)


def test_rsync_command_is_incremental_read_only_and_excludes_overlay(tmp_path: Path) -> None:
    command = build_rsync_command(
        "agx-wifi",
        "/home/robot/datasets/raw_capture",
        "lib_20260722_1958_session",
        tmp_path,
        dry_run=True,
    )
    assert "--dry-run" in command
    assert "--partial" in command
    assert not any(arg == "--delete" or arg.startswith("--remove-source") for arg in command)
    assert "--exclude=/images/overlay/***" in command


def test_normalize_date() -> None:
    assert normalize_date("2026-07-22") == "20260722"
    assert normalize_date("20260722") == "20260722"


def test_panel_distance_excludes_button_bbox_and_invalid_depth() -> None:
    depth = np.full((100, 100), 1000, dtype=np.uint16)
    depth[40:60, 40:60] = 450  # protruding button surface
    depth[20:25, 20:25] = 0
    depth[25, 25] = 5000  # isolated background/outlier value
    result = calculate_panel_distance(depth, 0.001, [[40, 40, 60, 60]])
    assert result["distance_auto_cm"] == pytest.approx(100.0)
    assert result["distance_method"] == "panel_surface_median"
    assert 0.9 < result["distance_valid_ratio"] <= 1.0


def test_distance_scales_rgb_bboxes_and_reads_metadata_scale(tmp_path: Path) -> None:
    depth = np.full((50, 100), 800, dtype=np.uint16)
    # RGB bbox [80, 40, 120, 60] maps to [40, 20, 60, 30] in depth.
    depth[20:30, 40:60] = 300
    path = tmp_path / "depth.png"
    assert cv2.imwrite(str(path), depth)
    result = estimate_panel_distance(
        path,
        {"width": 200, "height": 100, "depth_scale_m_per_unit": 0.001},
        [{"bbox_xyxy": [80, 40, 120, 60]}],
    )
    assert result["distance_auto_cm"] == pytest.approx(80.0)


def test_distance_rejects_wrong_dtype_and_missing_valid_pixels() -> None:
    with pytest.raises(DistanceEstimationError, match="uint16"):
        calculate_panel_distance(np.ones((10, 10), dtype=np.float32), 0.001)
    with pytest.raises(DistanceEstimationError, match="not enough valid"):
        calculate_panel_distance(np.zeros((20, 20), dtype=np.uint16), 0.001)


def test_estimate_distance_rejects_bad_metadata_dimensions(tmp_path: Path) -> None:
    depth = np.full((20, 20), 1000, dtype=np.uint16)
    path = tmp_path / "depth.png"
    assert cv2.imwrite(str(path), depth)
    with pytest.raises(DistanceEstimationError, match="dimensions"):
        estimate_panel_distance(
            path,
            {"width": "bad", "height": 20, "depth_scale_m_per_unit": 0.001},
        )
