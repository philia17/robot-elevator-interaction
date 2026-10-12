import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from scripts.annotation_web.app import AgxRemoteStore, Workspace, create_app


def make_workspace(root: Path) -> tuple[Path, str]:
    session_name = "pmb_20260722_1605_session"
    session = root / session_name
    (session / "images").mkdir(parents=True)
    (session / "depth").mkdir()
    stem = "capture_20260722_160550_043"
    image = np.full((60, 80, 3), 127, dtype=np.uint8)
    depth = np.full((60, 80), 1250, dtype=np.uint16)
    assert cv2.imwrite(str(session / "images" / f"{stem}.jpg"), image)
    assert cv2.imwrite(str(session / "depth" / f"{stem}_depth.png"), depth)
    record = {
        "filename": f"images/{stem}.jpg",
        "depth_filename": f"depth/{stem}_depth.png",
        "timestamp": "2026-07-22T16:05:50.043+08:00",
        "camera": "Orbbec-Gemini-335",
        "width": 80,
        "height": 60,
        "depth_width": 80,
        "depth_height": 60,
        "depth_scale_m_per_unit": 0.001,
    }
    (session / "metadata.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    return root, f"{session_name}::{stem}"


def test_dataset_annotation_distance_and_exclusion(tmp_path: Path) -> None:
    root, item_id = make_workspace(tmp_path)
    client = TestClient(create_app(root))

    dataset = client.get("/api/dataset")
    assert dataset.status_code == 200
    assert dataset.json()["stats"]["total"] == 1
    item = dataset.json()["items"][0]
    assert item["building"] == "pmb"
    assert item["metadata"]["camera_model"] == "Orbbec-Gemini-335"

    annotation = client.get(f"/api/items/{item_id}/annotation").json()
    annotation["ground_truth"] = [
        {
            "bbox_xyxy": [20, 15, 35, 30],
            "coarse_type": "floor_button_candidate",
            "label": "1",
            "source": "ai",
            "confidence": 0.8,
            "reviewed": False,
        }
    ]
    annotation["image_meta"].update(
        {"view_angle": "front", "fill_light": "off", "polarizer": "installed"}
    )
    saved = client.put(f"/api/items/{item_id}/annotation", json=annotation)
    assert saved.status_code == 200
    assert saved.json()["review_status"] == "unreviewed"
    assert saved.json()["ground_truth"][0]["source"] == "ai"

    measured = client.post(f"/api/items/{item_id}/compute-distance")
    assert measured.status_code == 200
    assert measured.json()["image_meta"]["distance_auto_cm"] == 125.0
    assert measured.json()["image_meta"]["distance_cm"] == 125.0

    assert client.post(f"/api/items/{item_id}/exclude").json()["excluded"] is True
    session = root / "pmb_20260722_1605_session"
    assert not (session / "images" / "capture_20260722_160550_043.jpg").exists()
    assert not (session / "depth" / "capture_20260722_160550_043_depth.png").exists()
    trash = root / "_trash" / "pmb_20260722_1605_session" / "capture_20260722_160550_043"
    assert (trash / "capture_20260722_160550_043.jpg").is_file()
    assert (trash / "capture_20260722_160550_043_depth.png").is_file()
    assert client.get(f"/api/items/{item_id}/image").status_code == 200

    # A new server process must rediscover excluded files from the trash manifest.
    restarted = TestClient(create_app(root))
    assert restarted.get("/api/dataset").json()["stats"]["total"] == 0
    assert restarted.post(f"/api/items/{item_id}/restore").json()["excluded"] is False
    assert restarted.get("/api/dataset").json()["stats"]["total"] == 1
    assert (session / "images" / "capture_20260722_160550_043.jpg").is_file()
    assert (session / "depth" / "capture_20260722_160550_043_depth.png").is_file()
    assert not trash.exists()
    assert (root / "_annotations" / "pmb_20260722_1605_session").is_dir()


def test_workspace_ignores_metadata_path_escape(tmp_path: Path) -> None:
    session = tmp_path / "pmb_20260722_1605_session"
    session.mkdir()
    (session / "metadata.jsonl").write_text(
        json.dumps({"filename": "../secret.jpg", "depth_filename": "depth/a.png"}) + "\n",
        encoding="utf-8",
    )
    workspace = Workspace(tmp_path)
    assert workspace.dataset_payload()["stats"]["total"] == 0


def test_invalid_bbox_is_rejected_without_overwriting(tmp_path: Path) -> None:
    root, item_id = make_workspace(tmp_path)
    client = TestClient(create_app(root))
    before = client.get(f"/api/items/{item_id}/annotation").json()
    before["ground_truth"] = [{"bbox_xyxy": [10, 10, 10, 20]}]
    response = client.put(f"/api/items/{item_id}/annotation", json=before)
    assert response.status_code == 422
    annotation_path = (
        root
        / "_annotations"
        / "pmb_20260722_1605_session"
        / "capture_20260722_160550_043.json"
    )
    assert not annotation_path.exists()


def test_exclude_deletes_the_metadata_backed_agx_pair(tmp_path: Path) -> None:
    root, item_id = make_workspace(tmp_path)
    calls: list[dict[str, object]] = []

    def remote_runner(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"deleted": True, "metadata_rows_removed": 1}),
            stderr="",
        )

    remote = AgxRemoteStore(
        "agx-wifi", "/home/robot/datasets/raw_capture", runner=remote_runner
    )
    client = TestClient(create_app(root, remote_store=remote))

    response = client.post(f"/api/items/{item_id}/exclude")
    assert response.status_code == 200
    source = response.json()["source"]
    assert source["agx_host"] == "agx-wifi"
    assert source["agx_delete_result"]["metadata_rows_removed"] == 1
    assert len(calls) == 1
    request = json.loads(calls[0]["input"])
    assert request == {
        "session": "pmb_20260722_1605_session",
        "image": "images/capture_20260722_160550_043.jpg",
        "depth": "depth/capture_20260722_160550_043_depth.png",
        "timestamp": "2026-07-22T16:05:50.043+08:00",
    }
    session = root / "pmb_20260722_1605_session"
    assert not (session / "images" / "capture_20260722_160550_043.jpg").exists()
    trash = root / "_trash" / "pmb_20260722_1605_session" / "capture_20260722_160550_043"
    assert not trash.exists()
    assert client.get("/api/dataset").json()["stats"]["total"] == 0
    assert client.get(f"/api/items/{item_id}/annotation").status_code == 404
    restarted = TestClient(create_app(root, remote_store=remote))
    assert restarted.get("/api/dataset").json()["stats"]["total"] == 0


def test_exclude_restores_the_local_pair_when_agx_delete_fails(tmp_path: Path) -> None:
    root, item_id = make_workspace(tmp_path)

    def remote_runner(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 2, stdout=json.dumps({"deleted": False, "error": "offline"}), stderr=""
        )

    remote = AgxRemoteStore(
        "agx-wifi", "/home/robot/datasets/raw_capture", runner=remote_runner
    )
    client = TestClient(create_app(root, remote_store=remote))

    response = client.post(f"/api/items/{item_id}/exclude")
    assert response.status_code == 422
    assert "local photo was restored" in response.json()["detail"]
    session = root / "pmb_20260722_1605_session"
    assert (session / "images" / "capture_20260722_160550_043.jpg").is_file()
    assert (session / "depth" / "capture_20260722_160550_043_depth.png").is_file()
    assert not (
        root / "_trash" / "pmb_20260722_1605_session" / "capture_20260722_160550_043"
    ).exists()


def test_team_token_protects_dataset_and_sets_cookie(tmp_path: Path) -> None:
    root, _item_id = make_workspace(tmp_path)
    token = "team-token-with-16-plus-characters"
    client = TestClient(create_app(root, team_token=token))

    unauthorized = client.get("/api/dataset")
    assert unauthorized.status_code == 401

    entry = client.get(
        f"/?building=pmb&token={token}", follow_redirects=False
    )
    assert entry.status_code == 303
    assert entry.headers["location"].endswith("/?building=pmb")
    assert "lift_annotation_team=" in entry.headers["set-cookie"]
    assert "httponly" in entry.headers["set-cookie"].lower()

    authorized = client.get("/api/dataset")
    assert authorized.status_code == 200
    assert authorized.json()["stats"]["total"] == 1


def test_team_token_rejects_short_secrets(tmp_path: Path) -> None:
    root, _item_id = make_workspace(tmp_path)
    try:
        create_app(root, team_token="too-short")
    except ValueError as exc:
        assert "at least 16 characters" in str(exc)
    else:
        raise AssertionError("short team tokens must be rejected")
