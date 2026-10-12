#!/usr/bin/env python3
"""Local RGB-D dataset annotation server.

The server deliberately treats imported capture sessions as immutable. Human
annotations, editable metadata, and exclusion state live below ``_annotations``
in the selected workspace. It only listens on loopback by default.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import shlex
import shutil
import socket
import subprocess
import tempfile
import threading
import webbrowser
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, urlencode

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

try:
    from .distance import estimate_panel_distance
except ImportError:  # Direct ``python app.py`` execution.
    from distance import estimate_panel_distance


SESSION_RE = re.compile(
    r"^(?P<building>[a-z]+)_(?P<date>\d{8})(?:_(?P<time>\d{4}))?_session$"
)
ITEM_ID_RE = re.compile(r"^[a-z0-9_]+_session::[A-Za-z0-9_.-]+$")
BUILDINGS = ("pmb", "db", "ieb", "trent", "lib", "pb", "tb", "iamet")
SAFE_SSH_HOST_RE = re.compile(r"^[A-Za-z0-9_.@:-]+$")
EDITABLE_META_FIELDS = {
    "building",
    "location",
    "view_angle",
    "fill_light",
    "polarizer",
    "camera_model",
    "distance_auto_cm",
    "distance_override_cm",
    "distance_cm",
    "distance_method",
    "distance_valid_ratio",
    "notes",
}


class AgxRemoteDeleteError(RuntimeError):
    """Raised when an AGX paired delete cannot be confirmed."""


class AgxRemoteStore:
    """Delete only metadata-backed capture pairs through a fixed AGX helper."""

    def __init__(
        self,
        host: str,
        root: str,
        helper: str = "/home/robot/scripts/agx/delete_capture_pair.py",
        *,
        runner: Any = subprocess.run,
    ):
        if not SAFE_SSH_HOST_RE.fullmatch(host):
            raise ValueError(f"Unsafe AGX SSH host: {host!r}")
        if not root.startswith("/") or "\n" in root or "\r" in root:
            raise ValueError("AGX capture root must be an absolute path")
        if not helper.startswith("/") or "\n" in helper or "\r" in helper:
            raise ValueError("AGX delete helper must be an absolute path")
        self.host = host
        self.root = root.rstrip("/")
        self.helper = helper
        self.runner = runner

    @staticmethod
    def _remote_relative(value: Any, field: str, top: str) -> str:
        path = PurePosixPath(str(value or ""))
        if (
            not str(value or "")
            or path.is_absolute()
            or ".." in path.parts
            or "." in path.parts
            or not path.parts
            or path.parts[0] != top
        ):
            raise AgxRemoteDeleteError(f"Unsafe remote {field}: {value!r}")
        return path.as_posix()

    def delete_pair(self, item: dict[str, Any]) -> dict[str, Any]:
        session = str(item.get("session") or "")
        if not SESSION_RE.fullmatch(session):
            raise AgxRemoteDeleteError(f"Unsafe remote session: {session!r}")
        raw = item.get("raw_metadata") or {}
        request = {
            "session": session,
            "image": self._remote_relative(raw.get("filename"), "image", "images"),
            "depth": self._remote_relative(
                raw.get("depth_filename"), "depth", "depth"
            ),
            "timestamp": raw.get("timestamp"),
        }
        remote_command = shlex.join(
            ["python3", self.helper, "--root", self.root, "--request-json"]
        )
        try:
            completed = self.runner(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=8",
                    self.host,
                    remote_command,
                ],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                timeout=35,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise AgxRemoteDeleteError(f"Cannot contact AGX: {exc}") from exc
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise AgxRemoteDeleteError(
                f"AGX returned an invalid response: {detail or 'empty response'}"
            ) from exc
        if completed.returncode != 0 or not response.get("deleted"):
            detail = response.get("error") or completed.stderr.strip() or "unknown error"
            raise AgxRemoteDeleteError(f"AGX delete failed: {detail}")
        return response


def utc_local_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"Path escapes session: {relative!r}")
    return candidate


def metadata_stem(record: dict[str, Any]) -> str:
    filename = str(record.get("filename", ""))
    return Path(filename).stem


def normalize_box(raw: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    coords = raw.get("bbox_xyxy")
    if not isinstance(coords, list) or len(coords) != 4:
        raise ValueError("bbox_xyxy must contain four coordinates")
    try:
        x1, y1, x2, y2 = (float(value) for value in coords)
    except (TypeError, ValueError) as exc:
        raise ValueError("bbox coordinates must be numeric") from exc
    x1, x2 = sorted((max(0.0, min(x1, width)), max(0.0, min(x2, width))))
    y1, y2 = sorted((max(0.0, min(y1, height)), max(0.0, min(y2, height))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError("bbox is too small")
    box: dict[str, Any] = {
        "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        "coarse_type": str(raw.get("coarse_type") or "button_candidate"),
        "label": str(raw.get("label") or "").strip(),
        "lit": bool(raw.get("lit", False)),
        "source": "ai" if raw.get("source") == "ai" else "manual",
        "reviewed": bool(raw.get("reviewed", raw.get("source") != "ai")),
    }
    confidence = raw.get("confidence")
    if confidence is not None:
        try:
            box["confidence"] = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            pass
    for key in ("model_id", "prediction_id", "label_model_id"):
        if raw.get(key) is not None:
            box[key] = str(raw[key])
    if raw.get("label_confidence") is not None:
        try:
            box["label_confidence"] = max(0.0, min(1.0, float(raw["label_confidence"])))
        except (TypeError, ValueError):
            pass
    return box


class Workspace:
    def __init__(self, root: Path, remote_store: AgxRemoteStore | None = None):
        self.root = root.expanduser().resolve()
        self.annotation_root = self.root / "_annotations"
        self.cache_root = self.root / "_cache" / "depth_previews"
        self.trash_root = self.root / "_trash"
        self._lock = threading.RLock()
        self._items: dict[str, dict[str, Any]] = {}
        self.remote_store = remote_store
        self.refresh()

    def annotation_path(self, item: dict[str, Any]) -> Path:
        return self.annotation_root / item["session"] / f"{item['stem']}.json"

    def trash_dir(self, item: dict[str, Any]) -> Path:
        return self.trash_root / item["session"] / item["stem"]

    def _read_records(self, session: Path) -> list[dict[str, Any]]:
        meta_path = session / "metadata.jsonl"
        if not meta_path.is_file():
            return []
        records: list[dict[str, Any]] = []
        for line_no, line in enumerate(meta_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{meta_path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise RuntimeError(f"{meta_path}:{line_no}: record is not an object")
            records.append(record)
        return records

    def refresh(self) -> None:
        if not self.root.is_dir():
            raise RuntimeError(f"Workspace does not exist: {self.root}")
        items: dict[str, dict[str, Any]] = {}
        for session in sorted(self.root.iterdir()):
            if not session.is_dir() or session.name.startswith("_"):
                continue
            match = SESSION_RE.match(session.name)
            if not match:
                continue
            building = match.group("building")
            for record in self._read_records(session):
                stem = metadata_stem(record)
                if not stem:
                    continue
                image_rel = str(record.get("filename") or "")
                depth_rel = str(record.get("depth_filename") or "")
                try:
                    image_path = safe_child(session, image_rel)
                    depth_path = safe_child(session, depth_rel)
                except ValueError:
                    continue
                item_id = f"{session.name}::{stem}"
                item = {
                    "id": item_id,
                    "building": building,
                    "session": session.name,
                    "stem": stem,
                    "session_path": session,
                    "image_path": image_path,
                    "depth_path": depth_path,
                    "raw_metadata": record,
                    "trashed": False,
                }
                if not image_path.is_file() or not depth_path.is_file():
                    trash_dir = self.trash_dir(item)
                    manifest_path = trash_dir / "manifest.json"
                    annotation_path = self.annotation_path(item)
                    try:
                        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    trash_image = safe_child(trash_dir, str(manifest.get("trash_image") or ""))
                    trash_depth = safe_child(trash_dir, str(manifest.get("trash_depth") or ""))
                    if not annotation.get("excluded") or not trash_image.is_file() or not trash_depth.is_file():
                        continue
                    item["image_path"] = trash_image
                    item["depth_path"] = trash_depth
                    item["trashed"] = True
                items[item_id] = item
        with self._lock:
            self._items = items

    def get_item(self, item_id: str) -> dict[str, Any]:
        if not ITEM_ID_RE.fullmatch(item_id):
            raise KeyError(item_id)
        try:
            return self._items[item_id]
        except KeyError:
            raise KeyError(item_id) from None

    def default_annotation(self, item: dict[str, Any]) -> dict[str, Any]:
        raw = item["raw_metadata"]
        return {
            "schema_version": 2,
            "image_id": item["id"],
            "image_stem": item["stem"],
            "session": item["session"],
            "created_at": utc_local_iso(),
            "updated_at": utc_local_iso(),
            "review_status": "unreviewed",
            "excluded": False,
            "ground_truth": [],
            "image_meta": {
                "building": item["building"],
                "location": "unknown",
                "view_angle": "unknown",
                "fill_light": "unknown",
                "polarizer": "unknown",
                "camera_model": raw.get("camera") or "unknown",
                "distance_auto_cm": None,
                "distance_override_cm": None,
                "distance_cm": None,
                "distance_method": None,
                "distance_valid_ratio": None,
                "notes": "",
            },
            "source": {
                "image": raw.get("filename"),
                "depth": raw.get("depth_filename"),
                "timestamp": raw.get("timestamp"),
                "raw_metadata": raw,
            },
        }

    def load_annotation(self, item: dict[str, Any]) -> dict[str, Any]:
        path = self.annotation_path(item)
        if not path.is_file():
            return self.default_annotation(item)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read annotation {path}: {exc}") from exc
        default = self.default_annotation(item)
        if not isinstance(payload, dict):
            return default
        default.update(payload)
        meta = default["image_meta"] = {
            **self.default_annotation(item)["image_meta"],
            **(payload.get("image_meta") or {}),
        }
        override = meta.get("distance_override_cm")
        meta["distance_cm"] = override if override is not None else meta.get("distance_auto_cm")
        return default

    def normalize_annotation(
        self, item: dict[str, Any], payload: dict[str, Any]
    ) -> dict[str, Any]:
        current = self.load_annotation(item)
        raw_width = int(item["raw_metadata"].get("width") or 0)
        raw_height = int(item["raw_metadata"].get("height") or 0)
        if raw_width <= 0 or raw_height <= 0:
            image = cv2.imread(str(item["image_path"]), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Cannot read source image")
            raw_height, raw_width = image.shape[:2]

        raw_boxes = payload.get("ground_truth", current.get("ground_truth", []))
        if not isinstance(raw_boxes, list):
            raise ValueError("ground_truth must be a list")
        boxes = [normalize_box(box, raw_width, raw_height) for box in raw_boxes]

        incoming_meta = payload.get("image_meta") or {}
        if not isinstance(incoming_meta, dict):
            raise ValueError("image_meta must be an object")
        meta = dict(current["image_meta"])
        for key, value in incoming_meta.items():
            if key in EDITABLE_META_FIELDS:
                meta[key] = value
        if meta.get("building") not in BUILDINGS:
            raise ValueError(f"Invalid building: {meta.get('building')!r}")
        for field, choices in {
            "location": {"inside", "outside", "unknown"},
            "view_angle": {"front", "oblique", "unknown"},
            "fill_light": {"on", "off", "unknown"},
            "polarizer": {"installed", "not_installed", "unknown"},
        }.items():
            if meta.get(field) not in choices:
                raise ValueError(f"Invalid {field}: {meta.get(field)!r}")
        for field in ("distance_auto_cm", "distance_override_cm"):
            value = meta.get(field)
            if value in ("", None):
                meta[field] = None
            else:
                value = float(value)
                if value <= 0 or value > 5000:
                    raise ValueError(f"Invalid {field}: {value}")
                meta[field] = round(value, 2)
        meta["distance_cm"] = (
            meta["distance_override_cm"]
            if meta["distance_override_cm"] is not None
            else meta.get("distance_auto_cm")
        )

        review_status = str(payload.get("review_status", current["review_status"]))
        if review_status not in {"unreviewed", "ai_draft", "reviewed", "skipped"}:
            raise ValueError(f"Invalid review_status: {review_status}")
        result = {
            **current,
            "schema_version": 2,
            "image_id": item["id"],
            "image_stem": item["stem"],
            "session": item["session"],
            "updated_at": utc_local_iso(),
            "review_status": review_status,
            "excluded": bool(payload.get("excluded", current.get("excluded", False))),
            "ground_truth": boxes,
            "image_meta": meta,
        }
        return result

    def save_annotation(self, item: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            normalized = self.normalize_annotation(item, payload)
            atomic_write_json(self.annotation_path(item), normalized)
            return normalized

    def compute_distance_for(self, item: dict[str, Any]) -> dict[str, Any]:
        annotation = self.load_annotation(item)
        raw = item["raw_metadata"]
        result = estimate_panel_distance(
            item["depth_path"],
            raw,
            bboxes=[box["bbox_xyxy"] for box in annotation["ground_truth"]],
        )
        meta = dict(annotation["image_meta"])
        meta.update(result)
        annotation["image_meta"] = meta
        return self.save_annotation(item, annotation)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def exclude_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Move RGB and depth together into a reversible local trash."""
        with self._lock:
            annotation = self.load_annotation(item)
            if item.get("trashed"):
                annotation["excluded"] = True
                return self.save_annotation(item, annotation)
            image_path = item["image_path"]
            depth_path = item["depth_path"]
            if not image_path.is_file() or not depth_path.is_file():
                raise ValueError("RGB or depth source is missing; nothing was deleted")
            final_dir = self.trash_dir(item)
            if final_dir.exists():
                raise ValueError(f"Trash entry already exists: {final_dir}")
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=f".{item['stem']}.", dir=final_dir.parent))
            staged_image = staging / image_path.name
            staged_depth = staging / depth_path.name
            try:
                shutil.copy2(image_path, staged_image)
                shutil.copy2(depth_path, staged_depth)
                image_hash = self._sha256(image_path)
                depth_hash = self._sha256(depth_path)
                if self._sha256(staged_image) != image_hash or self._sha256(staged_depth) != depth_hash:
                    raise RuntimeError("Trash copy checksum mismatch")
                atomic_write_json(
                    staging / "manifest.json",
                    {
                        "schema_version": 1,
                        "image_id": item["id"],
                        "deleted_at": utc_local_iso(),
                        "original_image": str(item["raw_metadata"].get("filename") or ""),
                        "original_depth": str(item["raw_metadata"].get("depth_filename") or ""),
                        "trash_image": staged_image.name,
                        "trash_depth": staged_depth.name,
                        "image_sha256": image_hash,
                        "depth_sha256": depth_hash,
                    },
                )
                os.replace(staging, final_dir)
                annotation["excluded"] = True
                saved = self.save_annotation(item, annotation)
                image_path.unlink()
                depth_path.unlink()
                item["image_path"] = final_dir / staged_image.name
                item["depth_path"] = final_dir / staged_depth.name
                item["trashed"] = True
                return saved
            except Exception:
                if staging.exists():
                    shutil.rmtree(staging)
                if final_dir.exists() and not annotation.get("excluded"):
                    shutil.rmtree(final_dir)
                raise

    def delete_item_everywhere(self, item: dict[str, Any]) -> dict[str, Any]:
        """Delete the local pair and its AGX source pair as one user action."""
        local_annotation = self.exclude_item(item)
        if self.remote_store is None:
            return local_annotation
        try:
            remote_result = self.remote_store.delete_pair(item)
        except Exception as exc:
            try:
                self.restore_item(item)
            except Exception as rollback_exc:
                raise ValueError(
                    "AGX deletion failed and the local rollback also failed: "
                    f"{exc}; local rollback: {rollback_exc}"
                ) from exc
            raise ValueError(
                f"AGX deletion failed; the local photo was restored: {exc}"
            ) from exc

        with self._lock:
            annotation = self.load_annotation(item)
            source = dict(annotation.get("source") or {})
            source["agx_deleted_at"] = utc_local_iso()
            source["agx_host"] = self.remote_store.host
            source["agx_root"] = self.remote_store.root
            source["agx_delete_result"] = remote_result
            annotation["source"] = source
            annotation["updated_at"] = utc_local_iso()
            atomic_write_json(self.annotation_path(item), annotation)
            # Once AGX confirms permanent deletion, remove the local trash copy
            # as well. Keeping it would make the deleted photo reappear in the
            # web UI as an excluded item after a refresh.
            trash_dir = self.trash_dir(item)
            if trash_dir.exists():
                shutil.rmtree(trash_dir)
            self._items.pop(item["id"], None)
            return annotation

    def restore_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Restore a paired RGB/depth item from local trash."""
        with self._lock:
            annotation = self.load_annotation(item)
            if (annotation.get("source") or {}).get("agx_deleted_at"):
                raise ValueError(
                    "Cannot restore: the corresponding AGX pair was permanently deleted"
                )
            if not item.get("trashed"):
                annotation["excluded"] = False
                return self.save_annotation(item, annotation)
            trash_dir = self.trash_dir(item)
            manifest_path = trash_dir / "manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Cannot read trash manifest: {exc}") from exc
            original_image = safe_child(item["session_path"], str(manifest["original_image"]))
            original_depth = safe_child(item["session_path"], str(manifest["original_depth"]))
            if original_image.exists() or original_depth.exists():
                raise ValueError("Cannot restore because an original RGB or depth path already exists")
            trash_image = safe_child(trash_dir, str(manifest["trash_image"]))
            trash_depth = safe_child(trash_dir, str(manifest["trash_depth"]))
            if self._sha256(trash_image) != manifest["image_sha256"]:
                raise ValueError("Trash RGB checksum mismatch")
            if self._sha256(trash_depth) != manifest["depth_sha256"]:
                raise ValueError("Trash depth checksum mismatch")
            original_image.parent.mkdir(parents=True, exist_ok=True)
            original_depth.parent.mkdir(parents=True, exist_ok=True)
            image_tmp = original_image.with_name(f".{original_image.name}.restore")
            depth_tmp = original_depth.with_name(f".{original_depth.name}.restore")
            try:
                shutil.copy2(trash_image, image_tmp)
                shutil.copy2(trash_depth, depth_tmp)
                if self._sha256(image_tmp) != manifest["image_sha256"]:
                    raise RuntimeError("Restored RGB checksum mismatch")
                if self._sha256(depth_tmp) != manifest["depth_sha256"]:
                    raise RuntimeError("Restored depth checksum mismatch")
                os.replace(image_tmp, original_image)
                os.replace(depth_tmp, original_depth)
                annotation["excluded"] = False
                saved = self.save_annotation(item, annotation)
                shutil.rmtree(trash_dir)
                item["image_path"] = original_image
                item["depth_path"] = original_depth
                item["trashed"] = False
                return saved
            except Exception:
                image_tmp.unlink(missing_ok=True)
                depth_tmp.unlink(missing_ok=True)
                raise

    def public_item(self, item: dict[str, Any]) -> dict[str, Any]:
        annotation = self.load_annotation(item)
        item_id = quote(item["id"], safe="")
        return {
            "id": item["id"],
            "building": annotation["image_meta"].get("building", item["building"]),
            "session": item["session"],
            "stem": item["stem"],
            "status": annotation["review_status"],
            "excluded": annotation["excluded"],
            "trashed": bool(item.get("trashed")),
            "agx_deleted": bool(
                (annotation.get("source") or {}).get("agx_deleted_at")
            ),
            "box_count": len(annotation["ground_truth"]),
            "ai_box_count": sum(
                1 for box in annotation["ground_truth"]
                if box.get("source") == "ai" and not box.get("reviewed")
            ),
            "image_url": f"/api/items/{item_id}/image",
            "depth_preview_url": f"/api/items/{item_id}/depth-preview",
            "annotation_url": f"/api/items/{item_id}/annotation",
            "metadata": annotation["image_meta"],
            "timestamp": item["raw_metadata"].get("timestamp"),
        }

    def dataset_payload(self) -> dict[str, Any]:
        # Deleted items are kept internally only long enough for a local
        # recovery operation. They must not reappear in the annotation queue.
        public = [
            row
            for item in self._items.values()
            if not (row := self.public_item(item))["excluded"]
        ]
        public.sort(key=lambda row: (row.get("timestamp") or "", row["id"]))
        status_counts: dict[str, int] = {}
        building_counts: dict[str, dict[str, int]] = {}
        for row in public:
            state = "excluded" if row["excluded"] else row["status"]
            status_counts[state] = status_counts.get(state, 0) + 1
            b = building_counts.setdefault(row["building"], {"total": 0, "reviewed": 0, "excluded": 0})
            b["total"] += 1
            if row["status"] == "reviewed" and not row["excluded"]:
                b["reviewed"] += 1
            if row["excluded"]:
                b["excluded"] += 1
        return {
            "workspace": str(self.root),
            "items": public,
            "stats": {
                "total": len(public),
                "status": status_counts,
                "buildings": building_counts,
            },
            "buildings": list(BUILDINGS),
        }

    def depth_preview(self, item: dict[str, Any]) -> Path:
        target = self.cache_root / item["session"] / f"{item['stem']}.png"
        if target.is_file() and target.stat().st_mtime_ns >= item["depth_path"].stat().st_mtime_ns:
            return target
        depth = cv2.imread(str(item["depth_path"]), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.ndim != 2:
            raise RuntimeError("Cannot decode depth image")
        valid = depth > 0
        normalized = np.zeros(depth.shape, dtype=np.uint8)
        if np.any(valid):
            low, high = np.percentile(depth[valid], [2, 98])
            if high <= low:
                high = low + 1
            clipped = np.clip(depth.astype(np.float32), low, high)
            normalized[valid] = np.round((high - clipped[valid]) * 255 / (high - low)).astype(np.uint8)
        preview = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        preview[~valid] = 0
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), preview):
            raise RuntimeError("Cannot write depth preview")
        return target


def create_app(
    workspace_path: Path,
    remote_store: AgxRemoteStore | None = None,
    team_token: str | None = None,
) -> FastAPI:
    if team_token is not None and len(team_token) < 16:
        raise ValueError("Team access token must contain at least 16 characters")
    workspace = Workspace(workspace_path, remote_store=remote_store)
    static_dir = Path(__file__).with_name("static")
    app = FastAPI(title="Lift Robot RGB-D Annotation", version="2")
    app.state.workspace = workspace
    app.state.team_access_enabled = team_token is not None

    if team_token is not None:
        cookie_name = "lift_annotation_team"

        @app.middleware("http")
        async def require_team_access(
            request: Request, call_next: Any
        ) -> Response:
            query_token = request.query_params.get("token")
            cookie_token = request.cookies.get(cookie_name)
            supplied = query_token or cookie_token or ""
            if not hmac.compare_digest(supplied, team_token):
                detail = "A valid team access token is required"
                if request.url.path.startswith("/api/"):
                    return JSONResponse(status_code=401, content={"detail": detail})
                return Response(detail, status_code=401, media_type="text/plain")
            if query_token is not None:
                clean_query = urlencode(
                    [
                        (key, value)
                        for key, value in request.query_params.multi_items()
                        if key != "token"
                    ]
                )
                response = RedirectResponse(
                    str(request.url.replace(query=clean_query)), status_code=303
                )
                response.set_cookie(
                    cookie_name,
                    team_token,
                    httponly=True,
                    samesite="strict",
                    path="/",
                )
                return response
            return await call_next(request)

    @app.exception_handler(ValueError)
    async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    def find(item_id: str) -> dict[str, Any]:
        try:
            return workspace.get_item(item_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown image") from None

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "workspace": str(workspace.root),
            "items": len(workspace._items),
            "agx_delete_enabled": workspace.remote_store is not None,
            "agx_host": (
                workspace.remote_store.host if workspace.remote_store is not None else None
            ),
        }

    @app.get("/api/dataset")
    def dataset(refresh: bool = False) -> dict[str, Any]:
        if refresh:
            workspace.refresh()
        return workspace.dataset_payload()

    @app.get("/api/items/{item_id}/image")
    def image(item_id: str) -> FileResponse:
        item = find(item_id)
        return FileResponse(item["image_path"], media_type="image/jpeg")

    @app.get("/api/items/{item_id}/depth-preview")
    def depth_preview(item_id: str) -> FileResponse:
        item = find(item_id)
        return FileResponse(workspace.depth_preview(item), media_type="image/png")

    @app.get("/api/items/{item_id}/annotation")
    def get_annotation(item_id: str) -> dict[str, Any]:
        return workspace.load_annotation(find(item_id))

    @app.put("/api/items/{item_id}/annotation")
    async def put_annotation(item_id: str, request: Request) -> dict[str, Any]:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="Payload must be an object")
        return workspace.save_annotation(find(item_id), payload)

    @app.post("/api/items/{item_id}/exclude")
    def exclude(item_id: str) -> dict[str, Any]:
        return workspace.delete_item_everywhere(find(item_id))

    @app.post("/api/items/{item_id}/restore")
    def restore(item_id: str) -> dict[str, Any]:
        return workspace.restore_item(find(item_id))

    @app.post("/api/items/{item_id}/compute-distance")
    def compute_distance(item_id: str) -> dict[str, Any]:
        return workspace.compute_distance_for(find(item_id))

    @app.post("/api/bulk-metadata")
    async def bulk_metadata(request: Request) -> dict[str, Any]:
        payload = await request.json()
        ids = payload.get("ids") if isinstance(payload, dict) else None
        updates = payload.get("image_meta") if isinstance(payload, dict) else None
        if not isinstance(ids, list) or not isinstance(updates, dict):
            raise HTTPException(status_code=422, detail="ids and image_meta are required")
        invalid = sorted(set(updates) - EDITABLE_META_FIELDS)
        if invalid:
            raise HTTPException(status_code=422, detail=f"Unsupported metadata fields: {invalid}")
        saved = []
        for item_id in ids:
            item = find(str(item_id))
            current = workspace.load_annotation(item)
            current["image_meta"] = {**current["image_meta"], **updates}
            saved.append(workspace.save_annotation(item, current)["image_id"])
        return {"updated": saved, "count": len(saved)}

    @app.post("/api/compute-distances")
    async def compute_distances(request: Request) -> dict[str, Any]:
        payload = await request.json()
        ids = payload.get("ids", []) if isinstance(payload, dict) else []
        if not isinstance(ids, list):
            raise HTTPException(status_code=422, detail="ids must be a list")
        results = []
        for item_id in ids:
            item = find(str(item_id))
            annotation = workspace.compute_distance_for(item)
            results.append({"id": item_id, "image_meta": annotation["image_meta"]})
        return {"results": results, "count": len(results)}

    @app.get("/api/items/{item_id}/export-gt")
    def export_gt(item_id: str) -> Response:
        annotation = workspace.load_annotation(find(item_id))
        payload = {
            "image_stem": annotation["image_stem"],
            "annotated_at": annotation["updated_at"][:10],
            "ground_truth": [
                {
                    "bbox_xyxy": [int(round(v)) for v in box["bbox_xyxy"]],
                    "coarse_type": box["coarse_type"],
                    **({"label": box["label"]} if box.get("label") else {}),
                    **({"lit": True} if box.get("lit") else {}),
                }
                for box in annotation["ground_truth"]
            ],
            "image_meta": annotation["image_meta"],
        }
        return Response(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            media_type="application/json",
        )

    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.home() / "Datasets/Lift_Robot/annotation_workspace/20260722",
        help="Imported annotation workspace containing capture session folders.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Listen address (defaults to 127.0.0.1, or 0.0.0.0 in team mode).",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--team-mode",
        action="store_true",
        help="Allow token-protected access from other computers on the local network.",
    )
    parser.add_argument(
        "--team-token",
        default=os.environ.get("ANNOTATION_TEAM_TOKEN"),
        help="Shared team token (generated for this run when omitted in team mode).",
    )
    parser.add_argument("--agx-host", default="agx-wifi")
    parser.add_argument(
        "--agx-root", default="/home/robot/datasets/raw_capture"
    )
    parser.add_argument(
        "--agx-delete-helper",
        default="/home/robot/scripts/agx/delete_capture_pair.py",
    )
    parser.add_argument(
        "--no-agx-delete",
        action="store_true",
        help="Keep the legacy local-only trash behavior.",
    )
    return parser.parse_args()


def local_lan_ip() -> str:
    """Return the preferred LAN address without sending network traffic."""
    for interface in ("en0", "en1"):
        try:
            completed = subprocess.run(
                ["ipconfig", "getifaddr", interface],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            break
        candidate = completed.stdout.strip()
        if candidate:
            address = ipaddress.ip_address(candidate)
            if address.version == 4 and address.is_private:
                return candidate
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return str(probe.getsockname()[0])
    except OSError:
        return socket.gethostbyname(socket.gethostname())
    finally:
        probe.close()


def main() -> None:
    args = parse_args()
    host = args.host or ("0.0.0.0" if args.team_mode else "127.0.0.1")
    team_token = args.team_token
    if args.team_mode and team_token is None:
        team_token = secrets.token_urlsafe(24)
    remote_store = None
    if not args.no_agx_delete:
        remote_store = AgxRemoteStore(
            args.agx_host, args.agx_root, args.agx_delete_helper
        )
    app = create_app(
        args.workspace, remote_store=remote_store, team_token=team_token
    )
    if args.team_mode:
        lan_ip = local_lan_ip()
        print(
            f"Team access: http://{lan_ip}:{args.port}/?token={team_token}",
            flush=True,
        )
    if not args.no_browser:
        local_url = f"http://127.0.0.1:{args.port}/"
        if team_token is not None:
            local_url += f"?token={quote(team_token, safe='')}"
        threading.Timer(0.8, lambda: webbrowser.open(local_url)).start()
    uvicorn.run(app, host=host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
