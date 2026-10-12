#!/usr/bin/env python3
"""Export reviewed, non-excluded annotations in the legacy GT JSON layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .app import Workspace, atomic_write_json
except ImportError:
    from app import Workspace, atomic_write_json


def export_annotations(workspace_root: Path, output: Path, include_drafts: bool = False) -> dict[str, int]:
    workspace = Workspace(workspace_root)
    output.mkdir(parents=True, exist_ok=True)
    counts = {"exported": 0, "excluded": 0, "not_reviewed": 0}
    excluded_manifest = []
    for item in workspace._items.values():
        annotation = workspace.load_annotation(item)
        if annotation["excluded"]:
            counts["excluded"] += 1
            excluded_manifest.append(
                {
                    "image_id": item["id"],
                    "image": item["raw_metadata"].get("filename"),
                    "depth": item["raw_metadata"].get("depth_filename"),
                    "session": item["session"],
                }
            )
            continue
        if annotation["review_status"] != "reviewed" and not include_drafts:
            counts["not_reviewed"] += 1
            continue
        legacy = {
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
            "capture_session": annotation["session"],
            "capture_source": annotation["source"],
        }
        target = output / item["session"] / f"{item['stem']}_gt.json"
        atomic_write_json(target, legacy)
        counts["exported"] += 1
    atomic_write_json(
        output / "excluded_manifest.json",
        {"schema_version": 1, "excluded": excluded_manifest},
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-drafts", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export_annotations(args.workspace, args.output, args.include_drafts), indent=2))


if __name__ == "__main__":
    main()
