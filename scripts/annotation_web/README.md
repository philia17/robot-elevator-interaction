# Mac RGB-D Annotation Web App

Local annotation app for raw elevator RGB-D capture sessions. Imported RGB,
depth, and `metadata.jsonl` files are treated as immutable. All review state is
stored under the selected workspace's `_annotations/` directory.

## Today's workspace

```text
/home/developer/Datasets/Lift_Robot/annotation_workspace/20260722
```

It contains 114 validated RGB-D pairs from six sessions: `pmb`, `iamet`,
`ieb`, `db`, `lib`, and `pb`.

## Start

```bash
cd /home/developer/Lift_Robot
bash scripts/annotation_web/start_annotation_web.sh
```

The service listens on `http://127.0.0.1:8765` and opens the browser. It does
not upload images or expose the service beyond the Mac.

## Let teammates annotate by building

Keep the dataset on this Mac and start one token-protected server for the local
network:

```bash
bash scripts/annotation_web/start_annotation_web.sh --team-mode --no-browser
```

The terminal prints a private team URL. Open it once on each teammate's
computer, then select a building or add `?building=pmb` (or `db`, `ieb`, `lib`,
`pb`, `iamet`) to the URL. The page keeps the building in the address bar, so
the resulting URL can be copied directly.

Everyone writes to the same `_annotations/` workspace on this Mac. Assign each
building to only one person at a time to avoid two browsers editing the same
photo. Keep the team URL private: it also grants access to photo deletion,
including synchronized AGX deletion.

Deleting a photo in the page first moves both its local RGB image and paired
depth PNG into the workspace `_trash/` directory after SHA-256 verification,
then deletes the exact metadata-backed RGB-D pair on AGX. The AGX helper also
removes its matching `metadata.jsonl` row. If AGX deletion cannot be confirmed,
the local pair is automatically restored, so a failed request does not leave
the two machines out of sync. A successfully synchronized AGX deletion is
permanent, so the page disables Restore for that item.

## Re-import a date from AGX

Preview first, then run the incremental import:

```bash
python3 scripts/annotation_web/import_capture.py --date 20260722 --dry-run
python3 scripts/annotation_web/import_capture.py --date 20260722
```

The importer itself never sends a delete command to AGX. It excludes detection
overlays and requires an exact RGB/depth/metadata match after transfer.

## Local AI pre-annotation

Prepare the existing manual GT as a local YOLO dataset:

```bash
python3 scripts/annotation_web/preannotate.py prepare \
  --gt-root datasets/ground_truth \
  --output ~/Datasets/Lift_Robot/annotation_workspace/20260722/_ai/yolo_dataset
```

Train on Apple Silicon and write AI drafts:

```bash
python3 scripts/annotation_web/preannotate.py train \
  --dataset-yaml ~/Datasets/Lift_Robot/annotation_workspace/20260722/_ai/yolo_dataset/dataset.yaml \
  --output ~/Datasets/Lift_Robot/annotation_workspace/20260722/_ai/runs

python3 scripts/annotation_web/preannotate.py predict \
  --workspace ~/Datasets/Lift_Robot/annotation_workspace/20260722 \
  --weights ~/Datasets/Lift_Robot/annotation_workspace/20260722/_ai/runs/button_preannotation/weights/best.pt
```

AI boxes remain `ai_draft` until reviewed in the page. Existing annotations
are not replaced unless `predict --force` is explicitly used.

An optional second local model can prefill common semantic labels. Prepare it
with `prepare --class-mode label`, train it, and pass its `best.pt` to
`predict --label-weights`. Labels without sufficient training evidence remain
blank and must be confirmed in the page.

## Export reviewed GT

```bash
python3 scripts/annotation_web/export_annotations.py \
  --workspace ~/Datasets/Lift_Robot/annotation_workspace/20260722 \
  --output ~/Datasets/Lift_Robot/annotation_exports/20260722
```

Excluded photos are omitted and listed in `excluded_manifest.json`. The export
uses the project's legacy `*_gt.json` contract.

## Tests

```bash
python3 -m pytest -q scripts/annotation_web/tests
```
