#!/usr/bin/env bash
# Run VLM evaluation on RGB-D dataset captures (RGB frames only).
#
# Usage:
#   ./run_vlm_on_dataset.sh library
#   ./run_vlm_on_dataset.sh all
#
# Requires models at /data/models/ (see elevator_demo*.py env vars).

set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/data/datasets}"
MAX_DIM="${MAX_DIM:-768}"
BUILDING="${1:-}"

if [[ -z "$BUILDING" ]]; then
  echo "Usage: $0 <building|all>" >&2
  exit 1
fi

run_qwen3() {
  local dir="$1"
  echo "=== Qwen3-VL-8B: $dir ==="
  python3 ~/scripts/sprint2/elevator_demo_qwen3.py --mode elevator \
    --image-folder "$dir" --max-dim "$MAX_DIM" \
    --output-dir "$dir/output_qwen3"
}

run_qwen25_baseline() {
  local dir="$1"
  echo "=== Qwen2.5-VL-3B baseline: $dir ==="
  python3 ~/scripts/sprint2/elevator_demo.py --mode elevator \
    --image-folder "$dir" --max-dim "$MAX_DIM" \
    --output-dir "$dir/output_qwen25"
}

process_building() {
  local name="$1"
  local dir="$DATASET_ROOT/$name"
  if [[ ! -d "$dir" ]]; then
    echo "Skip $name — directory not found: $dir" >&2
    return
  fi
  run_qwen3 "$dir"
  run_qwen25_baseline "$dir"
}

if [[ "$BUILDING" == "all" ]]; then
  for dir in "$DATASET_ROOT"/*/; do
    [[ -d "$dir" ]] || continue
    process_building "$(basename "$dir")"
  done
else
  process_building "$BUILDING"
fi

echo "Done."
