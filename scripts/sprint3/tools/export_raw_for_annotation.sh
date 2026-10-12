#!/usr/bin/env bash
# Pull raw_capture RGB images and metadata from AGX for GT annotation.
#
# Usage (on Mac):
#   cd /home/developer/Lift_Robot
#   bash scripts/sprint3/tools/export_raw_for_annotation.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
GT_DIR="${REPO_ROOT}/datasets/ground_truth"
IMAGES_DIR="${GT_DIR}/images"
AGX_HOST="${AGX_HOST:-agx-wifi}"

RAW_CAPTURE_SESSIONS=(
  "db_20260705_1538_session"
  "ieb_20260705_1556_session"
  "pmb_20260705_1517_session"
)

mkdir -p "${IMAGES_DIR}"

for session in "${RAW_CAPTURE_SESSIONS[@]}"; do
  remote_base="${AGX_HOST}:~/datasets/raw_capture/${session}"
  echo "Pulling ${session} images..."
  rsync -avz --include='*_rgb.jpg' --include='*_rgb.jpeg' --include='*_rgb.png' --exclude='*' \
    "${remote_base}/images/" "${IMAGES_DIR}/"

  echo "Pulling ${session} metadata..."
  rsync -avz \
    "${remote_base}/metadata.jsonl" \
    "${GT_DIR}/${session}_metadata.jsonl"
done

count="$(find "${IMAGES_DIR}" -maxdepth 1 -type f \( -name '*_rgb.jpg' -o -name '*_rgb.jpeg' -o -name '*_rgb.png' \) | wc -l | tr -d ' ')"
echo "Images ready: datasets/ground_truth/images/ (应该 51 张，当前 ${count} 张)"
