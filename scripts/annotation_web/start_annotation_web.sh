#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${ANNOTATION_WORKSPACE:-${HOME}/Datasets/Lift_Robot/annotation_workspace/20260722}"

exec python3 "${SCRIPT_DIR}/app.py" --workspace "${WORKSPACE}" "$@"
