#!/usr/bin/env bash
# Install Grounding DINO (Swin-T) on AGX Orin for Sprint 3 zero-shot bbox experiments.
#
# Run on AGX:
#   bash ~/scripts/sprint3/install_grounding_dino.sh
#
# Prerequisites: JetPack 36.x, CUDA visible, /data mounted, git, python3-venv.
# After install:
#   source /data/venvs/groundingdino/bin/activate
#   python3 ~/scripts/sprint3/smoke_test_grounding_dino.py

set -euo pipefail

VENV="/data/venvs/groundingdino"
REPO="/data/tools/GroundingDINO"
WEIGHTS_DIR="/data/models/groundingdino"
CHECKPOINT="$WEIGHTS_DIR/groundingdino_swint_ogc.pth"
CONFIG="$REPO/groundingdino/config/GroundingDINO_SwinT_OGC.py"

echo "=== Grounding DINO install (AGX) ==="

if [[ ! -d /data ]]; then
  echo "ERROR: /data not mounted." >&2
  exit 1
fi

mkdir -p /data/venvs /data/tools /data/models/groundingdino

if [[ ! -d "$REPO" ]]; then
  git clone --depth 1 https://github.com/IDEA-Research/GroundingDINO.git "$REPO"
else
  echo "Repo exists: $REPO"
fi

if [[ ! -d "$VENV" ]]; then
  # Reuse Jetson system torch (installed outside the venv).
  python3 -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

pip install --upgrade 'pip<26' 'setuptools>=70,<80' wheel

# Jetson: prefer system torch if already installed (JetPack wheel).
if python3 -c "import torch" 2>/dev/null; then
  echo "Using existing torch: $(python3 -c 'import torch; print(torch.__version__)')"
else
  echo "WARNING: torch not found. Install Jetson PyTorch wheel before continuing."
  echo "  See: https://forums.developer.nvidia.com/t/pytorch-for-jetson/"
  exit 1
fi

# Grounding DINO breaks on transformers 5.x (BertModel.get_head_mask removed).
# Install transformers into venv only; reuse ~/.local Jetson torch via --system-site-packages.
pip install --ignore-installed 'transformers>=4.30,<5.0'
pip install 'numpy>=1.23,<2' addict yapf opencv-python-headless
# Do NOT --ignore-installed timm/supervision — that pulls generic PyPI torch (~1.5 GB CUDA 13).
pip install --no-deps timm supervision

PIP_NO_BUILD_ISOLATION=1 pip install -e "$REPO" --no-deps

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Downloading Swin-T checkpoint..."
  wget -q --show-progress -O "$CHECKPOINT" \
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
fi

echo ""
echo "=== Install complete ==="
echo "Activate:  source $VENV/bin/activate"
echo "Smoke test:"
echo "  python3 ~/scripts/sprint3/smoke_test_grounding_dino.py \\"
echo "    --image /data/demo_images/ieb_inside_panel_full_01.jpg"
echo ""
echo "See docs/agx_grounding_dino_setup.md for Phase 4B experiment commands."
