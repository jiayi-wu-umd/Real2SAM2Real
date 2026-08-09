#!/bin/bash
# Download the base model + Real2SAM2Real inference weights (HuggingFace Hub).
#
# Usage:  bash download.sh
# Requires: pip install -U "huggingface_hub[cli]"

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-$REPO_ROOT/models/Diffusion_Transformer/Wan2.2-Animate-14B}"
echo "==> Downloading base model (Wan2.2-Animate-14B) to $BASE_MODEL_DIR"
hf download Wan-AI/Wan2.2-Animate-14B --local-dir "$BASE_MODEL_DIR"

# Real2SAM2Real weights (LoRA adapter + pose_patch_embedding). The repo places
# them under checkpoint-infer/ ; the teaser assets are ignored for inference.
RSR_DIR="${RSR_DIR:-$REPO_ROOT/checkpoints}"
echo "==> Downloading Real2SAM2Real weights to $RSR_DIR"
hf download JiayiWuLeo/Real2SAM2Real \
    --include "checkpoint-infer/*" \
    --local-dir "$RSR_DIR"

echo
echo "Download complete."
echo "  Base:    $BASE_MODEL_DIR"
echo "  Weights: $RSR_DIR/checkpoint-infer/"
