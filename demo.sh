#!/bin/bash
# Run a Real2SAM2Real demo on the example scene(s) under assets/demo/.
#
# Each sample dir must contain:
#   source_image.png              reference image of the real scene
#   object_animation_normal.mp4   normal-map control video (SAM3D render)
#   caption.json                  optional; {"text": "<prompt>"} used as the prompt
#
# Output: output/checkpoint-infer/<scene>/<camera>/generated.mp4
#
# Usage:  bash demo.sh    (override any path via the env vars below)

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

export VIDEOX_FUN_ROOT="${VIDEOX_FUN_ROOT:-$REPO_ROOT/VideoX-Fun}"

BASE_MODEL="${BASE_MODEL:-$REPO_ROOT/models/Diffusion_Transformer/Wan2.2-Animate-14B}"
CONFIG_PATH="${CONFIG_PATH:-$VIDEOX_FUN_ROOT/config/wan2.2/wan_civitai_animate.yaml}"
CKPT_ROOT="${CKPT_ROOT:-$REPO_ROOT/checkpoints}"     # holds checkpoint-infer/
CKPT_STEPS="${CKPT_STEPS:-infer}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/assets}"   # discovers demo/camera_control (scene/camera 2 levels deep)
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output}"
VIDEO_SAMPLE_SIZE="${VIDEO_SAMPLE_SIZE:-960}"        # 960 -> 1280x720 (training res)
GPU="${GPU:-0}"

CUDA_VISIBLE_DEVICES=${GPU} python3 real2sam2real/infer.py \
    --base_model "${BASE_MODEL}" \
    --config_path "${CONFIG_PATH}" \
    --checkpoints_root "${CKPT_ROOT}" \
    --checkpoint_steps "${CKPT_STEPS}" \
    --data_root "${DATA_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --video_sample_size ${VIDEO_SAMPLE_SIZE} \
    --gpu 0

echo "[$(date)] Done. Output under ${OUTPUT_ROOT}/checkpoint-${CKPT_STEPS}/"
