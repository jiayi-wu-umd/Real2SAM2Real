#!/bin/bash
# Set up Real2SAM2Real inference. Clones the (unmodified) VideoX-Fun framework
# at the exact commit this release was built against, then installs deps.
#
# Usage:  bash setup.sh
# Prereqs: an active conda/venv with python>=3.10 and CUDA-capable PyTorch.

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

VIDEOX_FUN_COMMIT="4a86483cc2fca7bd146108382a139e1ddc4ecdbc"
VIDEOX_FUN_ROOT="${VIDEOX_FUN_ROOT:-$REPO_ROOT/VideoX-Fun}"

if [ ! -d "$VIDEOX_FUN_ROOT/.git" ]; then
    echo "==> Cloning VideoX-Fun into $VIDEOX_FUN_ROOT"
    git clone https://github.com/aigc-apps/VideoX-Fun.git "$VIDEOX_FUN_ROOT"
fi
echo "==> Pinning VideoX-Fun to $VIDEOX_FUN_COMMIT"
git -C "$VIDEOX_FUN_ROOT" fetch --all --tags
git -C "$VIDEOX_FUN_ROOT" checkout "$VIDEOX_FUN_COMMIT"

echo "==> Installing dependencies"
pip install -r "$VIDEOX_FUN_ROOT/requirements.txt"
pip install -r "$REPO_ROOT/requirements.txt"

echo
echo "Setup complete. Next: bash download.sh && bash demo.sh"
