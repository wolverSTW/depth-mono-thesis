#!/usr/bin/env bash
# Shared paths/session activation only; sourcing never starts a model.
set -euo pipefail
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd -- "$PROJECT_DIR"
env_file=${VAST_ENV_FILE:-$PROJECT_DIR/local.vast.env}
if [[ -f "$env_file" ]]; then source "$env_file"; fi
export TASK_ROOT=${TASK_ROOT:-/workspace/depth-monoia}
export KITTI_ROOT=${KITTI_ROOT:-$TASK_ROOT/datasets/KITTIDataset}
export OUTPUT_ROOT=${OUTPUT_ROOT:-$TASK_ROOT/experiments}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-$TASK_ROOT/pretrained}
export MANIFEST=${MANIFEST:-$TASK_ROOT/evidence/local.dataset.json}
export KITTI_SOURCE_SPEC=${KITTI_SOURCE_SPEC:-$PROJECT_DIR/local.kitti_sources.json}
export REFERENCE_PREDICTIONS=${REFERENCE_PREDICTIONS:-$TASK_ROOT/evidence/val_predictions}
export REFERENCE_METRICS=${REFERENCE_METRICS:-$TASK_ROOT/evidence/val_reference.json}
export ENV_NAME=${ENV_NAME:-depth-monoia-gpu}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_HOME=${TORCH_HOME:-$TASK_ROOT/torch_cache}
export MAX_JOBS=${MAX_JOBS:-4}
export CUDA_VISIBLE_DEVICES=${GPU_DEVICE:-0}
export PYTHONDONTWRITEBYTECODE=1

vast_activate() {
  [[ $(uname -s) == Linux ]] || { echo 'Vast tasks require the authorized Linux GPU server; use --help locally.' >&2; return 2; }
  command -v conda >/dev/null || export PATH=/opt/conda/bin:$PATH
  command -v conda >/dev/null || { echo 'Use the documented PyTorch Conda developer image.' >&2; return 2; }
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
}
