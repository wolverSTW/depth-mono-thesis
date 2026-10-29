#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --help ]]; then
  echo 'Usage: bash scripts/vast/setup.sh (authorized RTX 4090 Linux server only; no training)'
  exit 0
fi
(($# == 0)) || { echo 'Setup takes no arguments; customize local.vast.env.' >&2; exit 2; }
source "$(dirname -- "$0")/common.sh"
[[ $(uname -s) == Linux ]] || { echo 'Setup requires Linux CUDA on Vast.ai.' >&2; exit 2; }
[[ $EUID == 0 ]] || { echo 'This setup targets the documented root SSH template; install system prerequisites with your administrator first.' >&2; exit 2; }
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPU_DEVICE:-0}")
[[ "$gpu_name" == *'RTX 4090'* ]] || { echo "Actual GPU differs from the requested RTX 4090: $gpu_name" >&2; exit 2; }
apt-get update
apt-get install -y build-essential git ninja-build unzip curl rsync tmux libgl1 libglib2.0-0 ca-certificates
mkdir -p "$TASK_ROOT"/{evidence,pretrained,experiments,backups,setup_logs,downloads/kitti}
command -v conda >/dev/null || export PATH=/opt/conda/bin:$PATH
bash scripts/gpu/setup_environment.sh --env-name "$ENV_NAME" 2>&1 | tee "$TASK_ROOT/setup_logs/setup.log"
vast_activate
python - <<'PY'
import torch
name = torch.cuda.get_device_name(0)
print('Actual GPU:', name, 'VRAM bytes:', torch.cuda.get_device_properties(0).total_memory)
PY
python -m pip freeze > "$TASK_ROOT/setup_logs/pip-freeze.txt"
git rev-parse HEAD > "$TASK_ROOT/setup_logs/source_commit.txt"
git submodule status > "$TASK_ROOT/setup_logs/submodules.txt"
nvidia-smi > "$TASK_ROOT/setup_logs/nvidia-smi.txt"
nvcc --version > "$TASK_ROOT/setup_logs/nvcc.txt"
python scripts/research.py fetch-checkpoint --name dav2_small --checkpoint-dir "$CHECKPOINT_DIR"
echo 'Environment and verified DA-V2 download prepared. No dataset, GPU gates or training started.'
