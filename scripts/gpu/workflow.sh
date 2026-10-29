#!/usr/bin/env bash
set -euo pipefail

action=${1:?Specify workflow action}
shift
project_dir=${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
python_bin=${PYTHON_BIN:-python}
gpu_device=${GPU_DEVICE:-0}
env_name=${ENV_NAME:-depth-monoia-gpu}
forward=()
while (($#)); do
  case "$1" in
    --project-dir|--python|--gpu-device|--env-name)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --project-dir) project_dir=$2 ;;
        --python) python_bin=$2 ;;
        --gpu-device) gpu_device=$2 ;;
        --env-name) env_name=$2 ;;
      esac
      shift 2 ;;
    *) forward+=("$1"); shift ;;
  esac
done
cd -- "$project_dir"
export CUDA_VISIBLE_DEVICES="$gpu_device"
export PYTHONDONTWRITEBYTECODE=1
if [[ "$action" == setup ]]; then
  [[ $(uname -s) == Linux ]] || { echo 'Setup requires Linux; use local verification on Windows.' >&2; exit 2; }
  ((${#forward[@]} == 0)) || { echo 'Setup accepts --env-name, --project-dir and common flags only.' >&2; exit 2; }
  command -v nvcc >/dev/null || { echo 'Install CUDA toolkit 12.1/nvcc first.' >&2; exit 2; }
  command -v conda >/dev/null || { echo 'Install Conda first.' >&2; exit 2; }
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda run -n "$env_name" python -c 'import sys; assert sys.version_info[:2] == (3,9)' >/dev/null 2>&1; then
    conda create -n "$env_name" python=3.9 -y
  fi
  conda activate "$env_name"
  python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
  python -m pip install -r requirements/linux-cu121.txt
  python scripts/build_ops.py build_ext --build-lib build/ops
  python scripts/research.py verify-gpu
  python -m pip freeze
  exit 0
fi
exec "$python_bin" scripts/research.py "$action" "${forward[@]}"
