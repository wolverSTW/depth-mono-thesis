#!/usr/bin/env bash
# Shared orchestration: research.py remains the only gate/training implementation.
set -euo pipefail
experiment=${1:?Specify A0, A1, A2 or A3}
shift
case "$experiment" in
  A0) config=configs/baseline/A0.yaml; run_folder=A0_baseline ;;
  A1) config=configs/ablation/A1.yaml; run_folder=A1_simple_fusion ;;
  A2) config=configs/ablation/A2.yaml; run_folder=A2_gated_fusion ;;
  A3) config=configs/proposed/A3.yaml; run_folder=A3_intrinsic_fusion ;;
  *) echo 'Only A0–A3 are supported.' >&2; exit 2 ;;
esac
if [[ ${1:-} == --help ]]; then
  echo "Usage: bash scripts/vast/train_${experiment,,}.sh [--verify-only] [--batch-size N] [--workers N] [--run-name NAME] [--resume RUN_DIRECTORY]"
  echo "Uses $config; full real G1–G4 must pass before training. No automatic batch/AMP changes."
  exit 0
fi
source "$(dirname -- "$0")/common.sh"
verify_only=false
resume_dir=
run_name=${RUN_NAME:-run_$(date -u +%Y%m%dT%H%M%SZ)}
batch_size=${VAST_BATCH_SIZE:-}
workers=${VAST_WORKERS:-}
while (($#)); do
  case "$1" in
    --verify-only) verify_only=true; shift ;;
    --batch-size|--workers|--run-name|--resume)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --batch-size) batch_size=$2 ;;
        --workers) workers=$2 ;;
        --run-name) run_name=$2 ;;
        --resume) resume_dir=$2 ;;
      esac
      shift 2 ;;
    *) echo "Unsupported argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo 'Invalid run name.' >&2; exit 2; }
if [[ -n "$resume_dir" ]]; then
  [[ -f "$resume_dir/config.yaml" && -f "$resume_dir/checkpoints/last.pth" ]] || { echo 'Resume requires an existing run config and last.pth.' >&2; exit 2; }
  config=$resume_dir/config.yaml
elif [[ "$verify_only" == false && -e "$OUTPUT_ROOT/$run_folder/$run_name" ]]; then
  echo 'Run already exists. Choose --run-name or explicitly --resume its directory.' >&2; exit 2
fi
[[ -f "$MANIFEST" ]] || { echo 'Missing dataset manifest; run prepare_kitti.sh with authentic inputs first.' >&2; exit 2; }
if [[ ! -d "$REFERENCE_PREDICTIONS" || ! -f "$REFERENCE_METRICS" ]]; then
  if [[ "$verify_only" == false ]]; then
    echo 'Missing independent real G4 reference predictions/metrics; no full training permitted.' >&2; exit 2
  fi
  echo 'Reference evidence missing: verification may measure G2/G3 memory but G4 will remain BLOCKED and the command will exit nonzero.' >&2
fi
[[ -z "$(git status --porcelain)" ]] || { echo 'Reviewed source must be committed and clean before GPU gates/training.' >&2; exit 2; }
vast_activate
args=(--config "$config" --experiment "$experiment" --dataset-root "$KITTI_ROOT" --manifest "$MANIFEST" --checkpoint-dir "$CHECKPOINT_DIR")
if [[ -n "$batch_size" ]]; then args+=(--batch-size "$batch_size"); fi
if [[ -n "$workers" ]]; then args+=(--workers "$workers"); fi
mkdir -p "$TASK_ROOT/evidence/gates"
gates=$TASK_ROOT/evidence/gates/${experiment}_$(date -u +%Y%m%dT%H%M%S)_${BASHPID}.json
echo "Running real configured-batch GPU gates. Evidence: $gates"
if ! python scripts/research.py verify "${args[@]}" --gpu \
    --reference-predictions "$REFERENCE_PREDICTIONS" --reference-metrics "$REFERENCE_METRICS" --output "$gates"; then
  echo "Gates failed. Inspect $gates for CUDA_OOM, measured memory or missing evidence. Batch/precision unchanged; no training started." >&2
  exit 2
fi
[[ "$verify_only" == false ]] || { echo 'Verification only: no full training started.'; exit 0; }
if [[ -n "$resume_dir" ]]; then
  exec python scripts/research.py train "${args[@]}" --gates "$gates" --run-dir "$resume_dir" --resume "$resume_dir/checkpoints/last.pth"
fi
echo "Starting explicitly requested $experiment training: $OUTPUT_ROOT/$run_folder/$run_name"
exec python scripts/research.py train "${args[@]}" --gates "$gates" --output-root "$OUTPUT_ROOT" --run-name "$run_name"
