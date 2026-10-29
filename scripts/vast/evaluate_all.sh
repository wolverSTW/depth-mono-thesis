#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --help ]]; then
  echo 'Usage: bash scripts/vast/evaluate_all.sh'
  echo 'Evaluate actual best.pth files with saved configs and aggregate machine-readable results; no training.'
  exit 0
fi
(($# == 0)) || { echo 'Evaluation takes no arguments; customize local.vast.env paths.' >&2; exit 2; }
source "$(dirname -- "$0")/common.sh"
vast_activate
python scripts/research.py evaluate-all --search-root "$OUTPUT_ROOT" \
  --dataset-root "$KITTI_ROOT" --manifest "$MANIFEST" --checkpoint-dir "$CHECKPOINT_DIR"
python scripts/research.py aggregate --search-root "$OUTPUT_ROOT" --output "$TASK_ROOT/model_comparison.csv"
