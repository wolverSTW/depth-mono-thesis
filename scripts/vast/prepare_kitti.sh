#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --help ]]; then
  echo 'Usage: bash scripts/vast/prepare_kitti.sh [--source-spec FILE]'
  echo 'Authorized official downloads, verified extraction, original splits and manifest only; no training.'
  exit 0
fi
source "$(dirname -- "$0")/common.sh"
if (($#)); then
  [[ $# == 2 && $1 == --source-spec ]] || { echo 'Expected --source-spec FILE' >&2; exit 2; }
  KITTI_SOURCE_SPEC=$2
fi
[[ -f "$KITTI_SOURCE_SPEC" ]] || { echo 'Missing source specification. Copy configs/vast/kitti_sources.example.json to local.kitti_sources.json and supply authorized official URLs, trusted split hashes and provenance.' >&2; exit 2; }
vast_activate
exec python scripts/research.py prepare-kitti --source-spec "$KITTI_SOURCE_SPEC" \
  --download-root "$TASK_ROOT/downloads/kitti" --dataset-root "$KITTI_ROOT" --manifest "$MANIFEST"
