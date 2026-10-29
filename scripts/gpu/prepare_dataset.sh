#!/usr/bin/env bash
set -euo pipefail
# Validate licensed, already-extracted KITTI; never download/rewrite labels or splits.
exec bash "$(dirname -- "$0")/workflow.sh" prepare-dataset "$@"
