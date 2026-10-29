#!/usr/bin/env bash
set -euo pipefail
resume_found=false
for argument in "$@"; do
  [[ "$argument" != --resume ]] || resume_found=true
done
[[ "$resume_found" == true ]] || { echo 'Resume requires --resume CHECKPOINT; no fresh training will be started.' >&2; exit 2; }
exec bash "$(dirname -- "$0")/workflow.sh" train "$@"
