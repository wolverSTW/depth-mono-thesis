#!/usr/bin/env bash
set -euo pipefail
experiment=
arguments=("$@")
for ((i=0; i<${#arguments[@]}; i++)); do
  if [[ "${arguments[$i]}" == --experiment ]]; then
    experiment=${arguments[$((i+1))]:-}
  fi
done
[[ "$experiment" == A1 || "$experiment" == A2 ]] || { echo 'Specify --experiment A1 or A2; A4 remains locked.' >&2; exit 2; }
exec bash "$(dirname -- "$0")/workflow.sh" train "$@"
