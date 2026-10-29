#!/usr/bin/env bash
set -euo pipefail
# Call once per requested --name; immutable source and checksum are enforced.
exec bash "$(dirname -- "$0")/workflow.sh" fetch-checkpoint "$@"
