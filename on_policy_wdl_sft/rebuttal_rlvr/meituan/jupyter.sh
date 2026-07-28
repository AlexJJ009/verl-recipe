#!/usr/bin/env bash
set -euo pipefail

: "${EXPERIMENT:?EXPERIMENT must be set by the platform manifest}"
: "${REPO_ROOT:?REPO_ROOT must be exported by the platform shim}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAMILY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

if [ "${ALLOW_BASE_PLACEHOLDER:-0}" = "1" ] && [ "${RUN_MODE:-formal}" != "smoke" ]; then
    echo "ERROR: ALLOW_BASE_PLACEHOLDER is restricted to RUN_MODE=smoke" >&2
    exit 2
fi

RUN_SCRIPT="${FAMILY_DIR}/run_experiment.sh"
if [ ! -f "$RUN_SCRIPT" ]; then
    echo "ERROR: unified experiment entrypoint not found: $RUN_SCRIPT" >&2
    exit 2
fi

cd "$REPO_ROOT"
exec bash "$RUN_SCRIPT" "$EXPERIMENT"
