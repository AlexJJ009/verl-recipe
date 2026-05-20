#!/usr/bin/env bash
# Thin wrapper for WDL group-advantage IS 1A.
set -xeuo pipefail

export RUN_PREFIX=${RUN_PREFIX:-"WDL-GROUP-ADV-IS-Qwen3-4B-MATH-1A"}
export LR=${LR:-5e-7}
export LOSS_MODE=${LOSS_MODE:-wdl_group_adv_is}

WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WRAPPER_SCRIPT_DIR
# shellcheck disable=SC1091
source "${WRAPPER_SCRIPT_DIR}/_common_group_adv_is.sh" "$@"
