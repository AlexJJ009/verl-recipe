#!/usr/bin/env bash
# Stage 2: joint v1 WDL-SFT, beta=0.2.
set -xeuo pipefail

export RUN_PREFIX=${RUN_PREFIX:-"WDL-SFT-STAGED-V1-Qwen3-4B-MATH-S2-BETA02"}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.2}
export LR=${LR:-5e-7}

WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WRAPPER_SCRIPT_DIR
# shellcheck disable=SC1091
source "${WRAPPER_SCRIPT_DIR}/_run_stage2_common.sh" "$@"

