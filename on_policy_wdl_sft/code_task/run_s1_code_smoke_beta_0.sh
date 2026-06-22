#!/usr/bin/env bash
set -euo pipefail
export RUN_PREFIX=${RUN_PREFIX:-CODE-S1-SMOKE-BETA0}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.0}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-5}
export TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-8}
export ROLLOUT_N=${ROLLOUT_N:-2}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_s1_code_base.sh" "$@"
