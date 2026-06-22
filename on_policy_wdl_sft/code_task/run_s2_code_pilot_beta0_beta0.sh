#!/usr/bin/env bash
set -euo pipefail
export RUN_PREFIX=${RUN_PREFIX:-CODE-S2-PILOT-BETA0-BETA0}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.0}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-20}
export TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-16}
export ROLLOUT_N=${ROLLOUT_N:-4}
export STAGE2_HANDOFF_STEP=${STAGE2_HANDOFF_STEP:-20}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_s2_code_model2_rollout_common.sh" "$@"
