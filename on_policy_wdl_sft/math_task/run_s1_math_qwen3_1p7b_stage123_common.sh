#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/qwen3_1p7b_math_stage123_resource_profile.sh"
: "${RUN_PREFIX:?RUN_PREFIX required}"
: "${INIT_MODEL_PATH:?INIT_MODEL_PATH required}"
: "${TRAIN_FILE:?TRAIN_FILE required}"
: "${TOTAL_TRAINING_STEPS:?TOTAL_TRAINING_STEPS required}"
: "${WDL_SFT_BETA:?WDL_SFT_BETA required}"
export LR=${LR:-1e-6}
export LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-0}
mapfile -t macro_overrides < <(math_stage123_macro_overrides)
exec bash "${SCRIPT_DIR}/../staged_v1/run_s1_base_sft.sh" data.shuffle=False "${macro_overrides[@]}" "$@"
