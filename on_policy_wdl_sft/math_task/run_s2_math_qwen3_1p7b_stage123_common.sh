#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/qwen3_1p7b_math_stage123_resource_profile.sh"
: "${RUN_PREFIX:?RUN_PREFIX required}"
: "${BASE_MODEL_PATH:?selected Model1 path required}"
: "${MODEL2_PATH:?merged Stage1 Model2 path required}"
: "${TRAIN_FILE:?TRAIN_FILE required}"
: "${TOTAL_TRAINING_STEPS:?TOTAL_TRAINING_STEPS required}"
: "${WDL_SFT_BETA:?WDL_SFT_BETA required}"
export JOINT_VALIDATION_VIEWS=${JOINT_VALIDATION_VIEWS:-"[model1,model2]"}
export BEST_CKPT_METRIC_KEY=val-core/model2/math7_macro/acc/mean@3
mapfile -t macro_overrides < <(math_stage123_macro_overrides)
exec bash "${SCRIPT_DIR}/../staged_v1/_run_stage2_model2_rollout_common.sh" "${macro_overrides[@]}" "$@"
