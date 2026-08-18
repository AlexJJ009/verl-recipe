#!/usr/bin/env bash
# Stage1 wrapper for the Stage123 family; all resource values come from one profile.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/qwen3_1p7b_stage123_resource_profile.sh"
: "${STAGE123_RUN_ID:?STAGE123_RUN_ID required}"
source "${SCRIPT_DIR}/stage123_manifest_gate.sh"
stage123_require_formal_admission "$STAGE123_RUN_ID"
: "${RUN_PREFIX:?}"
: "${INIT_MODEL_PATH:?}"
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.1}
export LOSS_MODE=${LOSS_MODE:-wdl_sft}
export LR=${LR:-1e-6}
export LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-0}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:?}
export CODE_ONLINE_LCB_V5_SUBSET_VAL_FILE=${CODE_ONLINE_LCB_V5_SUBSET_VAL_FILE:-/data-1/dataset/code/verl_rl/online_full_livecodebench_v5/official_livecodebench_val.parquet}
export CODE_VAL_FILES=${CODE_VAL_FILES:-"['/data-1/dataset/code/verl_rl/online_full_humaneval_plus/official_humaneval_plus_val.parquet','/data-1/dataset/code/verl_rl/online_full_mbpp_plus/official_mbpp_plus_val.parquet','$CODE_ONLINE_LCB_V5_SUBSET_VAL_FILE']"}
export TEST_FILES="$CODE_VAL_FILES"
stage123_print_profile STAGE1
mapfile -t macro_overrides < <(code_stage123_macro_overrides)
exec bash "${SCRIPT_DIR}/run_s1_code_base.sh" "${macro_overrides[@]}" "$@"
