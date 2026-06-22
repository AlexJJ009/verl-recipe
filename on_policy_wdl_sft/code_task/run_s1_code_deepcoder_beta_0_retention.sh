#!/usr/bin/env bash
# DeepCoder Stage1 code-task On-Policy SFT, beta=0.0, dense handoff retention.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_PREFIX=${RUN_PREFIX:-ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA0-V1-RETENTION}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.0}
export CODE_TRAIN_FILE=${CODE_TRAIN_FILE:-/data-1/dataset/code/verl_rl/deepcoder_preview_train_prompt1024_rl_format.parquet}
export TRAIN_FILE=${TRAIN_FILE:-$CODE_TRAIN_FILE}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-150}
export SAVE_FREQ=${SAVE_FREQ:-5}
export TEST_FREQ=${TEST_FREQ:-5}
export DATA_SEED=${DATA_SEED:-20260604}
export DATA_SHUFFLE=${DATA_SHUFFLE:-True}
export PROTECTED_CKPT_STEPS=${PROTECTED_CKPT_STEPS:-[30,40,50,60,70,80,90,100,110,120,130,140]}
export PROTECTED_CKPT_STRIP_OPTIMIZER=${PROTECTED_CKPT_STRIP_OPTIMIZER:-True}
export MIN_FREE_GB_FOR_CKPT=${MIN_FREE_GB_FOR_CKPT:-300}
export CODE_REWARD_TIMEOUT=${CODE_REWARD_TIMEOUT:-30}

exec bash "${SCRIPT_DIR}/run_s1_code_onpolicy_sft_beta_0_retention.sh" "$@"
