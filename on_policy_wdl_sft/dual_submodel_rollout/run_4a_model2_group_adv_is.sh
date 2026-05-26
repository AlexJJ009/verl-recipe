#!/usr/bin/env bash
# 4A: model2-only rollout, fused training, group advantage + MiniRL-style IS.
set -xeuo pipefail

export RUN_PREFIX=${RUN_PREFIX:-"WDL-SFT-Qwen3-4B-MATH-4A-DUAL-M2-GROUP-ADV-IS"}
export LOSS_MODE=${LOSS_MODE:-dual_model2_group_adv_is}
export JOINT_ROLLOUT_SOURCES=${JOINT_ROLLOUT_SOURCES:-"[sub_model_1]"}
export JOINT_ROLLOUT_SELECT=${JOINT_ROLLOUT_SELECT:-sub_model_1}
export JOINT_ROLLOUT_TRAIN_ON_SELECTED_ONLY=${JOINT_ROLLOUT_TRAIN_ON_SELECTED_ONLY:-true}
export TRAIN_FILE=${TRAIN_FILE:-"/data-1/dataset/math/train_rl_format.parquet"}
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-seq-mean-token-sum}
export GAMMA_POS_SFT=${GAMMA_POS_SFT:-1.0}
export TIS_THRESHOLD=${TIS_THRESHOLD:-5.0}
export ROLLOUT_IS=${ROLLOUT_IS:-null}
export USE_KL_LOSS=${USE_KL_LOSS:-False}
export KL_LOSS_COEF=${KL_LOSS_COEF:-0.0}
export LR=${LR:-5e-7}

# MATH train has 7500 raw rows; max_prompt_length=500 filtering yields a
# 115-batch epoch with TRAIN_PROMPT_BSZ=64 and drop_last=True.
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-115}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}

WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WRAPPER_SCRIPT_DIR
# shellcheck disable=SC1091
source "${WRAPPER_SCRIPT_DIR}/_common_dual_rollout.sh" "$@"
