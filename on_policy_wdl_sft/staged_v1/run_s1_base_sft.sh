#!/usr/bin/env bash
# Stage 1: single-model On-Policy SFT from Qwen3-4B-Base using v1 wdl_sft loss.
set -xeuo pipefail

export RUN_PREFIX=${RUN_PREFIX:-"ONPOLICY-SFT-Qwen3-4B-MATH-S1-BASE-V1"}
export INIT_MODEL_PATH=${INIT_MODEL_PATH:-"/data-1/.cache/huggingface/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539"}
export LOSS_MODE=${LOSS_MODE:-"wdl_sft"}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.0}
export LR=${LR:-5e-7}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-150}
export WANDB_PROJECT=${WANDB_PROJECT:-"OnPolicySFT-Then-WDLSFT-StagedV1"}
export WANDB_MODE=${WANDB_MODE:-offline}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export MIN_FREE_GB_FOR_CKPT=${MIN_FREE_GB_FOR_CKPT:-100}
export TRAIN_FILE=${TRAIN_FILE:-"/data-1/dataset/EnsembleLLM-data-processed/staged_v1/train_rl_format_boxed_prompt.parquet"}

# v1 On-Policy SFT: no old/current IS, no rollout IS correction.
export JOINT_TRAINING=${JOINT_TRAINING:-False}
export ROLLOUT_IS=${ROLLOUT_IS:-null}
export ROLLOUT_RS=${ROLLOUT_RS:-null}
export ROLLOUT_CALCULATE_LOG_PROBS=${ROLLOUT_CALCULATE_LOG_PROBS:-False}
export NORM_ADV_BY_STD_IN_GRPO=${NORM_ADV_BY_STD_IN_GRPO:-False}
export USE_KL_IN_REWARD=${USE_KL_IN_REWARD:-False}
export KL_COEF=${KL_COEF:-0.0}
export USE_KL_LOSS=${USE_KL_LOSS:-False}
export KL_LOSS_COEF=${KL_LOSS_COEF:-0.0}

# Dense online curve, but keep only latest + best checkpoints.
export TEST_FREQ=${TEST_FREQ:-5}
export SAVE_FREQ=${SAVE_FREQ:-5}
export VAL_N=${VAL_N:-3}
export MAX_ACTOR_CKPTS_TO_KEEP=${MAX_ACTOR_CKPTS_TO_KEEP:-1}
export MAX_CRITIC_CKPTS_TO_KEEP=${MAX_CRITIC_CKPTS_TO_KEEP:-1}
export KEEP_BEST_CKPT=${KEEP_BEST_CKPT:-True}
export BEST_CKPT_METRIC_KEY=${BEST_CKPT_METRIC_KEY:-"val-core/HuggingFaceH4/MATH-500/acc/mean@3"}
export BEST_CKPT_METRIC_MODE=${BEST_CKPT_METRIC_MODE:-max}
export BEST_CKPT_STRIP_OPTIMIZER=${BEST_CKPT_STRIP_OPTIMIZER:-True}
export DATA_SEED=${DATA_SEED:-20260528}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}

WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WRAPPER_SCRIPT_DIR
# shellcheck disable=SC1091
source "${WRAPPER_SCRIPT_DIR}/../ablation_single_model/_common_ablation.sh" \
    data.seed=${DATA_SEED} \
    data.train_max_samples=${TRAIN_MAX_SAMPLES} \
    "$@"
