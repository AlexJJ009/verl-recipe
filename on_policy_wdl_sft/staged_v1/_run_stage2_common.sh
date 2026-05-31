#!/usr/bin/env bash
# Common Stage 2 handoff: resolve Stage 1 model2, then run joint v1 WDL-SFT.

set -xeuo pipefail

: "${RUN_PREFIX:?RUN_PREFIX must be set by the caller}"
: "${WDL_SFT_BETA:?WDL_SFT_BETA must be set by the caller}"
: "${LR:?LR must be set by the caller}"

WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WRAPPER_SCRIPT_DIR

export LOSS_MODE=${LOSS_MODE:-wdl_sft}
export ROLLOUT_IS=${ROLLOUT_IS:-null}
export ROLLOUT_RS=${ROLLOUT_RS:-null}
export ROLLOUT_IS_THRESHOLD=${ROLLOUT_IS_THRESHOLD:-5.0}
export ROLLOUT_IS_BATCH_NORMALIZE=${ROLLOUT_IS_BATCH_NORMALIZE:-false}
export VAL_N=${VAL_N:-3}
export TEST_FREQ=${TEST_FREQ:-5}
export SAVE_FREQ=${SAVE_FREQ:-5}
export MAX_ACTOR_CKPTS_TO_KEEP=${MAX_ACTOR_CKPTS_TO_KEEP:-1}
export MAX_CRITIC_CKPTS_TO_KEEP=${MAX_CRITIC_CKPTS_TO_KEEP:-1}
export KEEP_BEST_CKPT=${KEEP_BEST_CKPT:-True}
export BEST_CKPT_METRIC_KEY=${BEST_CKPT_METRIC_KEY:-"val-core/HuggingFaceH4/MATH-500/acc/mean@3"}
export BEST_CKPT_METRIC_MODE=${BEST_CKPT_METRIC_MODE:-max}
export BEST_CKPT_STRIP_OPTIMIZER=${BEST_CKPT_STRIP_OPTIMIZER:-True}
export DATA_SEED=${DATA_SEED:-20260528}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-150}
export WANDB_PROJECT=${WANDB_PROJECT:-"OnPolicySFT-Then-WDLSFT-StagedV1"}
export WANDB_MODE=${WANDB_MODE:-offline}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}

# shellcheck disable=SC1091
source "${WRAPPER_SCRIPT_DIR}/_resolve_stage1_model2.sh"

# shellcheck disable=SC1091
source "${WRAPPER_SCRIPT_DIR}/../_common_wdl_sft_is_joint.sh" \
    data.seed=${DATA_SEED} \
    data.train_max_samples=${TRAIN_MAX_SAMPLES} \
    "$@"
