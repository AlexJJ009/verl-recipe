#!/usr/bin/env bash
# ==============================================================================
# 4C-MATH-SFT: single-model current group-advantage IS algorithm on MATH train,
# initialized from Qwen3-4B-Base-SFT-stage-1.
# ==============================================================================
set -xeuo pipefail

DATA_ROOT=${DATA_ROOT:-/data-1}
export MATH_TRAIN_FILE=${MATH_TRAIN_FILE:-"${DATA_ROOT}/dataset/math/train_rl_format.parquet"}

if [ ! -f "$MATH_TRAIN_FILE" ]; then
    PREPARE_SCRIPT="${DATA_ROOT}/dataset/math/prepare_train_rl_format.py"
    RAW_MATH_TRAIN="${DATA_ROOT}/dataset/math/data/train-00000-of-00001-2a3b87ca709c844c.parquet"
    if [ -f "$PREPARE_SCRIPT" ] && [ -f "$RAW_MATH_TRAIN" ]; then
        python3 "$PREPARE_SCRIPT" --raw "$RAW_MATH_TRAIN" --output "$MATH_TRAIN_FILE"
    else
        echo "ERROR: MATH_TRAIN_FILE not found and local preparation inputs are missing: $MATH_TRAIN_FILE" >&2
        exit 1
    fi
fi

export TRAIN_FILE="$MATH_TRAIN_FILE"
export RUN_PREFIX=${RUN_PREFIX:-"WDL-GROUP-ADV-IS-Qwen3-4B-MATH-4C-MATHDATA-SFT-E1"}
export INIT_MODEL_PATH=${INIT_MODEL_PATH:-"${DATA_ROOT}/.cache/Qwen3-4B-Base-SFT-stage-1"}
export LOSS_MODE=${LOSS_MODE:-"wdl_group_adv_is"}
export LR=${LR:-5e-7}

# Raw MATH train has 7500 rows; max_prompt_length=500 filtering keeps 7405
# prompts, so drop_last=True gives floor(7405 / 64) = 115 steps per epoch.
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-115}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export VAL_N=${VAL_N:-3}
export BEST_CKPT_METRIC_KEY="val-core/HuggingFaceH4/MATH-500/acc/mean@3"

export NORM_ADV_BY_STD_IN_GRPO=${NORM_ADV_BY_STD_IN_GRPO:-false}
export CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
export CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.27}
export ALL_CORRECT_SFT_FALLBACK=${ALL_CORRECT_SFT_FALLBACK:-true}
export POS_SFT_FALLBACK_COEF=${POS_SFT_FALLBACK_COEF:-1.0}

WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WRAPPER_SCRIPT_DIR
# shellcheck disable=SC1091
source "${WRAPPER_SCRIPT_DIR}/_common_ablation.sh" "$@"
