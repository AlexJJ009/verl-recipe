#!/usr/bin/env bash
# WDL group-advantage IS 1D: same validation protocol as 1A, but train on Hendrycks MATH train.
set -xeuo pipefail

WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WRAPPER_SCRIPT_DIR

DATA_ROOT=${DATA_ROOT:-/data-1}
export TRAIN_FILE=${TRAIN_FILE:-"${DATA_ROOT}/dataset/math/train_rl_format.parquet"}

if [ ! -f "$TRAIN_FILE" ]; then
    python3 "${DATA_ROOT}/dataset/math/prepare_train_rl_format.py" \
        --raw "${DATA_ROOT}/dataset/math/data/train-00000-of-00001-2a3b87ca709c844c.parquet" \
        --output "$TRAIN_FILE"
fi

export RUN_PREFIX=${RUN_PREFIX:-"WDL-GROUP-ADV-IS-Qwen3-4B-MATH-1D-MATHDATA-E1"}
export LR=${LR:-5e-7}
export LOSS_MODE=${LOSS_MODE:-wdl_group_adv_is}

# The raw MATH train has 7500 rows. With max_prompt_length=500 the current
# tokenizer/filtering path keeps 7405 prompts, and drop_last=True gives
# floor(7405 / 64) = 115 optimizer steps per epoch.
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-115}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}

# Keep validation standards identical to 1A by inheriting TEST_FILES from _common_group_adv_is.sh.
# shellcheck disable=SC1091
source "${WRAPPER_SCRIPT_DIR}/_common_group_adv_is.sh" "$@"
