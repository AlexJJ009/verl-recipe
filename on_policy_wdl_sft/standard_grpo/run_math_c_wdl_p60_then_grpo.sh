#!/usr/bin/env bash
set -euo pipefail

# Budget-matched continuation:
#   Stage1 P40 -> C-WDL P60 -> one fresh, uninterrupted 100-step GRPO run.
# It consumes the same post-Stage1 Stage2->Stage3 parquet used by C. The hard
# step cap, not an artificial P98 process boundary, defines the terminal step.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON="${SCRIPT_DIR}/run_qwen3_1p7b_standard_grpo.sh"

export TASK=math
export PIPELINE=c_wdl_p60_grpo
export LR=${LR:-1e-6}
export LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-0}
export RUN_PREFIX=${RUN_PREFIX:-MATH-QWEN3-1P7B-C-WDL-P60-THEN-GRPO}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-${RUN_PREFIX}_$(date +%s)}
export STAGE1_MODEL_PATH=${STAGE1_MODEL_PATH:-${INIT_MODEL_PATH:-}}
export INIT_MODEL_PATH=${INIT_MODEL_PATH:-${STAGE1_MODEL_PATH}}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
export PROTECTED_CKPT_STEPS=${PROTECTED_CKPT_STEPS:-'[20,40,60,80]'}
export RESUME_MODE=disable

if [[ -z "${INIT_MODEL_PATH}" ]]; then
  echo "ERROR: INIT_MODEL_PATH or STAGE1_MODEL_PATH must identify C-P60 Model2" >&2
  exit 2
fi

exec bash "${COMMON}" "$@"
