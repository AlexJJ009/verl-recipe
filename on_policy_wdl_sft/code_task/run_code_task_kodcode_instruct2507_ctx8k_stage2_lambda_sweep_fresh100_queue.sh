#!/usr/bin/env bash
# Fresh KodCode Instruct-2507 CTX8K P40 Stage2 fusion_lambda sweep.
# Starts every item from Stage1 step40 and trains 60 Stage2 steps
# (effective step100) under new prefixes to avoid resume contamination.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/run_code_task_kodcode_instruct2507_ctx8k_stage2_lambda_sweep_fresh100_queue.log}
export QUEUE_TMUX=${QUEUE_TMUX:-code_task_kodcode_i2507_ctx8k_s2_lambda_sweep_fresh100_queue}
export QUEUE_STATUS_FILE=${QUEUE_STATUS_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/run_code_task_kodcode_instruct2507_ctx8k_stage2_lambda_sweep_fresh100_status.tsv}

export STAGE2_P40_BETA01_LAMBDA06_PREFIX=${STAGE2_P40_BETA01_LAMBDA06_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-LAMBDA06-FRESH100-V1}
export STAGE2_P40_BETA01_LAMBDA07_PREFIX=${STAGE2_P40_BETA01_LAMBDA07_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-LAMBDA07-FRESH100-V1}
export STAGE2_P40_BETA01_LAMBDA08_PREFIX=${STAGE2_P40_BETA01_LAMBDA08_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-LAMBDA08-FRESH100-V1}
export STAGE2_P40_BETA01_LAMBDA09_PREFIX=${STAGE2_P40_BETA01_LAMBDA09_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-LAMBDA09-FRESH100-V1}

export STAGE2_TOTAL_TRAINING_STEPS=${STAGE2_TOTAL_TRAINING_STEPS:-60}
export STAGE2_BETA01_P40_HANDOFF_STEP=${STAGE2_BETA01_P40_HANDOFF_STEP:-40}
export START_INDEX=${START_INDEX:-0}
export END_INDEX=${END_INDEX:-3}
export MIN_FREE_GB=${MIN_FREE_GB:-300}
export QUEUE_CONTINUE_ON_FAILURE=${QUEUE_CONTINUE_ON_FAILURE:-0}
export ALLOW_RESUME=${ALLOW_RESUME:-0}

exec bash "${SCRIPT_DIR}/run_code_task_kodcode_instruct2507_ctx8k_stage2_lambda_sweep_queue.sh" "$@"
