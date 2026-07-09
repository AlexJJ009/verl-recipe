#!/usr/bin/env bash
# Monitor for the fresh P40 -> effective100 Stage2 lambda sweep.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export QUEUE_MODE=kodcode_instruct2507_ctx8k_stage2_lambda_sweep
export MONITOR_NAME=${MONITOR_NAME:-code_task_kodcode_instruct2507_ctx8k_stage2_lambda_sweep_fresh100}
export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/monitor_code_task_kodcode_instruct2507_ctx8k_stage2_lambda_sweep_fresh100_notify.log}
export QUEUE_TMUX=${QUEUE_TMUX:-code_task_kodcode_i2507_ctx8k_s2_lambda_sweep_fresh100_queue}
export STAGE2_TOTAL_TRAINING_STEPS=${STAGE2_TOTAL_TRAINING_STEPS:-60}

export STAGE2_P40_BETA01_LAMBDA06_PREFIX=${STAGE2_P40_BETA01_LAMBDA06_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-LAMBDA06-FRESH100-V1}
export STAGE2_P40_BETA01_LAMBDA07_PREFIX=${STAGE2_P40_BETA01_LAMBDA07_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-LAMBDA07-FRESH100-V1}
export STAGE2_P40_BETA01_LAMBDA08_PREFIX=${STAGE2_P40_BETA01_LAMBDA08_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-LAMBDA08-FRESH100-V1}
export STAGE2_P40_BETA01_LAMBDA09_PREFIX=${STAGE2_P40_BETA01_LAMBDA09_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-LAMBDA09-FRESH100-V1}

exec bash "${SCRIPT_DIR}/monitor_code_task_queue_notify.sh" "$@"
