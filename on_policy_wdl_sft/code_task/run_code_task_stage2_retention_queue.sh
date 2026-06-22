#!/usr/bin/env bash
# Host-side matched-beta code-task Stage2 queue after Stage1 retention reruns.
set -euo pipefail

export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/run_code_task_stage2_retention_queue.log}
export QUEUE_TMUX=${QUEUE_TMUX:-code_task_stage2_retention_queue}
export QUEUE_MODE=stage2_retention
export START_INDEX=${START_INDEX:-0}
export END_INDEX=${END_INDEX:-1}
export QUEUE_CONTINUE_ON_FAILURE=${QUEUE_CONTINUE_ON_FAILURE:-0}
export QUEUE_STATUS_FILE=${QUEUE_STATUS_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/run_code_task_stage2_retention_queue_status.tsv}
export MIN_FREE_GB=${MIN_FREE_GB:-300}
export MODEL2_ROOT=${MODEL2_ROOT:-/data-1/model_weights/code_task}
export STAGE2_TOTAL_TRAINING_STEPS=${STAGE2_TOTAL_TRAINING_STEPS:-30}
export STAGE2_TRAIN_BATCH_SIZE=${STAGE2_TRAIN_BATCH_SIZE:-64}
export STAGE2_BETA0_HANDOFF_STEP=${STAGE2_BETA0_HANDOFF_STEP:-70}
export STAGE2_BETA01_HANDOFF_STEP=${STAGE2_BETA01_HANDOFF_STEP:-70}

: "${STAGE2_BETA0_HANDOFF_STEP:?set STAGE2_BETA0_HANDOFF_STEP from the selected beta=0.0 Stage1 curve}"
: "${STAGE2_BETA01_HANDOFF_STEP:?set STAGE2_BETA01_HANDOFF_STEP from the selected beta=0.1 Stage1 curve}"

if [ "${DRY_RUN:-0}" != "1" ] && [ "${ALLOW_CODE_STAGE2_RETENTION_TRAINING:-0}" != "1" ]; then
    echo "[code-task stage2 retention queue] ERROR: requires ALLOW_CODE_STAGE2_RETENTION_TRAINING=1" >&2
    exit 1
fi
if [ "${ALLOW_CODE_STAGE2_RETENTION_TRAINING:-0}" = "1" ]; then
    export ALLOW_G2_TRAINING_SMOKE=${ALLOW_G2_TRAINING_SMOKE:-1}
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARD_SCRIPT="${SCRIPT_DIR}/create_code_stage2_nonoverlap_shard.py"
export STAGE2_BETA0_TRAIN_FILE=${STAGE2_BETA0_TRAIN_FILE:-/data-1/dataset/code/verl_rl/kodcode_stage2_after_s1_seed20260604_beta0_p70_handoff.parquet}
export STAGE2_BETA01_TRAIN_FILE=${STAGE2_BETA01_TRAIN_FILE:-/data-1/dataset/code/verl_rl/kodcode_stage2_after_s1_seed20260604_beta01_p70_handoff.parquet}

create_or_verify_shard() {
    local label="$1" step="$2" output="$3"
    local args=(
        --output "$output"
        --seed 20260604
        --stage1-steps "$step"
        --stage1-train-batch-size 64
        --stage2-steps "$STAGE2_TOTAL_TRAINING_STEPS"
        --stage2-train-batch-size "$STAGE2_TRAIN_BATCH_SIZE"
    )
    if [ -f "$output" ] && [ -f "$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).with_suffix(".manifest.json"))' "$output")" ]; then
        echo "[code-task stage2 retention queue] verifying ${label} shard: ${output}"
        python3 "$SHARD_SCRIPT" "${args[@]}" --verify-only
    else
        echo "[code-task stage2 retention queue] creating ${label} shard: ${output}"
        python3 "$SHARD_SCRIPT" "${args[@]}"
    fi
}

if [ "${CREATE_STAGE2_SHARDS:-1}" = "1" ]; then
    create_or_verify_shard "beta0" "$STAGE2_BETA0_HANDOFF_STEP" "$STAGE2_BETA0_TRAIN_FILE"
    create_or_verify_shard "beta01" "$STAGE2_BETA01_HANDOFF_STEP" "$STAGE2_BETA01_TRAIN_FILE"
fi

exec bash "${SCRIPT_DIR}/run_code_task_smoke_queue.sh" "$@"
