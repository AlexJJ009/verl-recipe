#!/usr/bin/env bash
# Host-side KodCode Instruct-2507 CTX8K P60 matched-beta Stage2 queue.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/run_code_task_kodcode_instruct2507_ctx8k_stage2_p60_queue.log}
export QUEUE_TMUX=${QUEUE_TMUX:-code_task_kodcode_i2507_ctx8k_s2_p60_queue}
export QUEUE_MODE=kodcode_instruct2507_ctx8k_stage2_p60
export START_INDEX=${START_INDEX:-0}
export END_INDEX=${END_INDEX:-1}
export QUEUE_CONTINUE_ON_FAILURE=${QUEUE_CONTINUE_ON_FAILURE:-0}
export QUEUE_STATUS_FILE=${QUEUE_STATUS_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/run_code_task_kodcode_instruct2507_ctx8k_stage2_p60_queue_status.tsv}
export MIN_FREE_GB=${MIN_FREE_GB:-300}
export MODEL2_ROOT=${MODEL2_ROOT:-/data-1/model_weights/code_task/kodcode_instruct2507_ctx8k_stage2_p60}
export STAGE2_TOTAL_TRAINING_STEPS=${STAGE2_TOTAL_TRAINING_STEPS:-40}
export STAGE2_TRAIN_BATCH_SIZE=${STAGE2_TRAIN_BATCH_SIZE:-64}
export STAGE2_BETA0_HANDOFF_STEP=${STAGE2_BETA0_HANDOFF_STEP:-60}
export STAGE2_BETA01_HANDOFF_STEP=${STAGE2_BETA01_HANDOFF_STEP:-60}
export STAGE2_BETA0_STAGE1_PREFIX=${STAGE2_BETA0_STAGE1_PREFIX:-ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA0-V1}
export STAGE2_BETA01_STAGE1_PREFIX=${STAGE2_BETA01_STAGE1_PREFIX:-ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA01-V1}
export STAGE2_BETA0_STAGE1_CKPT_DIR=${STAGE2_BETA0_STAGE1_CKPT_DIR:-/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA0-V1_1782371396}
export STAGE2_BETA01_STAGE1_CKPT_DIR=${STAGE2_BETA01_STAGE1_CKPT_DIR:-/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA01-V1_1782398871}
export STAGE2_BETA0_TRAIN_FILE=${STAGE2_BETA0_TRAIN_FILE:-/data-1/dataset/code/verl_rl/kodcode_stage2_after_s1_seed20260604_beta0_p60_handoff.parquet}
export STAGE2_BETA01_TRAIN_FILE=${STAGE2_BETA01_TRAIN_FILE:-/data-1/dataset/code/verl_rl/kodcode_stage2_after_s1_seed20260604_beta01_p60_handoff.parquet}
export STAGE2_TOKENIZER_MODEL_PATH=${STAGE2_TOKENIZER_MODEL_PATH:-/data-1/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}
export STAGE2_SOURCE_TRAIN_FILE=${STAGE2_SOURCE_TRAIN_FILE:-/data-1/dataset/code/verl_rl/kodcode_light_rl_10k_train_rl_format.parquet}
export SHARD_DOCKER_IMAGE=${SHARD_DOCKER_IMAGE:-${DOCKER_IMAGE:-verl-harness:latest}}

if [ "${DRY_RUN:-0}" != "1" ] && [ "${ALLOW_KODCODE_INSTRUCT2507_CTX8K_STAGE2_P60_TRAINING:-0}" != "1" ]; then
    echo "[kodcode instruct2507 ctx8k stage2 p60 queue] ERROR: non-dry-run requires explicit ALLOW_KODCODE_INSTRUCT2507_CTX8K_STAGE2_P60_TRAINING=1" >&2
    exit 1
fi
if [ "${ALLOW_KODCODE_INSTRUCT2507_CTX8K_STAGE2_P60_TRAINING:-0}" = "1" ]; then
    export ALLOW_G2_TRAINING_SMOKE=${ALLOW_G2_TRAINING_SMOKE:-1}
fi

SHARD_SCRIPT="${SCRIPT_DIR}/create_code_stage2_nonoverlap_shard.py"
SHARD_SCRIPT_CONTAINER="/workspace/verl/recipe/on_policy_wdl_sft/code_task/create_code_stage2_nonoverlap_shard.py"

run_shard_script() {
    shift
    if [ "${SHARD_IN_DOCKER:-1}" = "1" ]; then
        docker run --rm --ipc=host --network=host --shm-size=32g \
            -v /data-1:/data-1 \
            -v /data-1/verl07/verl:/workspace/verl \
            -w /workspace/verl \
            "$SHARD_DOCKER_IMAGE" \
            bash -lc 'python3 "$0" "$@"' "$SHARD_SCRIPT_CONTAINER" "$@"
    else
        python3 "$SHARD_SCRIPT" "$@"
    fi
}

create_or_verify_shard() {
    local label="$1" step="$2" output="$3"
    local args=(
        --source "$STAGE2_SOURCE_TRAIN_FILE"
        --output "$output"
        --model-path "$STAGE2_TOKENIZER_MODEL_PATH"
        --seed 20260604
        --stage1-steps "$step"
        --stage1-train-batch-size 64
        --stage2-steps "$STAGE2_TOTAL_TRAINING_STEPS"
        --stage2-train-batch-size "$STAGE2_TRAIN_BATCH_SIZE"
    )
    if [ -f "$output" ] && [ -f "$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).with_suffix(".manifest.json"))' "$output")" ]; then
        echo "[kodcode instruct2507 ctx8k stage2 p60 queue] verifying ${label} shard: ${output}"
        run_shard_script "$output" "${args[@]}" --verify-only
    else
        echo "[kodcode instruct2507 ctx8k stage2 p60 queue] creating ${label} shard: ${output}"
        run_shard_script "$output" "${args[@]}"
    fi
}

if [ "${CREATE_STAGE2_SHARDS:-1}" = "1" ]; then
    create_or_verify_shard "beta0-p60" "$STAGE2_BETA0_HANDOFF_STEP" "$STAGE2_BETA0_TRAIN_FILE"
    create_or_verify_shard "beta01-p60" "$STAGE2_BETA01_HANDOFF_STEP" "$STAGE2_BETA01_TRAIN_FILE"
fi

exec bash "${SCRIPT_DIR}/run_code_task_smoke_queue.sh" "$@"
