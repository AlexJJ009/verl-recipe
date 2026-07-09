#!/usr/bin/env bash
# Formal KodCode Instruct-2507 CTX8K P40 Stage2 submodel-KL lambda sweep.
# Starts every item from Stage1 step40 and trains 60 Stage2 steps
# (effective step100). KL type is low_var_kl and coef defaults to 0.01.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/run_code_task_kodcode_instruct2507_ctx8k_stage2_p40_subkl_lambda_sweep_queue.log}
export QUEUE_TMUX=${QUEUE_TMUX:-code_task_kodcode_i2507_ctx8k_s2_p40_subkl_lambda_queue}
export QUEUE_MODE=kodcode_instruct2507_ctx8k_stage2_p40_subkl_lambda_sweep
export QUEUE_STATUS_FILE=${QUEUE_STATUS_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/run_code_task_kodcode_instruct2507_ctx8k_stage2_p40_subkl_lambda_sweep_status.tsv}

export STAGE2_TOTAL_TRAINING_STEPS=${STAGE2_TOTAL_TRAINING_STEPS:-60}
export STAGE2_BETA01_P40_HANDOFF_STEP=${STAGE2_BETA01_P40_HANDOFF_STEP:-40}
export STAGE2_BETA01_P40_TRAIN_FILE=${STAGE2_BETA01_P40_TRAIN_FILE:-/data-1/dataset/code/verl_rl/kodcode_stage2_after_s1_seed20260604_beta01_p40_handoff_s2steps60.parquet}
export START_INDEX=${START_INDEX:-0}
export END_INDEX=${END_INDEX:-14}
export MIN_FREE_GB=${MIN_FREE_GB:-300}
export QUEUE_CONTINUE_ON_FAILURE=${QUEUE_CONTINUE_ON_FAILURE:-0}
export ALLOW_RESUME=${ALLOW_RESUME:-0}

export SUBMODEL_KL_MODEL1_COEF_DEFAULT=${SUBMODEL_KL_MODEL1_COEF_DEFAULT:-0.01}
export SUBMODEL_KL_MODEL2_COEF_DEFAULT=${SUBMODEL_KL_MODEL2_COEF_DEFAULT:-0.01}
export MAX_ACTOR_CKPTS_TO_KEEP=${MAX_ACTOR_CKPTS_TO_KEEP:-1}
export MAX_CRITIC_CKPTS_TO_KEEP=${MAX_CRITIC_CKPTS_TO_KEEP:-1}
export KEEP_BEST_CKPT=${KEEP_BEST_CKPT:-True}

if [ "${DRY_RUN:-0}" != "1" ] && [ "${ALLOW_KODCODE_INSTRUCT2507_CTX8K_STAGE2_P40_SUBKL_LAMBDA_SWEEP_TRAINING:-0}" != "1" ]; then
    echo "[kodcode instruct2507 ctx8k stage2 p40 subkl lambda sweep queue] ERROR: non-dry-run requires explicit ALLOW_KODCODE_INSTRUCT2507_CTX8K_STAGE2_P40_SUBKL_LAMBDA_SWEEP_TRAINING=1" >&2
    exit 1
fi
if [ "${ALLOW_KODCODE_INSTRUCT2507_CTX8K_STAGE2_P40_SUBKL_LAMBDA_SWEEP_TRAINING:-0}" = "1" ]; then
    export ALLOW_G2_TRAINING_SMOKE=${ALLOW_G2_TRAINING_SMOKE:-1}
fi

export STAGE2_TRAIN_BATCH_SIZE=${STAGE2_TRAIN_BATCH_SIZE:-64}
export STAGE2_TOKENIZER_MODEL_PATH=${STAGE2_TOKENIZER_MODEL_PATH:-/data-1/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}
export STAGE2_SOURCE_TRAIN_FILE=${STAGE2_SOURCE_TRAIN_FILE:-/data-1/dataset/code/verl_rl/kodcode_light_rl_10k_train_rl_format.parquet}
export SHARD_DOCKER_IMAGE=${SHARD_DOCKER_IMAGE:-${DOCKER_IMAGE:-verl-harness:latest}}

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

if [ "${CREATE_STAGE2_SHARDS:-1}" = "1" ]; then
    shard_args=(
        --source "$STAGE2_SOURCE_TRAIN_FILE"
        --output "$STAGE2_BETA01_P40_TRAIN_FILE"
        --model-path "$STAGE2_TOKENIZER_MODEL_PATH"
        --seed 20260604
        --stage1-steps "$STAGE2_BETA01_P40_HANDOFF_STEP"
        --stage1-train-batch-size 64
        --stage2-steps "$STAGE2_TOTAL_TRAINING_STEPS"
        --stage2-train-batch-size "$STAGE2_TRAIN_BATCH_SIZE"
    )
    if [ -f "$STAGE2_BETA01_P40_TRAIN_FILE" ] && [ -f "$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).with_suffix(".manifest.json"))' "$STAGE2_BETA01_P40_TRAIN_FILE")" ]; then
        echo "[kodcode instruct2507 ctx8k stage2 p40 subkl lambda sweep queue] verifying beta01-p40 shard: ${STAGE2_BETA01_P40_TRAIN_FILE}"
        run_shard_script "$STAGE2_BETA01_P40_TRAIN_FILE" "${shard_args[@]}" --verify-only
    else
        echo "[kodcode instruct2507 ctx8k stage2 p40 subkl lambda sweep queue] creating beta01-p40 shard: ${STAGE2_BETA01_P40_TRAIN_FILE}"
        run_shard_script "$STAGE2_BETA01_P40_TRAIN_FILE" "${shard_args[@]}"
    fi
fi

exec bash "${SCRIPT_DIR}/run_code_task_smoke_queue.sh" "$@"
