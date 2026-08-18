#!/usr/bin/env bash
# Qwen3-1.7B KodCode cold-start fraction Stage2 P40 queue.
# Matrix: fraction {25%,50%} x KL {off, model2-only} at fusion_lambda=0.8 by default.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/run_code_task_kodcode_qwen3_1p7b_coldstart_fraction_stage2_p40_m2kl_vs_nokl_queue.log}
export QUEUE_TMUX=${QUEUE_TMUX:-code_task_kodcode_qwen3_1p7b_coldstart_fraction_s2_p40_m2kl_vs_nokl_queue}
export QUEUE_MODE=kodcode_qwen3_1p7b_coldstart_fraction_stage2_p40_m2kl_vs_nokl
export QUEUE_STATUS_FILE=${QUEUE_STATUS_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/run_code_task_kodcode_qwen3_1p7b_coldstart_fraction_stage2_p40_m2kl_vs_nokl_status.tsv}

export STAGE2_TOTAL_TRAINING_STEPS=${STAGE2_TOTAL_TRAINING_STEPS:-60}
export STAGE2_FRAC25_BETA01_P40_HANDOFF_STEP=${STAGE2_FRAC25_BETA01_P40_HANDOFF_STEP:-40}
export STAGE2_FRAC50_BETA01_P40_HANDOFF_STEP=${STAGE2_FRAC50_BETA01_P40_HANDOFF_STEP:-40}
export STAGE2_FRAC25_BETA01_STAGE1_PREFIX=${STAGE2_FRAC25_BETA01_STAGE1_PREFIX:-ONPOLICY-SFT-Qwen3-1P7B-COLDSTART-FRAC25-COT-V3-CODE-KODCODE-CTX8K-S1-BETA01-V1}
export STAGE2_FRAC50_BETA01_STAGE1_PREFIX=${STAGE2_FRAC50_BETA01_STAGE1_PREFIX:-ONPOLICY-SFT-Qwen3-1P7B-COLDSTART-FRAC50-COT-V3-CODE-KODCODE-CTX8K-S1-BETA01-V1}
export STAGE2_FRAC25_BETA01_TRAIN_FILE=${STAGE2_FRAC25_BETA01_TRAIN_FILE:-/data-1/dataset/code/verl_rl/kodcode_stage2_after_s1_seed20260604_qwen3_1p7b_coldstart_frac25_beta01_p40_handoff_s2steps60.parquet}
export STAGE2_FRAC50_BETA01_TRAIN_FILE=${STAGE2_FRAC50_BETA01_TRAIN_FILE:-/data-1/dataset/code/verl_rl/kodcode_stage2_after_s1_seed20260604_qwen3_1p7b_coldstart_frac50_beta01_p40_handoff_s2steps60.parquet}
export COLDSTART_FRACTION_STAGE2_FUSION_LAMBDAS=${COLDSTART_FRACTION_STAGE2_FUSION_LAMBDAS:-0.8}

export START_INDEX=${START_INDEX:-0}
export END_INDEX=${END_INDEX:-3}
export MIN_FREE_GB=${MIN_FREE_GB:-300}
export QUEUE_CONTINUE_ON_FAILURE=${QUEUE_CONTINUE_ON_FAILURE:-0}
export ALLOW_RESUME=${ALLOW_RESUME:-0}

export QWEN3_1P7B_MODEL_PATH=${QWEN3_1P7B_MODEL_PATH:-/data-1/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
export BASE_MODEL_PATH=${BASE_MODEL_PATH:-$QWEN3_1P7B_MODEL_PATH}
export STAGE2_TOKENIZER_MODEL_PATH=${STAGE2_TOKENIZER_MODEL_PATH:-$QWEN3_1P7B_MODEL_PATH}
export STAGE2_SOURCE_TRAIN_FILE=${STAGE2_SOURCE_TRAIN_FILE:-/data-1/dataset/code/verl_rl/kodcode_light_rl_10k_train_rl_format.parquet}
export STAGE2_TRAIN_BATCH_SIZE=${STAGE2_TRAIN_BATCH_SIZE:-64}
export SHARD_DOCKER_IMAGE=${SHARD_DOCKER_IMAGE:-${DOCKER_IMAGE:-verl-harness:latest}}
export MODEL2_ROOT=${MODEL2_ROOT:-/data-1/model_weights/code_task/kodcode_qwen3_1p7b_coldstart_fraction_cot_v3_ctx8k_stage2_p40}

export SUBMODEL_KL_MODEL2_COEF_DEFAULT=${SUBMODEL_KL_MODEL2_COEF_DEFAULT:-0.01}
export MAX_ACTOR_CKPTS_TO_KEEP=${MAX_ACTOR_CKPTS_TO_KEEP:-1}
export MAX_CRITIC_CKPTS_TO_KEEP=${MAX_CRITIC_CKPTS_TO_KEEP:-1}
export KEEP_BEST_CKPT=${KEEP_BEST_CKPT:-True}

if [ "${DRY_RUN:-0}" != "1" ] && [ "${ALLOW_KODCODE_QWEN3_1P7B_COLDSTART_FRACTION_STAGE2_P40_TRAINING:-0}" != "1" ]; then
    echo "[qwen3 1p7b coldstart fraction stage2 p40 queue] ERROR: non-dry-run requires ALLOW_KODCODE_QWEN3_1P7B_COLDSTART_FRACTION_STAGE2_P40_TRAINING=1" >&2
    exit 1
fi
if [ "${ALLOW_KODCODE_QWEN3_1P7B_COLDSTART_FRACTION_STAGE2_P40_TRAINING:-0}" = "1" ]; then
    export ALLOW_G2_TRAINING_SMOKE=${ALLOW_G2_TRAINING_SMOKE:-1}
fi

SHARD_SCRIPT="${SCRIPT_DIR}/create_code_stage2_nonoverlap_shard.py"
SHARD_SCRIPT_CONTAINER="/workspace/verl/recipe/on_policy_wdl_sft/code_task/create_code_stage2_nonoverlap_shard.py"

run_shard_script() {
    shift
    if [ "${SHARD_IN_DOCKER:-1}" = "1" ]; then
        docker run --rm --ipc=host --network=host --shm-size=32g \
            -v /data-1:/data-1 \
            -v /data-2:/data-2 \
            -v /data-1/verl07/verl:/workspace/verl \
            -w /workspace/verl \
            "$SHARD_DOCKER_IMAGE" \
            bash -lc 'python3 "$0" "$@"' "$SHARD_SCRIPT_CONTAINER" "$@"
    else
        python3 "$SHARD_SCRIPT" "$@"
    fi
}

manifest_path() {
    python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).with_suffix(".manifest.json"))' "$1"
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
    if [ -f "$output" ] && [ -f "$(manifest_path "$output")" ]; then
        echo "[qwen3 1p7b coldstart fraction stage2 queue] verifying ${label} shard: ${output}"
        run_shard_script "$output" "${args[@]}" --verify-only
    else
        echo "[qwen3 1p7b coldstart fraction stage2 queue] creating ${label} shard: ${output}"
        run_shard_script "$output" "${args[@]}"
    fi
}

CREATE_STAGE2_SHARDS=${CREATE_STAGE2_SHARDS:-1}
if [ "${DRY_RUN:-0}" = "1" ] && [ "${CREATE_STAGE2_SHARDS_EXPLICIT:-0}" != "1" ]; then
    CREATE_STAGE2_SHARDS=0
fi
if [ "$CREATE_STAGE2_SHARDS" = "1" ]; then
    create_or_verify_shard "qwen3-1p7b-coldstart-frac25-beta01-p40" "$STAGE2_FRAC25_BETA01_P40_HANDOFF_STEP" "$STAGE2_FRAC25_BETA01_TRAIN_FILE"
    create_or_verify_shard "qwen3-1p7b-coldstart-frac50-beta01-p40" "$STAGE2_FRAC50_BETA01_P40_HANDOFF_STEP" "$STAGE2_FRAC50_BETA01_TRAIN_FILE"
else
    echo "[qwen3 1p7b coldstart fraction stage2 queue] skipping shard creation/verification CREATE_STAGE2_SHARDS=${CREATE_STAGE2_SHARDS}"
fi

exec bash "${SCRIPT_DIR}/run_code_task_smoke_queue.sh" "$@"
