#!/usr/bin/env bash
set -euo pipefail

# Portable in-container entrypoint for one n-sample shard. Multiple hosts can
# run disjoint SHARD_INDEX values and merge them only after exact coverage is
# verified by merge_passk_shards.py.

: "${TASK:?TASK must be math or code}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

SHARD_INDEX=${SHARD_INDEX:-0}
N_PER_SHARD=${N_PER_SHARD:-32}
TOTAL_N=${TOTAL_N:-256}
BASE_SEED=${BASE_SEED:-20260811}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
if [ -z "${MAX_NUM_SEQS:-}" ]; then
    if [ "${TASK}" = code ]; then
        MAX_NUM_SEQS=32
    else
        MAX_NUM_SEQS=64
    fi
fi
ENFORCE_EAGER=${ENFORCE_EAGER:-false}
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-20}
MIN_P=${MIN_P:-0.0}
if [ -z "${MAX_TOKENS:-}" ]; then
    if [ "${TASK}" = code ]; then
        MAX_TOKENS=8192
    else
        MAX_TOKENS=4096
    fi
fi
ENABLE_THINKING=${ENABLE_THINKING:-true}

if (( N_PER_SHARD <= 0 || TOTAL_N <= 0 || TOTAL_N % N_PER_SHARD != 0 )); then
    echo "ERROR: TOTAL_N must be a positive multiple of N_PER_SHARD" >&2
    exit 2
fi
NUM_SHARDS=$((TOTAL_N / N_PER_SHARD))
if (( SHARD_INDEX < 0 || SHARD_INDEX >= NUM_SHARDS )); then
    echo "ERROR: SHARD_INDEX=${SHARD_INDEX} outside [0,$((NUM_SHARDS - 1))]" >&2
    exit 2
fi
SAMPLE_OFFSET=$((SHARD_INDEX * N_PER_SHARD))
SEED=$((BASE_SEED + SHARD_INDEX))
SHARD_DIR="${OUTPUT_ROOT}/shard_$(printf '%02d' "${SHARD_INDEX}")"
mkdir -p "${SHARD_DIR}"

echo "[passk-shard] task=${TASK} shard=${SHARD_INDEX}/${NUM_SHARDS} n=${N_PER_SHARD} offset=${SAMPLE_OFFSET} seed=${SEED}"
echo "[passk-shard] model=${MODEL_PATH} output=${SHARD_DIR}"
echo "[passk-shard] thinking=${ENABLE_THINKING} temperature=${TEMPERATURE} top_p=${TOP_P} top_k=${TOP_K} min_p=${MIN_P}"
echo "[passk-shard] engine=tp${TENSOR_PARALLEL} mem=${GPU_MEMORY_UTILIZATION} max_num_seqs=${MAX_NUM_SEQS} max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} enforce_eager=${ENFORCE_EAGER}"

eager_args=()
if [ "${ENFORCE_EAGER}" = true ]; then
    eager_args+=(--enforce-eager)
fi

case "${TASK}" in
    math)
        : "${MATH_TEST_FILES:?MATH_TEST_FILES is required as a whitespace-separated path list}"
        # shellcheck disable=SC2206
        test_files=( ${MATH_TEST_FILES} )
        python -u recipe/joint_training/offline_eval.py \
            --model_path "${MODEL_PATH}" \
            --tensor_parallel "${TENSOR_PARALLEL}" \
            --n "${N_PER_SHARD}" \
            --sample-offset "${SAMPLE_OFFSET}" \
            --seed "${SEED}" \
            --temperature "${TEMPERATURE}" \
            --top_p "${TOP_P}" \
            --top_k "${TOP_K}" \
            --min_p "${MIN_P}" \
            --max_tokens "${MAX_TOKENS}" \
            --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
            --max-num-seqs "${MAX_NUM_SEQS}" \
            --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
            "${eager_args[@]}" \
            --enable-thinking "${ENABLE_THINKING}" \
            --require-explicit-thinking \
            --output_dir "${SHARD_DIR}" \
            --test_files "${test_files[@]}"
        ;;
    code)
        : "${VALIDATION_PARQUET:?VALIDATION_PARQUET is required for code}"
        python -u recipe/on_policy_wdl_sft/code_task/eval_code_vllm.py \
            --model "${MODEL_PATH}" \
            --validation-parquet "${VALIDATION_PARQUET}" \
            --output "${SHARD_DIR}/raw_generations.jsonl" \
            --summary "${SHARD_DIR}/generation_summary.json" \
            --tensor-parallel "${TENSOR_PARALLEL}" \
            --n "${N_PER_SHARD}" \
            --sample-offset "${SAMPLE_OFFSET}" \
            --seed "${SEED}" \
            --temperature "${TEMPERATURE}" \
            --top-p "${TOP_P}" \
            --top-k "${TOP_K}" \
            --min-p "${MIN_P}" \
            --max-tokens "${MAX_TOKENS}" \
            --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
            --max-num-seqs "${MAX_NUM_SEQS}" \
            --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
            "${eager_args[@]}" \
            --enable-thinking "${ENABLE_THINKING}" \
            --require-explicit-thinking
        ;;
    *)
        echo "ERROR: unsupported TASK=${TASK}" >&2
        exit 2
        ;;
esac

echo "[passk-shard] complete: ${SHARD_DIR}"
