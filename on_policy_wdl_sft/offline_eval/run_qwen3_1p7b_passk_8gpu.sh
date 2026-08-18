#!/usr/bin/env bash
set -euo pipefail

# One Qwen3-1.7B replica per L40S. Each GPU generates one disjoint n=32
# sample shard, giving exact n=256 coverage without tensor-parallel overhead.
: "${TASK:?TASK must be math or code}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a gpu_ids <<< "${GPU_IDS}"
TOTAL_N=${TOTAL_N:-256}
N_PER_SHARD=${N_PER_SHARD:-32}
num_shards=$((TOTAL_N / N_PER_SHARD))
if [ -z "${MAX_NUM_SEQS:-}" ]; then
    if [ "${TASK}" = code ]; then
        MAX_NUM_SEQS=32
    else
        MAX_NUM_SEQS=64
    fi
fi

if (( TOTAL_N <= 0 || N_PER_SHARD <= 0 || TOTAL_N % N_PER_SHARD != 0 )); then
    echo "ERROR: TOTAL_N must be a positive multiple of N_PER_SHARD" >&2
    exit 2
fi
if (( ${#gpu_ids[@]} != num_shards )); then
    echo "ERROR: GPU_IDS count (${#gpu_ids[@]}) must equal shard count (${num_shards})" >&2
    exit 2
fi
if [ "${EVAL_CONFIG_ONLY:-0}" = 1 ]; then
    printf '%s\n' \
        "task=${TASK}" "gpu_ids=${GPU_IDS}" "tensor_parallel=1" \
        "total_n=${TOTAL_N}" "n_per_shard=${N_PER_SHARD}" "num_shards=${num_shards}" \
        "temperature=${TEMPERATURE:-0.6}" "top_p=${TOP_P:-0.95}" "top_k=${TOP_K:-20}" \
        "gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.90}" \
        "max_num_seqs=${MAX_NUM_SEQS}" \
        "max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-8192}" \
        "enforce_eager=${ENFORCE_EAGER:-false}"
    exit 0
fi
if [ -z "${TMUX:-}" ] && [ "${EVAL_SCHEDULER_MANAGED:-0}" != 1 ]; then
    echo "ERROR: 8-GPU offline evaluation must run inside tmux or an admitted scheduler-managed worker" >&2
    exit 2
fi

export TOTAL_N N_PER_SHARD
export TENSOR_PARALLEL=1
# Code evaluation imports VERL's tokenizer helper before vLLM starts its
# EngineCore. Importing VERL probes CUDA availability, so vLLM must not fork
# the already CUDA-initialized parent process.
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
export MAX_NUM_SEQS
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
export ENFORCE_EAGER=${ENFORCE_EAGER:-false}
mkdir -p "${OUTPUT_ROOT}/logs"

pids=()
for ((shard=0; shard<num_shards; shard++)); do
    gpu=${gpu_ids[$shard]}
    log=${OUTPUT_ROOT}/logs/shard_$(printf '%02d' "${shard}").log
    echo "[passk-8gpu] GPU ${gpu} -> shard ${shard}; log=${log}"
    CUDA_VISIBLE_DEVICES=${gpu} SHARD_INDEX=${shard} \
        bash "${SCRIPT_DIR}/run_qwen3_1p7b_passk_shard.sh" >"${log}" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
if (( status != 0 )); then
    echo "ERROR: at least one evaluation shard failed; inspect ${OUTPUT_ROOT}/logs" >&2
    exit 1
fi
echo "[passk-8gpu] all ${num_shards} shards complete; run the exact-coverage merge next"
