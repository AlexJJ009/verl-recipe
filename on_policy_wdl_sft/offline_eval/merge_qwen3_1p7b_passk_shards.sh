#!/usr/bin/env bash
set -euo pipefail

: "${TASK:?TASK must be math or code}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

TOTAL_N=${TOTAL_N:-256}
N_PER_SHARD=${N_PER_SHARD:-32}
if (( TOTAL_N % N_PER_SHARD != 0 )); then
    echo "ERROR: TOTAL_N must be divisible by N_PER_SHARD" >&2
    exit 2
fi
NUM_SHARDS=$((TOTAL_N / N_PER_SHARD))
inputs=()
summaries=()
for ((index=0; index<NUM_SHARDS; index++)); do
    shard="${OUTPUT_ROOT}/shard_$(printf '%02d' "${index}")"
    if [ "${TASK}" = math ]; then
        inputs+=("${shard}/eval_details.parquet")
        summaries+=("${shard}/eval_metrics.json")
    else
        inputs+=("${shard}/raw_generations.jsonl")
        summaries+=("${shard}/generation_summary.json")
    fi
done

mkdir -p "${OUTPUT_ROOT}/merged"
if [ "${TASK}" = math ]; then
    output="${OUTPUT_ROOT}/merged/eval_details.parquet"
else
    output="${OUTPUT_ROOT}/merged/raw_generations.jsonl"
fi

python -u recipe/joint_training/merge_passk_shards.py \
    --kind "${TASK}" \
    --input "${inputs[@]}" \
    --contract-summary "${summaries[@]}" \
    --output "${output}" \
    --expected-n "${TOTAL_N}" \
    --summary "${OUTPUT_ROOT}/merged/merge_summary.json"

echo "[passk-merge] complete: ${OUTPUT_ROOT}/merged/merge_summary.json"
