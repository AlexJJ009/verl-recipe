#!/usr/bin/env bash
# Parameterized merge + offline-eval for non-joint single-model ablation runs
# (MINIRL/WDL-SFT ablation_single_model family, e.g. 2Z-SFT, 2A-SFT, 2B-SFT, 2C-SFT).
#
# Unlike the joint-model path (recipe/on_policy_wdl_sft/run_eval_*_model{1,2}.sh),
# the ablation runs train a standard Qwen3 backbone directly — no sub-model
# extraction. Merge FSDP → HF, then eval the merged dir in-place.
#
# Required env vars (caller sets these):
#   RUN_PREFIX   e.g. MINIRL-Qwen3-4B-MATH-2Z-SFT
#   TIMESTAMP    e.g. 1776855436
#   STEP         e.g. 275
#   N_SAMPLES    e.g. 3
#
# Derived paths:
#   FSDP_ACTOR_DIR=/data-1/checkpoints/${RUN_PREFIX}_${TIMESTAMP}/global_step_${STEP}/actor
#   MERGED_DIR=/data-1/model_weights/${RUN_PREFIX}/step_${STEP}
#   OUTPUT_DIR=${MERGED_DIR}/inference_n${N_SAMPLES}
#
# Assumes it runs inside the verl-harness docker container with /data-1 mounted
# and working dir /workspace/verl.

set -xeuo pipefail

: "${RUN_PREFIX:?RUN_PREFIX is required}"
: "${TIMESTAMP:?TIMESTAMP is required}"
: "${STEP:?STEP is required}"
: "${N_SAMPLES:=3}"

FSDP_ACTOR_DIR="/data-1/checkpoints/${RUN_PREFIX}_${TIMESTAMP}/global_step_${STEP}/actor"
MERGED_DIR="/data-1/model_weights/${RUN_PREFIX}/step_${STEP}"
OUTPUT_DIR="${MERGED_DIR}/inference_n${N_SAMPLES}"

echo "=== [${RUN_PREFIX} step_${STEP}] Phase 1: Merge FSDP → HF ==="
if [ -f "${MERGED_DIR}/model.safetensors" ] || [ -f "${MERGED_DIR}/model.safetensors.index.json" ]; then
    echo "  Merged weights already present at ${MERGED_DIR}, skipping merge."
else
    mkdir -p "$(dirname "${MERGED_DIR}")"
    CUDA_VISIBLE_DEVICES=0 python -u -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${FSDP_ACTOR_DIR}" \
        --target_dir "${MERGED_DIR}" \
        --trust-remote-code
    echo "  Merged → ${MERGED_DIR}"
fi

echo "=== [${RUN_PREFIX} step_${STEP}] Phase 2: vLLM offline eval (n=${N_SAMPLES}, tp=8) ==="
if [ -f "${OUTPUT_DIR}/eval_metrics.json" ]; then
    echo "  eval_metrics.json already exists at ${OUTPUT_DIR}, skipping eval."
    echo "  (delete it manually if you want to rerun.)"
    exit 0
fi
mkdir -p "${OUTPUT_DIR}"

python -u /workspace/verl/recipe/joint_training/offline_eval.py \
    --model_path "${MERGED_DIR}" \
    --tensor_parallel 8 \
    --n "${N_SAMPLES}" \
    --temperature 1.0 \
    --top_p 0.95 \
    --max_tokens 4096 \
    --output_dir "${OUTPUT_DIR}" \
    --test_files \
        /data-1/dataset/AIME-2025/aime-2025_with_system_prompt.parquet \
        /data-1/dataset/MATH-500/math500-test_with_system_prompt.parquet \
        /data-1/dataset/AMC23/amc23-test_with_system_prompt.parquet \
        /data-1/dataset/AQUA/aqua-test_with_system_prompt.parquet \
        /data-1/dataset/gsm8k/gsm8k-test_with_system_prompt.parquet \
        /data-1/dataset/MAWPS/mawps-test_with_system_prompt.parquet \
        /data-1/dataset/SVAMP/svamp-test_with_system_prompt.parquet

echo "=== [${RUN_PREFIX} step_${STEP}] eval complete. Results at ${OUTPUT_DIR} ==="
