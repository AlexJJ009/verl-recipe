#!/usr/bin/env bash
# Merge/extract as needed, then run Math-7 offline eval on a model2 checkpoint.
#
# Runs inside the verl-harness container with repo mounted at /workspace/verl.
# For Stage1 handoff models, set SOURCE_MODEL_PATH to an already-merged HF
# single-model directory. For Stage2 checkpoints, set FSDP_ACTOR_DIR and
# MERGED_JOINT_PATH; this script extracts model2 (sub_model_index=1).

set -xeuo pipefail

: "${MODEL_LABEL:?MODEL_LABEL is required}"
: "${OUTPUT_ROOT:=/data-1/model_weights/staged_v1/plateau_handoff_p60/math7_eval}"
: "${N_SAMPLES:=3}"
: "${TEMPERATURE:=1.0}"
: "${TOP_P:=0.95}"
: "${MAX_TOKENS:=4096}"
: "${TENSOR_PARALLEL:=8}"
: "${GPU_MEMORY_UTILIZATION:=0.85}"
: "${SEED:=42}"
: "${FORCE_EVAL:=0}"

mkdir -p "${OUTPUT_ROOT}"

if [ -n "${SOURCE_MODEL_PATH:-}" ]; then
    MODEL_PATH="${SOURCE_MODEL_PATH}"
    [ -f "${MODEL_PATH}/model.safetensors" ] || [ -f "${MODEL_PATH}/model.safetensors.index.json" ]
elif [ -n "${FSDP_ACTOR_DIR:-}" ] && [ -n "${MERGED_JOINT_PATH:-}" ]; then
    [ -d "${FSDP_ACTOR_DIR}" ]

    if [ -f "${MERGED_JOINT_PATH}/model.safetensors" ] || [ -f "${MERGED_JOINT_PATH}/model.safetensors.index.json" ]; then
        echo "Merged joint weights already exist at ${MERGED_JOINT_PATH}; skipping merge."
    else
        mkdir -p "$(dirname "${MERGED_JOINT_PATH}")"
        CUDA_VISIBLE_DEVICES="${MERGE_CUDA_VISIBLE_DEVICES:-0}" python -u -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "${FSDP_ACTOR_DIR}" \
            --target_dir "${MERGED_JOINT_PATH}" \
            --trust-remote-code
    fi

    MODEL_PATH="${MODEL2_PATH:-${MERGED_JOINT_PATH}_model2}"
    if [ -f "${MODEL_PATH}/model.safetensors" ] || [ -f "${MODEL_PATH}/model.safetensors.index.json" ]; then
        echo "model2 already extracted at ${MODEL_PATH}; skipping extraction."
    else
        python -u /workspace/verl/recipe/joint_training/extract_sub_model.py \
            --joint_model_path "${MERGED_JOINT_PATH}" \
            --output_path "${MODEL_PATH}" \
            --sub_model_index 1
    fi
else
    echo "ERROR: set SOURCE_MODEL_PATH, or set both FSDP_ACTOR_DIR and MERGED_JOINT_PATH." >&2
    exit 2
fi

OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_LABEL}/inference_math7_n${N_SAMPLES}_t${TEMPERATURE}_p${TOP_P}"
if [ -f "${OUTPUT_DIR}/eval_metrics.json" ] && [ "${FORCE_EVAL}" != "1" ]; then
    echo "eval_metrics.json already exists at ${OUTPUT_DIR}; skipping eval."
    exit 0
fi
mkdir -p "${OUTPUT_DIR}"

python -u /workspace/verl/recipe/joint_training/offline_eval.py \
    --model_path "${MODEL_PATH}" \
    --tensor_parallel "${TENSOR_PARALLEL}" \
    --n "${N_SAMPLES}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --max_tokens "${MAX_TOKENS}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --seed "${SEED}" \
    --output_dir "${OUTPUT_DIR}" \
    --test_files \
        /data-1/dataset/AIME-2025/aime-2025_with_system_prompt.parquet \
        /data-1/dataset/MATH-500/math500-test_with_system_prompt.parquet \
        /data-1/dataset/AMC23/amc23-test_with_system_prompt.parquet \
        /data-1/dataset/AQUA/aqua-test_with_system_prompt.parquet \
        /data-1/dataset/gsm8k/gsm8k-test_with_system_prompt.parquet \
        /data-1/dataset/MAWPS/mawps-test_with_system_prompt.parquet \
        /data-1/dataset/SVAMP/svamp-test_with_system_prompt.parquet

echo "Math-7 eval complete for ${MODEL_LABEL}. Results: ${OUTPUT_DIR}"
