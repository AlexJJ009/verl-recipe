#!/usr/bin/env bash
# Offline eval: WDL-SFT 1C (EXP-18) step 300 — model2 (strong/trainable, index=1)
# n=3, 7 benchmarks with system prompt
set -xeuo pipefail

FSDP_ACTOR_DIR=/data-1/checkpoints/WDL-SFT-Qwen3-4B-MATH-1C_1776768784/global_step_300/actor
MERGED_JOINT_PATH=/data-1/model_weights/WDL-SFT-4B-MATH-1C/step_300
MODEL2_PATH=/data-1/model_weights/WDL-SFT-4B-MATH-1C/step_300_model2
OUTPUT_DIR=/data-1/model_weights/WDL-SFT-4B-MATH-1C/step_300_model2/inference_n3

echo "=== Step 1: Merging FSDP checkpoint ==="
if [ -d "$MERGED_JOINT_PATH" ] && [ -f "$MERGED_JOINT_PATH/model.safetensors.index.json" ]; then
    echo "  Merged weights already exist, skipping merge."
else
    CUDA_VISIBLE_DEVICES=0 python -u -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$FSDP_ACTOR_DIR" \
        --target_dir "$MERGED_JOINT_PATH" \
        --trust-remote-code
    echo "  Merged to $MERGED_JOINT_PATH"
fi

echo "=== Step 2: Extracting model2 (index=1, strong/trainable) ==="
if [ -d "$MODEL2_PATH" ] && [ -f "$MODEL2_PATH/model.safetensors" ]; then
    echo "  model2 already extracted, skipping."
else
    python -u /workspace/verl/recipe/joint_training/extract_sub_model.py \
        --joint_model_path "$MERGED_JOINT_PATH" \
        --output_path "$MODEL2_PATH" \
        --sub_model_index 1
    echo "  model2 extracted to $MODEL2_PATH"
fi

echo "=== Step 3: Running vLLM offline eval on model2 (n=3, tp=8) ==="
python -u /workspace/verl/recipe/joint_training/offline_eval.py \
    --model_path "$MODEL2_PATH" \
    --tensor_parallel 8 \
    --n 3 \
    --temperature 1.0 \
    --top_p 0.95 \
    --max_tokens 4096 \
    --output_dir "$OUTPUT_DIR" \
    --test_files \
        /data-1/dataset/AIME-2025/aime-2025_with_system_prompt.parquet \
        /data-1/dataset/MATH-500/math500-test_with_system_prompt.parquet \
        /data-1/dataset/AMC23/amc23-test_with_system_prompt.parquet \
        /data-1/dataset/AQUA/aqua-test_with_system_prompt.parquet \
        /data-1/dataset/gsm8k/gsm8k-test_with_system_prompt.parquet \
        /data-1/dataset/MAWPS/mawps-test_with_system_prompt.parquet \
        /data-1/dataset/SVAMP/svamp-test_with_system_prompt.parquet

echo "=== model2 eval complete. Results at $OUTPUT_DIR ==="
