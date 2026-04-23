#!/usr/bin/env bash
# Offline eval: WDL-SFT M5.5 (EXP-13) step 300 — model1 (weak/anchor model, index=0)
# n=3, 7 benchmarks with system prompt
#
# Prerequisite: merged joint weights at /data-1/model_weights/WDL-SFT-4B-MATH-M5-5/step_300/
# (produced by run_eval_m5_5_step300.sh Step 1)
set -xeuo pipefail

MERGED_JOINT_PATH=/data-1/model_weights/WDL-SFT-4B-MATH-M5-5/step_300
MODEL1_PATH=/data-1/model_weights/WDL-SFT-4B-MATH-M5-5/step_300_model1
OUTPUT_DIR=/data-1/model_weights/WDL-SFT-4B-MATH-M5-5/step_300_model1/inference_n3

# Step 1: Extract model1 (index=0, anchor/weak model)
echo "=== Extracting model1 from merged joint checkpoint ==="
if [ -d "$MODEL1_PATH" ] && [ -f "$MODEL1_PATH/model.safetensors" ]; then
    echo "  model1 already extracted, skipping."
else
    python -u /workspace/verl/recipe/joint_training/extract_sub_model.py \
        --joint_model_path "$MERGED_JOINT_PATH" \
        --output_path "$MODEL1_PATH" \
        --sub_model_index 0
    echo "  model1 extracted to $MODEL1_PATH"
fi

# Step 2: Run vLLM offline eval on model1
echo "=== Running vLLM offline eval on model1 (n=3, tp=8) ==="
python -u /workspace/verl/recipe/joint_training/offline_eval.py \
    --model_path "$MODEL1_PATH" \
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

echo "=== model1 eval complete. Results at $OUTPUT_DIR ==="
