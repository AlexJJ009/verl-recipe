#!/usr/bin/env bash
# Serial driver: merge + eval 2A-SFT step_275 then step_300.
# Run inside verl-harness Docker container with /data-1 mounted.
# Total ~40-60 min on L40S tp=8 (single-model, same path as 2Z-SFT).
set -euo pipefail

cd /workspace/verl/recipe/on_policy_wdl_sft/ablation_single_model/eval
LOG_DIR=/workspace/verl/recipe/on_policy_wdl_sft/ablation_single_model/eval

log_section() {
    echo ""
    echo "==================================================================="
    echo "[ABL-MINIRL-02-EVAL-MARKER] [$(date -Iseconds)] $*"
    echo "==================================================================="
}

log_section "2A-SFT step 275 — merge + eval"
bash run_eval_2a_sft_step275.sh 2>&1 | tee "${LOG_DIR}/eval_2a_sft_step275.log"

log_section "2A-SFT step 300 — merge + eval"
bash run_eval_2a_sft_step300.sh 2>&1 | tee "${LOG_DIR}/eval_2a_sft_step300.log"

log_section "2A-SFT ALL EVALS COMPLETE"
echo "Metrics JSONs:"
echo "  /data-1/model_weights/WDL-SFT-Qwen3-4B-MATH-2A-SFT/step_275/inference_n3/eval_metrics.json"
echo "  /data-1/model_weights/WDL-SFT-Qwen3-4B-MATH-2A-SFT/step_300/inference_n3/eval_metrics.json"
