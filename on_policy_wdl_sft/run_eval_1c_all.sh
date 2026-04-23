#!/usr/bin/env bash
# Serial driver: run 1C step_150 + step_300 offline evals (model2 and model1 each).
# Run inside verl-harness Docker container with /data-1 mounted.
# Total ~3h on L40S tp=8.
set -euo pipefail

cd /workspace/verl/recipe/on_policy_wdl_sft
LOG_DIR=/workspace/verl/recipe/on_policy_wdl_sft

log_section() {
    echo ""
    echo "==================================================================="
    echo "[1C-EVAL-MARKER] [$(date -Iseconds)] $*"
    echo "==================================================================="
}

log_section "EXP-18 1C step 150 — merge + extract model2 + eval model2"
bash run_eval_1c_step150.sh 2>&1 | tee "$LOG_DIR/eval_1c_step150.log"

log_section "EXP-18 1C step 150 — extract model1 + eval model1"
bash run_eval_1c_step150_model1.sh 2>&1 | tee "$LOG_DIR/eval_1c_step150_model1.log"

log_section "EXP-18 1C step 300 — merge + extract model2 + eval model2"
bash run_eval_1c_step300.sh 2>&1 | tee "$LOG_DIR/eval_1c_step300.log"

log_section "EXP-18 1C step 300 — extract model1 + eval model1"
bash run_eval_1c_step300_model1.sh 2>&1 | tee "$LOG_DIR/eval_1c_step300_model1.log"

log_section "1C ALL FOUR EVALS COMPLETE"
echo "Metrics JSONs:"
echo "  /data-1/model_weights/WDL-SFT-4B-MATH-1C/step_150_model2/inference_n3/eval_metrics.json"
echo "  /data-1/model_weights/WDL-SFT-4B-MATH-1C/step_150_model1/inference_n3/eval_metrics.json"
echo "  /data-1/model_weights/WDL-SFT-4B-MATH-1C/step_300_model2/inference_n3/eval_metrics.json"
echo "  /data-1/model_weights/WDL-SFT-4B-MATH-1C/step_300_model1/inference_n3/eval_metrics.json"
