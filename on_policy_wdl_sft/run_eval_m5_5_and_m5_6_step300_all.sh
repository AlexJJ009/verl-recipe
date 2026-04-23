#!/usr/bin/env bash
# Serial driver: run M5.5 + M5.6 step 300 offline evals (model2 and model1 each),
# using the same conditions as LR3 step 125 (EVAL-10 / EVAL-11).
#
# Run inside the verl-harness Docker container with /data-1 and /data-2 mounted.
# The outer launcher (docker run ... + tmux) is recorded at the bottom of this file.
set -euo pipefail

cd /workspace/verl/recipe/on_policy_wdl_sft
LOG_DIR=/workspace/verl/recipe/on_policy_wdl_sft

log_section() {
    echo ""
    echo "==================================================================="
    echo "[$(date -Iseconds)] $*"
    echo "==================================================================="
}

log_section "EXP-13 M5.5 step 300 — merge + extract model2 + eval model2"
bash run_eval_m5_5_step300.sh 2>&1 | tee "$LOG_DIR/eval_m5_5_step300.log"

log_section "EXP-13 M5.5 step 300 — extract model1 + eval model1"
bash run_eval_m5_5_step300_model1.sh 2>&1 | tee "$LOG_DIR/eval_m5_5_step300_model1.log"

log_section "EXP-14 M5.6 step 300 — merge + extract model2 + eval model2"
bash run_eval_m5_6_step300.sh 2>&1 | tee "$LOG_DIR/eval_m5_6_step300.log"

log_section "EXP-14 M5.6 step 300 — extract model1 + eval model1"
bash run_eval_m5_6_step300_model1.sh 2>&1 | tee "$LOG_DIR/eval_m5_6_step300_model1.log"

log_section "All four evals complete."
echo "Metrics JSONs written to:"
echo "  /data-1/model_weights/WDL-SFT-4B-MATH-M5-5/step_300_model2/inference_n3/eval_metrics.json"
echo "  /data-1/model_weights/WDL-SFT-4B-MATH-M5-5/step_300_model1/inference_n3/eval_metrics.json"
echo "  /data-1/model_weights/WDL-SFT-4B-MATH-M5-6/step_300_model2/inference_n3/eval_metrics.json"
echo "  /data-1/model_weights/WDL-SFT-4B-MATH-M5-6/step_300_model1/inference_n3/eval_metrics.json"

# ------------------------------------------------------------------
# Outer launch reference (for restart / manual re-run):
#
#   tmux new-session -d -s eval_m5_all "\
#     docker run --rm --gpus all --ipc=host \
#       -v /data-1/verl07/verl:/workspace/verl \
#       -v /data-1:/data-1 \
#       -v /data-2:/data-2 \
#       --name eval_m5_all_${RANDOM} \
#       verl-harness \
#       bash /workspace/verl/recipe/on_policy_wdl_sft/run_eval_m5_5_and_m5_6_step300_all.sh \
#       2>&1 | tee /data-1/verl07/verl/recipe/on_policy_wdl_sft/eval_m5_all_master.log"
# ------------------------------------------------------------------
