#!/usr/bin/env bash
set -euo pipefail
REPO_HOST=${REPO_HOST:-/data-1/code/verl}
MONITOR_NAME=${MONITOR_NAME:-math-wdl-causal-p60}
QUEUE_TMUX=${QUEUE_TMUX:-math-wdl-causal-p60-queue}
CKPT_ROOT=${CKPT_ROOT:-/data-1/checkpoints}
METRICS_ROOT=${METRICS_ROOT:-/data-2/model_weights/math_task/qwen3_1p7b_wdl_causal_p60/logs/metrics}
WANDB_PROJECT=${WANDB_PROJECT:-OnPolicyWDLSFT-Math-1P7B-Causal-P60}
POLL_SEC=${POLL_SEC:-60}
LOG_FILE=${LOG_FILE:-/data-2/model_weights/math_task/qwen3_1p7b_wdl_causal_p60/monitor.log}
QUEUE_LOG=${QUEUE_LOG:-/data-2/model_weights/math_task/qwen3_1p7b_wdl_causal_p60/queue.log}
VALIDATION_ROOT=${VALIDATION_ROOT:-/data-2/model_weights/math_task/qwen3_1p7b_wdl_causal_p60/logs/validation}
FORMAT_GATE_LOG=${FORMAT_GATE_LOG:-/data-2/model_weights/math_task/qwen3_1p7b_wdl_causal_p60/format_gate.log}
TRAINING_RELEASE_GATE=${TRAINING_RELEASE_GATE:-0}
RUN_PREFIXES=(
    MATH-WDL-CAUSAL-P60-ARM-D0-QWEN3-1P7B
    MATH-WDL-CAUSAL-P60-ARM-C-QWEN3-1P7B
)
TMUX_NAMES=("$QUEUE_TMUX" "$QUEUE_TMUX")
FINAL_STEPS=(60 60)
python3 "${REPO_HOST}/scripts/math_wdl_first_step_gate.py" \
    --metrics-root "$METRICS_ROOT" \
    --project "$WANDB_PROJECT" \
    --run-prefix "${RUN_PREFIXES[0]}" \
    --expected-model1-gradient zero \
    --queue-tmux "$QUEUE_TMUX" \
    --timeout-seconds "${FIRST_STEP_TIMEOUT_SECONDS:-7200}" \
    --receipt /data-2/model_weights/math_task/qwen3_1p7b_wdl_causal_p60/admission/first_step_receipt_$(date +%s).json
if rg -n 'CUDA out of memory|NCCL ERROR|ActorDied|WorkerCrashed' \
    "$QUEUE_LOG" 2>/dev/null \
    > /data-2/model_weights/math_task/qwen3_1p7b_wdl_causal_p60/admission/first_step_hard_failures.txt; then
    echo "ERROR: hard-failure markers found after first step; inspect first_step_hard_failures.txt" >&2
    exit 1
fi
format_guard() {
    while tmux has-session -t "$QUEUE_TMUX" 2>/dev/null; do
        set +e
        python3 "${REPO_HOST}/scripts/math_wdl_format_gate.py" \
            --validation-root "$VALIDATION_ROOT" \
            --run-prefix "${RUN_PREFIXES[0]}" \
            --run-prefix "${RUN_PREFIXES[1]}" \
            --max-drop 0.05 \
            --required-consecutive 2 >>"$FORMAT_GATE_LOG" 2>&1
        gate_rc=$?
        set -e
        if [ "$gate_rc" -eq 2 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: sustained format-contract collapse; stopping ${QUEUE_TMUX}" \
                | tee -a "$FORMAT_GATE_LOG" "$LOG_FILE"
            tmux kill-session -t "$QUEUE_TMUX" 2>/dev/null || true
            return 2
        fi
        if [ "$gate_rc" -ne 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: format gate execution failed rc=${gate_rc}; stopping ${QUEUE_TMUX}" \
                | tee -a "$FORMAT_GATE_LOG" "$LOG_FILE"
            tmux kill-session -t "$QUEUE_TMUX" 2>/dev/null || true
            return "$gate_rc"
        fi
        sleep "$POLL_SEC"
    done
}
format_guard &
FORMAT_GUARD_PID=$!
trap 'kill "$FORMAT_GUARD_PID" 2>/dev/null || true' EXIT
source "${REPO_HOST}/scripts/training_queue_monitor.sh"
training_queue_monitor_main
