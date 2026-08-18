#!/usr/bin/env bash
set -euo pipefail
REPO_HOST=${REPO_HOST:-/data-1/code/verl}
MONITOR_NAME=${MONITOR_NAME:-math-wdl-fixed-m1-p60}
QUEUE_TMUX=${QUEUE_TMUX:-math-wdl-fixed-m1-p60-queue}
CKPT_ROOT=${CKPT_ROOT:-/data-1/checkpoints}
METRICS_ROOT=${METRICS_ROOT:-/data-2/model_weights/math_task/qwen3_1p7b_wdl_fixed_m1_p60/logs/metrics}
WANDB_PROJECT=${WANDB_PROJECT:-OnPolicyWDLSFT-Math-1P7B-Fixed-M1-P60}
POLL_SEC=${POLL_SEC:-60}
LOG_FILE=${LOG_FILE:-/data-2/model_weights/math_task/qwen3_1p7b_wdl_fixed_m1_p60/monitor.log}
QUEUE_LOG=${QUEUE_LOG:-/data-2/model_weights/math_task/qwen3_1p7b_wdl_fixed_m1_p60/queue.log}
VALIDATION_ROOT=${VALIDATION_ROOT:-/data-2/model_weights/math_task/qwen3_1p7b_wdl_fixed_m1_p60/logs/validation}
FORMAT_GATE_LOG=${FORMAT_GATE_LOG:-/data-2/model_weights/math_task/qwen3_1p7b_wdl_fixed_m1_p60/format_gate.log}
ADMISSION_ROOT=${ADMISSION_ROOT:-/data-2/model_weights/math_task/qwen3_1p7b_wdl_fixed_m1_p60/admission}
TRAINING_RELEASE_GATE=${TRAINING_RELEASE_GATE:-0}
RUN_PREFIXES=(
    MATH-WDL-FIXED-M1-COLD-START-P60-QWEN3-1P7B
    MATH-WDL-FIXED-M1-STAGE1-P60-QWEN3-1P7B
)
TMUX_NAMES=("$QUEUE_TMUX" "$QUEUE_TMUX")
FINAL_STEPS=(60 60)

mkdir -p "$ADMISSION_ROOT" "$(dirname "$LOG_FILE")"

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

# Each arm must independently prove zero Model1 gradient and nonzero Model2
# gradient at its first optimizer step. The second wait remains pending while
# the first arm runs because both share the queue tmux.
for prefix in "${RUN_PREFIXES[@]}"; do
    python3 "${REPO_HOST}/scripts/math_wdl_first_step_gate.py" \
        --metrics-root "$METRICS_ROOT" \
        --project "$WANDB_PROJECT" \
        --run-prefix "$prefix" \
        --expected-model1-gradient zero \
        --queue-tmux "$QUEUE_TMUX" \
        --timeout-seconds "${FIRST_STEP_TIMEOUT_SECONDS:-21600}" \
        --receipt "$ADMISSION_ROOT/first_step_${prefix}_$(date +%s).json"
done

if rg -n 'CUDA out of memory|NCCL ERROR|ActorDied|WorkerCrashed' \
    "$QUEUE_LOG" 2>/dev/null >"$ADMISSION_ROOT/first_step_hard_failures.txt"; then
    echo "ERROR: hard-failure markers found after fixed-M1 first-step gates" >&2
    exit 1
fi

source "${REPO_HOST}/scripts/training_queue_monitor.sh"
training_queue_monitor_main
