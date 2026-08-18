#!/usr/bin/env bash
# Monitor for the Qwen3-1.7B code format cold-start SFT fraction queue.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

QUEUE_TMUX=${QUEUE_TMUX:-code_task_qwen3_1p7b_coldstart_sft_fraction_queue}
CKPT_ROOT=${CKPT_ROOT:-/data-1/checkpoints/format_cold_start_fraction_cot_v3}
MERGED_ROOT=${MERGED_ROOT:-/data-1/model_weights/format_cold_start_fraction_cot_v3}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data-1/eval_outputs/code_task/qwen3_1p7b_coldstart_sft_fraction}
LOG_FILE=${LOG_FILE:-"${SCRIPT_DIR}/monitor_code_task_qwen3_1p7b_coldstart_sft_fraction_notify.log"}
POLL_SEC=${POLL_SEC:-300}
WXPUSHER_NOTIFY=${WXPUSHER_NOTIFY:-1}
WXPUSHER_SCRIPT=${WXPUSHER_SCRIPT:-/root/agent-core/skills/wxpusher-notify/scripts/wxpusher_notify.py}
EVAL_BENCHMARKS=${EVAL_BENCHMARKS:-"humaneval mbpp livecodebench"}

RUN_LABELS=("frac25" "frac50" "frac100")
RUN_PREFIXES=(
  "${FRAC25_RUN_PREFIX:-SFT-FORMAT-COLDSTART-Qwen3-1P7B-CODE-KODCODE-FRAC25-COT-V3}"
  "${FRAC50_RUN_PREFIX:-SFT-FORMAT-COLDSTART-Qwen3-1P7B-CODE-KODCODE-FRAC50-COT-V3}"
  "${FRAC100_RUN_PREFIX:-SFT-FORMAT-COLDSTART-Qwen3-1P7B-CODE-KODCODE-FRAC100-COT-V3}"
)
TMUX_NAMES=(
  "format_cold_start_sft_q17b_code_frac25_cot_v3"
  "format_cold_start_sft_q17b_code_frac50_cot_v3"
  "format_cold_start_sft_q17b_code_frac100_cot_v3"
)
FINAL_STEPS=(
  "${FRAC25_STEPS:-30}"
  "${FRAC50_STEPS:-60}"
  "${FRAC100_STEPS:-120}"
)

log() { mkdir -p "$(dirname "$LOG_FILE")"; echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

notify() {
    local title="$1" body="$2"
    if [ "$WXPUSHER_NOTIFY" != "1" ]; then
        return
    fi
    [ -f "$WXPUSHER_SCRIPT" ] && python3 "$WXPUSHER_SCRIPT" --title "$title" --body "$body" >>"$LOG_FILE" 2>&1 || true
}

latest_run_dir() {
    [ -d "$CKPT_ROOT" ] || return 0
    find "$CKPT_ROOT" -maxdepth 1 -type d -name "$1*" 2>/dev/null | sort | tail -1
}

latest_step() {
    local run_dir="$1"
    [ -n "$run_dir" ] && [ -d "$run_dir" ] || { echo 0; return; }
    if [ -f "${run_dir}/latest_checkpointed_iteration.txt" ]; then
        tr -dc '0-9' <"${run_dir}/latest_checkpointed_iteration.txt"
        return
    fi
    find "$run_dir" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1 | awk '{print $1 + 0}'
}

merged_ready() {
    [ -f "$1/model.safetensors" ] || [ -f "$1/model.safetensors.index.json" ] || [ -f "$1/pytorch_model.bin" ] || [ -f "$1/pytorch_model.bin.index.json" ]
}

eval_complete() {
    local label="$1" benchmark
    for benchmark in $EVAL_BENCHMARKS; do
        [ -s "${OUTPUT_ROOT}/${label}/${benchmark}/official_summary.json" ] || return 1
        [ -s "${OUTPUT_ROOT}/${label}/${benchmark}/conversion_report.json" ] || return 1
    done
}

log "qwen3_1p7b_coldstart_sft_fraction monitor start; queue=${QUEUE_TMUX}"
notify "Qwen3-1.7B coldstart SFT fraction monitor started" "queue=${QUEUE_TMUX}; output=${OUTPUT_ROOT}"

declare -A notified_complete
for label in "${RUN_LABELS[@]}"; do
    notified_complete["$label"]=0
done

while true; do
    any_active=0
    tmux has-session -t "$QUEUE_TMUX" 2>/dev/null && any_active=1
    for idx in "${!RUN_LABELS[@]}"; do
        label="${RUN_LABELS[$idx]}"
        prefix="${RUN_PREFIXES[$idx]}"
        tmux_name="${TMUX_NAMES[$idx]}"
        final="${FINAL_STEPS[$idx]}"
        merged="${MERGED_ROOT}/qwen3-1p7b-kodcode-format-sft-${label}"
        run_dir=$(latest_run_dir "$prefix")
        step=$(latest_step "$run_dir")
        tmux has-session -t "$tmux_name" 2>/dev/null && any_active=1
        train_status=missing
        [ -n "$run_dir" ] && train_status=checkpoint
        [ "$step" -ge "$final" ] && train_status=final
        merge_status=missing
        merged_ready "$merged" && merge_status=ready
        eval_status=missing
        eval_complete "$label" && eval_status=ready
        log "${label}: step=${step} final=${final} train=${train_status} merged=${merge_status} eval=${eval_status} run_dir=${run_dir:-none} merged_dir=${merged}"
        if [ "$train_status" = "final" ] && [ "$merge_status" = "ready" ] && [ "$eval_status" = "ready" ] && [ "${notified_complete[$label]}" = "0" ]; then
            notified_complete["$label"]=1
            notify "Qwen3-1.7B coldstart SFT fraction complete" "${label} reached final step, merged HF model, and eval summaries are ready. Evidence: ${OUTPUT_ROOT}/${label}"
        fi
    done
    if [ "$any_active" = "0" ]; then
        log "monitor exit: no queue/training tmux active"
        notify "Qwen3-1.7B coldstart SFT fraction monitor exited" "No active queue/training tmux. Review ${LOG_FILE} and ${OUTPUT_ROOT}/coldstart_sft_fraction_curve.csv."
        exit 0
    fi
    sleep "$POLL_SEC"
done
