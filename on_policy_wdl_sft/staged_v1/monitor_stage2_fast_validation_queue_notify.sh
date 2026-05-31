#!/usr/bin/env bash
# Monitor the local Stage 2 queue and send WxPusher notifications.

set -euo pipefail

CKPT_ROOT=${CKPT_ROOT:-/data-1/checkpoints}
METRICS_ROOT=${METRICS_ROOT:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/staged_v1/metrics}
WANDB_PROJECT=${WANDB_PROJECT:-OnPolicySFT-Then-WDLSFT-StagedV1}
FINAL_STEP=${FINAL_STEP:-75}
POLL_SEC=${POLL_SEC:-300}
QUEUE_TMUX=${QUEUE_TMUX:-staged_v1_s2_queue}
LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/staged_v1/monitor_stage2_fast_validation_queue_notify.log}
WXPUSHER_SCRIPT=${WXPUSHER_SCRIPT:-/root/agent-core/skills/wxpusher-notify/scripts/wxpusher_notify.py}

RUN_PREFIXES=(
    "WDL-SFT-STAGED-V1-S2-FROM-S1-BETA0-BETA0"
    "WDL-SFT-STAGED-V1-S2-FROM-S1-BETA01-BETA01"
)
TMUX_NAMES=(
    "staged_v1_s2_from_s1_beta0_beta0"
    "staged_v1_s2_from_s1_beta01_beta01"
)

log() {
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

notify() {
    local title="$1" body="$2"
    if [ ! -f "$WXPUSHER_SCRIPT" ]; then
        log "WARNING: WxPusher script not found: $WXPUSHER_SCRIPT"
        return
    fi
    if ! python3 "$WXPUSHER_SCRIPT" --title "$title" --body "$body" >>"$LOG_FILE" 2>&1; then
        log "WARNING: WxPusher notification failed: $title"
    fi
}

latest_ckpt_dir() {
    local prefix="$1"
    find "$CKPT_ROOT" -maxdepth 1 -type d -name "${prefix}_*" 2>/dev/null | sort | tail -1
}

latest_step() {
    local ckpt_dir="$1"
    if [ -f "$ckpt_dir/latest_checkpointed_iteration.txt" ]; then
        tr -dc '0-9' < "$ckpt_dir/latest_checkpointed_iteration.txt"
        return
    fi
    find "$ckpt_dir" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null \
        | sed 's/.*global_step_//' | sort -n | tail -1
}

metrics_path_for_ckpt() {
    local ckpt_dir="$1"
    printf '%s/%s/%s.jsonl\n' "$METRICS_ROOT" "$WANDB_PROJECT" "$(basename "$ckpt_dir")"
}

has_final_metrics() {
    local ckpt_dir="$1" metrics_path
    [ -n "$ckpt_dir" ] || return 1
    metrics_path=$(metrics_path_for_ckpt "$ckpt_dir")
    [ -f "$metrics_path" ] || return 1
    python3 - "$metrics_path" "$FINAL_STEP" <<'PY'
import json
import sys

path = sys.argv[1]
final_step = int(sys.argv[2])
required = {
    "training/global_step",
    "actor/wdl_sft_beta",
    "actor/wdl_sft_loss_positive",
    "actor/wdl_sft_loss_negative",
    "actor/wdl_sft_loss_total",
    "wdl_sft/correct_ratio",
    "actor/grad_norm",
    "response/aborted_ratio",
    "val-core/HuggingFaceH4/MATH-500/acc/mean@3",
}
with open(path, "rb") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        data = row.get("data", {})
        step = data.get("training/global_step", row.get("step"))
        if step is not None and int(step) >= final_step and required.issubset(data):
            raise SystemExit(0)
raise SystemExit(1)
PY
}

log "Stage2 WxPusher monitor started; final_step=${FINAL_STEP}"

declare -A done
declare -A launched
for prefix in "${RUN_PREFIXES[@]}"; do
    done["$prefix"]=0
    launched["$prefix"]=0
done

while true; do
    completed_count=0
    for idx in "${!RUN_PREFIXES[@]}"; do
        prefix="${RUN_PREFIXES[$idx]}"
        tmux_name="${TMUX_NAMES[$idx]}"

        if [ "${done[$prefix]}" = "1" ]; then
            completed_count=$((completed_count + 1))
            continue
        fi

        ckpt_dir=$(latest_ckpt_dir "$prefix" || true)
        step=""
        if [ -n "$ckpt_dir" ]; then
            step=$(latest_step "$ckpt_dir" || true)
        fi

        if [ "${launched[$prefix]}" = "0" ] && [ -n "$ckpt_dir" ] && tmux has-session -t "$tmux_name" 2>/dev/null; then
            launched["$prefix"]=1
            log "launched: prefix=${prefix} latest_step=${step:-none} ckpt=${ckpt_dir}"
            notify "Stage2 WDL-SFT run launched" "Status: started
What happened: ${prefix} has an active training tmux session.
Evidence: latest_step=${step:-none}; checkpoint=${ckpt_dir}; tmux=${tmux_name}
Next action: Monitor will notify again when this run reaches final_step=${FINAL_STEP}."
        fi

        metrics_state="missing"
        if [ -n "$ckpt_dir" ]; then
            metrics_file=$(metrics_path_for_ckpt "$ckpt_dir")
            if has_final_metrics "$ckpt_dir"; then
                metrics_state="final"
            elif [ -f "$metrics_file" ]; then
                metrics_state="present-no-final"
            fi
        fi

        if [ -n "$step" ] && [ "$step" -ge "$FINAL_STEP" ] && [ "$metrics_state" = "final" ]; then
            done["$prefix"]=1
            completed_count=$((completed_count + 1))
            log "complete: prefix=${prefix} step=${step} metrics=${metrics_state} ckpt=${ckpt_dir}"
            notify "Stage2 WDL-SFT run complete" "Status: completed
What happened: ${prefix} reached final_step=${FINAL_STEP} and final metrics are logged.
Evidence: latest_step=${step}; metrics=$(metrics_path_for_ckpt "$ckpt_dir"); checkpoint=${ckpt_dir}
Next action: The queue/monitor will continue to the next run or finish."
            continue
        fi

        if [ -n "$ckpt_dir" ] \
            && ! tmux has-session -t "$tmux_name" 2>/dev/null \
            && ! tmux has-session -t "$QUEUE_TMUX" 2>/dev/null; then
            log "failed: prefix=${prefix} latest_step=${step:-none} metrics=${metrics_state} ckpt=${ckpt_dir}"
            notify "Stage2 WDL-SFT run failed" "Status: failed
What happened: ${prefix} stopped before final_step=${FINAL_STEP} and final metrics were verified.
Evidence: latest_step=${step:-none}; metrics=${metrics_state}; checkpoint=${ckpt_dir}
Next action: Inspect the run log under recipe/on_policy_wdl_sft/staged_v1/."
            exit 1
        fi

        log "waiting: prefix=${prefix} latest_step=${step:-none} metrics=${metrics_state} ckpt=${ckpt_dir:-none}"
    done

    if [ "$completed_count" -eq "${#RUN_PREFIXES[@]}" ]; then
        log "all Stage2 runs complete"
        notify "Stage2 WDL-SFT queue complete" "Status: completed
What happened: Both Stage2 fast-validation runs completed.
Evidence: final_step=${FINAL_STEP}; log=${LOG_FILE}
Next action: Review validation metrics and compare against Stage1 baselines."
        exit 0
    fi

    sleep "$POLL_SEC"
done
