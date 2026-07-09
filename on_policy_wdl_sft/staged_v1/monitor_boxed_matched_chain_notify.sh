#!/usr/bin/env bash
# Monitor the boxed matched Stage1 -> fixed merge -> Stage2 chain queue.

set -euo pipefail

CKPT_ROOT=${CKPT_ROOT:-/data-1/checkpoints}
MODEL2_ROOT=${MODEL2_ROOT:-/data-1/model_weights/staged_v1/boxed_matched}
METRICS_ROOT=${METRICS_ROOT:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/staged_v1/metrics}
WANDB_PROJECT=${WANDB_PROJECT:-OnPolicySFT-Then-WDLSFT-StagedV1}
STAGE1_FINAL_STEP=${STAGE1_FINAL_STEP:-150}
STAGE2_FINAL_STEP=${STAGE2_FINAL_STEP:-75}
POLL_SEC=${POLL_SEC:-300}
QUEUE_TMUX=${QUEUE_TMUX:-staged_v1_boxed_matched_chain_queue}
LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/staged_v1/monitor_boxed_matched_chain_notify.log}
WXPUSHER_SCRIPT=${WXPUSHER_SCRIPT:-/root/agent-core/skills/wxpusher-notify/scripts/wxpusher_notify.py}
TRAINING_RELEASE_GATE_SHELL=${TRAINING_RELEASE_GATE_SHELL:-/data-1/verl07/verl/scripts/training_release_gate_shell.sh}

if [ -f "$TRAINING_RELEASE_GATE_SHELL" ]; then
    # shellcheck disable=SC1090
    source "$TRAINING_RELEASE_GATE_SHELL"
fi

BETA_LABELS=("beta0" "beta01")
STAGE1_PREFIXES=(
    "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BOXED-BETA0-V1"
    "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BOXED-BETA01-V1"
)
STAGE2_PREFIXES=(
    "WDL-SFT-STAGED-V1-S2-BOXED-FROM-S1-BETA0-BETA0"
    "WDL-SFT-STAGED-V1-S2-BOXED-FROM-S1-BETA01-BETA01"
)
STAGE1_TMUX_NAMES=(
    "staged_v1_chain_s1_boxed_beta0"
    "staged_v1_chain_s1_boxed_beta01"
)
STAGE2_TMUX_NAMES=(
    "staged_v1_chain_s2_boxed_beta0"
    "staged_v1_chain_s2_boxed_beta01"
)
MERGED_MODEL2_DIRS=(
    "${MODEL2_ROOT}/model2-from-s1-boxed-beta0-best"
    "${MODEL2_ROOT}/model2-from-s1-boxed-beta01-best"
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

has_stage2_final_metrics() {
    local ckpt_dir="$1" metrics_path
    [ -n "$ckpt_dir" ] || return 1
    metrics_path=$(metrics_path_for_ckpt "$ckpt_dir")
    [ -f "$metrics_path" ] || return 1
    python3 - "$metrics_path" "$STAGE2_FINAL_STEP" <<'PY'
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

has_model2_weights() {
    local model_dir="$1"
    [ -f "${model_dir}/model.safetensors" ] || [ -f "${model_dir}/model.safetensors.index.json" ]
}

log "boxed matched chain monitor started; stage1_final=${STAGE1_FINAL_STEP}; stage2_final=${STAGE2_FINAL_STEP}"

declare -A s1_done merge_done s2_launched s2_done
for beta in "${BETA_LABELS[@]}"; do
    s1_done["$beta"]=0
    merge_done["$beta"]=0
    s2_launched["$beta"]=0
    s2_done["$beta"]=0
done

while true; do
    completed=0
    for idx in "${!BETA_LABELS[@]}"; do
        beta="${BETA_LABELS[$idx]}"
        s1_prefix="${STAGE1_PREFIXES[$idx]}"
        s2_prefix="${STAGE2_PREFIXES[$idx]}"
        s1_tmux="${STAGE1_TMUX_NAMES[$idx]}"
        s2_tmux="${STAGE2_TMUX_NAMES[$idx]}"
        model2_dir="${MERGED_MODEL2_DIRS[$idx]}"

        s1_ckpt=$(latest_ckpt_dir "$s1_prefix" || true)
        s1_step=""
        if [ -n "$s1_ckpt" ]; then
            s1_step=$(latest_step "$s1_ckpt" || true)
        fi
        if [ "${s1_done[$beta]}" = "0" ] && [ -n "$s1_step" ] && [ "$s1_step" -ge "$STAGE1_FINAL_STEP" ]; then
            s1_done["$beta"]=1
            log "Stage1 complete: beta=${beta} step=${s1_step} ckpt=${s1_ckpt}"
            if declare -F training_release_gate_record_event >/dev/null 2>&1; then
                training_release_gate_record_event "boxed_matched_chain" "$s1_prefix" "success_complete" "$s1_step" "$STAGE1_FINAL_STEP" "$s1_ckpt" "$(metrics_path_for_ckpt "$s1_ckpt")" "Stage1 reached configured final checkpoint." "$LOG_FILE"
            fi
            notify "Boxed WDL-SFT Stage1 complete" "Status: completed
What happened: Stage1 reached final checkpoint for ${beta}.
Evidence: step=${s1_step}; checkpoint=${s1_ckpt}
Next action: Waiting for fixed Model2 merge."
        fi

        if [ "${merge_done[$beta]}" = "0" ] && has_model2_weights "$model2_dir"; then
            merge_done["$beta"]=1
            log "merge complete: beta=${beta} model2=${model2_dir}"
            notify "Boxed WDL-SFT Model2 ready" "Status: completed
What happened: Fixed Model2 dir is ready for ${beta}.
Evidence: model2=${model2_dir}
Next action: Waiting for matched Stage2."
        fi

        s2_ckpt=$(latest_ckpt_dir "$s2_prefix" || true)
        s2_step=""
        metrics_state="missing"
        if [ -n "$s2_ckpt" ]; then
            s2_step=$(latest_step "$s2_ckpt" || true)
            metrics_file=$(metrics_path_for_ckpt "$s2_ckpt")
            if has_stage2_final_metrics "$s2_ckpt"; then
                metrics_state="final"
            elif [ -f "$metrics_file" ]; then
                metrics_state="present-no-final"
            fi
        fi
        if [ "${s2_launched[$beta]}" = "0" ] && [ -n "$s2_ckpt" ] && tmux has-session -t "$s2_tmux" 2>/dev/null; then
            s2_launched["$beta"]=1
            log "Stage2 launched: beta=${beta} step=${s2_step:-none} ckpt=${s2_ckpt}"
            notify "Boxed WDL-SFT Stage2 launched" "Status: started
What happened: Stage2 has an active tmux session for ${beta}.
Evidence: checkpoint=${s2_ckpt}; tmux=${s2_tmux}
Next action: Waiting for final checkpoint and metrics."
        fi
        if [ "${s2_done[$beta]}" = "0" ] && [ -n "$s2_step" ] && [ "$s2_step" -ge "$STAGE2_FINAL_STEP" ] && [ "$metrics_state" = "final" ]; then
            s2_done["$beta"]=1
            log "Stage2 complete: beta=${beta} step=${s2_step} metrics=${metrics_state} ckpt=${s2_ckpt}"
            if declare -F training_release_gate_record_event >/dev/null 2>&1; then
                training_release_gate_record_event "boxed_matched_chain" "$s2_prefix" "success_complete" "$s2_step" "$STAGE2_FINAL_STEP" "$s2_ckpt" "$(metrics_path_for_ckpt "$s2_ckpt")" "Stage2 reached configured final checkpoint with final metrics evidence." "$LOG_FILE"
            fi
            notify "Boxed WDL-SFT Stage2 complete" "Status: completed
What happened: Stage2 reached final checkpoint and metrics for ${beta}.
Evidence: step=${s2_step}; metrics=$(metrics_path_for_ckpt "$s2_ckpt"); checkpoint=${s2_ckpt}
Next action: Queue will continue or finish."
        fi

        if [ "${s2_done[$beta]}" = "1" ]; then
            completed=$((completed + 1))
        fi

        if ! tmux has-session -t "$QUEUE_TMUX" 2>/dev/null \
            && ! tmux has-session -t "$s1_tmux" 2>/dev/null \
            && ! tmux has-session -t "$s2_tmux" 2>/dev/null \
            && [ "${s2_done[$beta]}" = "0" ] \
            && { [ -n "$s1_ckpt" ] || [ -n "$s2_ckpt" ]; }; then
            log "failed/stopped: beta=${beta} s1_step=${s1_step:-none} s2_step=${s2_step:-none} metrics=${metrics_state}"
            if declare -F training_release_gate_record_event >/dev/null 2>&1; then
                if [ -n "$s2_ckpt" ]; then
                    training_release_gate_record_event "boxed_matched_chain" "$s2_prefix" "failed" "${s2_step:-none}" "$STAGE2_FINAL_STEP" "$s2_ckpt" "$(metrics_path_for_ckpt "$s2_ckpt")" "Stage2 stopped before configured final checkpoint/final metrics." "$LOG_FILE"
                elif [ -n "$s1_ckpt" ]; then
                    training_release_gate_record_event "boxed_matched_chain" "$s1_prefix" "failed" "${s1_step:-none}" "$STAGE1_FINAL_STEP" "$s1_ckpt" "$(metrics_path_for_ckpt "$s1_ckpt")" "Stage1 stopped before downstream chain completion." "$LOG_FILE"
                fi
            fi
            notify "Boxed WDL-SFT chain stopped" "Status: failed
What happened: Queue and training tmux sessions are gone before ${beta} completed Stage2.
Evidence: s1_step=${s1_step:-none}; s2_step=${s2_step:-none}; metrics=${metrics_state}
Next action: Inspect queue and run logs under recipe/on_policy_wdl_sft/staged_v1/."
            exit 1
        fi

        log "waiting: beta=${beta} s1_step=${s1_step:-none} model2=$(has_model2_weights "$model2_dir" && echo ready || echo missing) s2_step=${s2_step:-none} metrics=${metrics_state}"
    done

    if [ "$completed" -eq "${#BETA_LABELS[@]}" ]; then
        log "all boxed matched chains complete"
        notify "Boxed WDL-SFT chain complete" "Status: completed
What happened: Both boxed matched chains completed.
Evidence: final Stage2 step=${STAGE2_FINAL_STEP}; log=${LOG_FILE}
Next action: Review validation metrics."
        exit 0
    fi

    sleep "$POLL_SEC"
done
