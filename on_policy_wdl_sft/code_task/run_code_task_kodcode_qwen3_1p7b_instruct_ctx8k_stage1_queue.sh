#!/usr/bin/env bash
# Host-side KodCode Qwen3-1.7B chat/instruct CTX8K Stage1 queue.
set -euo pipefail

REPO_HOST=${REPO_HOST:-/data-1/verl07/verl}
REPO_CONTAINER=${REPO_CONTAINER:-/workspace/verl}
DOCKER_IMAGE=${DOCKER_IMAGE:-verl-harness:latest}
CKPT_ROOT=${CKPT_ROOT:-/data-1/checkpoints}
METRICS_ROOT=${METRICS_ROOT:-"${REPO_HOST}/recipe/on_policy_wdl_sft/code_task/metrics"}
WANDB_PROJECT=${WANDB_PROJECT:-OnPolicyWDLSFT-CodeTask}
LOG_FILE=${LOG_FILE:-"${REPO_HOST}/recipe/on_policy_wdl_sft/code_task/run_code_task_kodcode_qwen3_1p7b_instruct_ctx8k_stage1_queue.log"}
QUEUE_STATUS_FILE=${QUEUE_STATUS_FILE:-"${LOG_FILE%.log}_status.tsv"}
START_INDEX=${START_INDEX:-0}
END_INDEX=${END_INDEX:-1}
QUEUE_CONTINUE_ON_FAILURE=${QUEUE_CONTINUE_ON_FAILURE:-0}
ALLOW_RESUME=${ALLOW_RESUME:-0}
MIN_FREE_GB=${MIN_FREE_GB:-300}
WXPUSHER_NOTIFY=${WXPUSHER_NOTIFY:-1}
WXPUSHER_SCRIPT=${WXPUSHER_SCRIPT:-/root/agent-core/skills/wxpusher-notify/scripts/wxpusher_notify.py}

RUN_LABELS=("kodcode-qwen3-1p7b-instruct-ctx8k-s1-beta0" "kodcode-qwen3-1p7b-instruct-ctx8k-s1-beta01")
RUN_SCRIPTS=(
  "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/code_task/run_s1_code_kodcode_qwen3_1p7b_instruct_ctx8k_beta_0.sh"
  "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/code_task/run_s1_code_kodcode_qwen3_1p7b_instruct_ctx8k_beta_01.sh"
)
RUN_PREFIXES=(
  "ONPOLICY-SFT-Qwen3-1P7B-INSTRUCT-CODE-KODCODE-CTX8K-S1-BETA0-V1"
  "ONPOLICY-SFT-Qwen3-1P7B-INSTRUCT-CODE-KODCODE-CTX8K-S1-BETA01-V1"
)
TMUX_NAMES=("code_task_s1_kodcode_qwen3_1p7b_beta0" "code_task_s1_kodcode_qwen3_1p7b_beta01")
FINAL_STEPS=(150 150)

if [ "${DRY_RUN:-0}" != "1" ] && [ "${ALLOW_KODCODE_QWEN3_1P7B_INSTRUCT_CTX8K_STAGE1_TRAINING:-0}" != "1" ]; then
    echo "[kodcode qwen3 1p7b instruct ctx8k stage1 queue] ERROR: non-dry-run requires explicit ALLOW_KODCODE_QWEN3_1P7B_INSTRUCT_CTX8K_STAGE1_TRAINING=1" >&2
    exit 1
fi

log() { mkdir -p "$(dirname "$LOG_FILE")"; echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

record_item_status() {
    local idx="$1" label="$2" prefix="$3" status="$4" evidence="$5"
    mkdir -p "$(dirname "$QUEUE_STATUS_FILE")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$idx" "$label" "$prefix" "$status" "$evidence" >>"$QUEUE_STATUS_FILE"
}

notify() {
    local title="$1" body="$2"
    if [ "$WXPUSHER_NOTIFY" != "1" ] || [ "${DRY_RUN:-0}" = "1" ]; then
        return
    fi
    if [ -f "$WXPUSHER_SCRIPT" ]; then
        python3 "$WXPUSHER_SCRIPT" --title "$title" --body "$body" >>"$LOG_FILE" 2>&1 || true
    fi
}

get_df_target() {
    local path="$1"
    while [ ! -e "$path" ] && [ "$path" != "/" ]; do
        path=$(dirname "$path")
    done
    printf '%s\n' "$path"
}

get_free_gb() {
    local target
    target=$(get_df_target "$1")
    df -Pk "$target" | awk 'NR==2 {print int($4 / 1024 / 1024)}'
}

latest_ckpt_dir() { find "$CKPT_ROOT" -maxdepth 1 -type d -name "$1_*" 2>/dev/null | sort | tail -1; }

latest_ckpt_step() {
    local ckpt_dir="$1" step
    if [ -z "$ckpt_dir" ] || [ ! -d "$ckpt_dir" ]; then
        echo 0
        return
    fi
    if [ -f "${ckpt_dir}/latest_checkpointed_iteration.txt" ]; then
        step=$(tr -dc '0-9' < "${ckpt_dir}/latest_checkpointed_iteration.txt")
        echo "${step:-0}"
        return
    fi
    step=$(find "$ckpt_dir" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null \
        | sed 's/.*global_step_//' | sort -n | tail -1 | awk '{print $1 + 0}')
    echo "${step:-0}"
}

latest_metrics_file() {
    local ckpt_dir="$1" run_name candidate
    [ -n "$ckpt_dir" ] || return
    run_name=$(basename "$ckpt_dir")
    candidate="${METRICS_ROOT}/${WANDB_PROJECT}/${run_name}.jsonl"
    [ -f "$candidate" ] && printf '%s\n' "$candidate"
}

is_run_complete() {
    local prefix="$1" final_step="$2" ckpt step metrics
    ckpt=$(latest_ckpt_dir "$prefix" || true)
    step=$(latest_ckpt_step "$ckpt")
    metrics=$(latest_metrics_file "$ckpt" || true)
    [ -n "$ckpt" ] && [ "$step" -ge "$final_step" ] && [ -n "$metrics" ]
}

precheck_disk_or_stop() {
    local free_gb
    [ "${DRY_RUN:-0}" = "1" ] && return 0
    free_gb=$(get_free_gb "$CKPT_ROOT")
    if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
        log "ERROR disk gate failed: CKPT_ROOT=${CKPT_ROOT} free=${free_gb}G need>=${MIN_FREE_GB}G"
        record_item_status "${CURRENT_IDX:-NA}" "${CURRENT_LABEL:-NA}" "${CURRENT_PREFIX:-NA}" "blocked_disk" "free=${free_gb}G need>=${MIN_FREE_GB}G"
        notify "KodCode Qwen3 1.7B Stage1 blocked" "Disk gate failed: free=${free_gb}G need>=${MIN_FREE_GB}G"
        exit 1
    fi
}

wait_for_tmux_run() {
    local tmux_name="$1" prefix="$2" final_step="$3" idx="$4" label="$5"
    [ "${DRY_RUN:-0}" = "1" ] && return 0
    while tmux has-session -t "$tmux_name" 2>/dev/null; do
        local ckpt step
        ckpt=$(latest_ckpt_dir "$prefix" || true)
        step=$(latest_ckpt_step "$ckpt")
        log "waiting ${tmux_name}: prefix=${prefix} step=${step} final=${final_step} ckpt=${ckpt:-none}"
        sleep "${QUEUE_POLL_SEC:-30}"
    done
    local ckpt step metrics
    ckpt=$(latest_ckpt_dir "$prefix" || true)
    step=$(latest_ckpt_step "$ckpt")
    metrics=$(latest_metrics_file "$ckpt" || true)
    if [ -z "$ckpt" ] || [ "$step" -lt "$final_step" ] || [ -z "$metrics" ]; then
        log "ERROR ${label} exited before final checkpoint+metrics: prefix=${prefix} step=${step} final=${final_step} ckpt=${ckpt:-none} metrics=${metrics:-none}"
        record_item_status "$idx" "$label" "$prefix" "failed" "step=${step};final=${final_step};ckpt=${ckpt:-none};metrics=${metrics:-none}"
        notify "KodCode Qwen3 1.7B Stage1 failed" "${label} stopped before final checkpoint/metrics."
        if [ "$QUEUE_CONTINUE_ON_FAILURE" = "1" ]; then
            return 1
        fi
        exit 1
    fi
    log "completed ${label}: prefix=${prefix} step=${step} ckpt=${ckpt} metrics=${metrics}"
    record_item_status "$idx" "$label" "$prefix" "completed" "step=${step};ckpt=${ckpt};metrics=${metrics}"
    notify "KodCode Qwen3 1.7B Stage1 complete" "${label} reached step ${step}."
}

launch_container() {
    local container_script="$1" tmux_name="$2"
    local host_script="${container_script/#$REPO_CONTAINER/$REPO_HOST}"
    [ -f "$host_script" ] || { log "ERROR missing script: $host_script"; exit 1; }
    if [ "${DRY_RUN:-0}" = "1" ]; then
        log "DRY_RUN would launch ${tmux_name}: ${host_script}"
        if [ "${QUEUE_DRY_RUN_VALIDATE_WRAPPERS:-0}" = "1" ]; then
            log "DRY_RUN validating wrapper ${host_script}"
            (cd "$REPO_HOST" && DRY_RUN=1 bash "$host_script")
        fi
        return
    fi
    log "launching ${tmux_name}: ${host_script}"
    tmux new-session -d -s "$tmux_name" \
        "docker run --rm --gpus all --ipc=host --network=host --shm-size=64g -v /data-1:/data-1 -v '$REPO_HOST':'$REPO_CONTAINER' -w '$REPO_CONTAINER' '$DOCKER_IMAGE' bash -lc \"bash '$container_script'\" 2>&1 | tee -a '$LOG_FILE'"
}

log "kodcode qwen3 1p7b instruct ctx8k stage1 queue start START_INDEX=${START_INDEX} END_INDEX=${END_INDEX} DRY_RUN=${DRY_RUN:-0} MIN_FREE_GB=${MIN_FREE_GB} STATUS_FILE=${QUEUE_STATUS_FILE}"
notify "KodCode Qwen3 1.7B Stage1 queue started" "range=${START_INDEX}-${END_INDEX}; min_free_gb=${MIN_FREE_GB}"

for idx in "${!RUN_LABELS[@]}"; do
    [ "$idx" -lt "$START_INDEX" ] && continue
    [ "$idx" -gt "$END_INDEX" ] && continue
    label="${RUN_LABELS[$idx]}"
    prefix="${RUN_PREFIXES[$idx]}"
    tmux_name="${TMUX_NAMES[$idx]}"
    final_step="${FINAL_STEPS[$idx]}"
    CURRENT_IDX="$idx"
    CURRENT_LABEL="$label"
    CURRENT_PREFIX="$prefix"

    ckpt=$(latest_ckpt_dir "$prefix" || true)
    if [ "${DRY_RUN:-0}" != "1" ] && is_run_complete "$prefix" "$final_step"; then
        metrics=$(latest_metrics_file "$ckpt" || true)
        step=$(latest_ckpt_step "$ckpt")
        log "already complete ${label}: prefix=${prefix} step=${step} ckpt=${ckpt} metrics=${metrics}"
        record_item_status "$idx" "$label" "$prefix" "already_complete" "step=${step};ckpt=${ckpt};metrics=${metrics}"
        continue
    fi
    if [ "${DRY_RUN:-0}" != "1" ] && tmux has-session -t "$tmux_name" 2>/dev/null; then
        log "adopting active ${tmux_name}: prefix=${prefix}"
        wait_for_tmux_run "$tmux_name" "$prefix" "$final_step" "$idx" "$label" || continue
        continue
    fi
    if [ -n "$ckpt" ] && [ "$ALLOW_RESUME" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
        log "ERROR existing partial checkpoint and ALLOW_RESUME=0: ${ckpt}"
        record_item_status "$idx" "$label" "$prefix" "blocked_partial_checkpoint" "ckpt=${ckpt}"
        exit 1
    fi
    precheck_disk_or_stop
    launch_container "${RUN_SCRIPTS[$idx]}" "$tmux_name"
    wait_for_tmux_run "$tmux_name" "$prefix" "$final_step" "$idx" "$label" || continue
done

log "kodcode qwen3 1p7b instruct ctx8k stage1 queue complete"
notify "KodCode Qwen3 1.7B Stage1 queue complete" "status_file=${QUEUE_STATUS_FILE}"
