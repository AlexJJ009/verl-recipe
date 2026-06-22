#!/usr/bin/env bash
# Host-side code-task Stage1 -> merge -> Stage2 smoke queue.
set -euo pipefail

REPO_HOST=${REPO_HOST:-/data-1/verl07/verl}
REPO_CONTAINER=${REPO_CONTAINER:-/workspace/verl}
LAUNCHER=${LAUNCHER:-/data-1/verl07/run_train.sh}
DOCKER_IMAGE=${DOCKER_IMAGE:-verl-harness:latest}
CKPT_ROOT=${CKPT_ROOT:-/data-1/checkpoints}
MODEL2_ROOT=${MODEL2_ROOT:-/data-1/model_weights/code_task/smoke}
METRICS_ROOT=${METRICS_ROOT:-"${REPO_HOST}/recipe/on_policy_wdl_sft/code_task/metrics"}
EXTRA_METRICS_ROOTS=${EXTRA_METRICS_ROOTS:-"${REPO_HOST}/recipe/on_policy_wdl_sft/staged_v1/metrics"}
WANDB_PROJECT=${WANDB_PROJECT:-OnPolicyWDLSFT-CodeTask}
LOG_FILE=${LOG_FILE:-"${REPO_HOST}/recipe/on_policy_wdl_sft/code_task/run_code_task_smoke_queue.log"}
QUEUE_TMUX=${QUEUE_TMUX:-code_task_smoke_queue}
START_INDEX=${START_INDEX:-0}
END_INDEX=${END_INDEX:-1}
ALLOW_RESUME=${ALLOW_RESUME:-0}
ALLOW_OVERWRITE_MERGED_MODEL2=${ALLOW_OVERWRITE_MERGED_MODEL2:-0}
STAGE2_HANDOFF_STEP=${STAGE2_HANDOFF_STEP:-5}
QUEUE_STATUS_FILE=${QUEUE_STATUS_FILE:-"${LOG_FILE%.log}_status.tsv"}
MIN_FREE_GB=${MIN_FREE_GB:-${MIN_FREE_GB_FOR_CKPT:-30}}
WXPUSHER_NOTIFY=${WXPUSHER_NOTIFY:-1}
WXPUSHER_SCRIPT=${WXPUSHER_SCRIPT:-/root/agent-core/skills/wxpusher-notify/scripts/wxpusher_notify.py}

if [ "${DRY_RUN:-0}" != "1" ] && [ "${ALLOW_G2_TRAINING_SMOKE:-0}" != "1" ]; then
    echo "[code-task queue] ERROR: non-dry-run queue requires explicit ALLOW_G2_TRAINING_SMOKE=1 approval" >&2
    exit 1
fi

QUEUE_MODE=${QUEUE_MODE:-smoke}
if [ "$QUEUE_MODE" = "full" ]; then
    QUEUE_CONTINUE_ON_FAILURE=${QUEUE_CONTINUE_ON_FAILURE:-0}
else
    QUEUE_CONTINUE_ON_FAILURE=${QUEUE_CONTINUE_ON_FAILURE:-1}
fi
if [ "$QUEUE_MODE" = "full" ]; then
    RUN_LABELS=("s1-full-beta0" "s1-full-beta01")
    RUN_SCRIPTS=(
      "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/code_task/run_s1_code_onpolicy_sft_beta_0.sh"
      "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/code_task/run_s1_code_onpolicy_sft_beta_01.sh"
    )
    RUN_PREFIXES=(
      "ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V2"
      "ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V2"
    )
    FINAL_STEPS=(150 150)
    TMUX_NAMES=("code_task_s1_kodcode_beta0" "code_task_s1_kodcode_beta01")
elif [ "$QUEUE_MODE" = "retention" ]; then
    RUN_LABELS=("s1-retention-beta0" "s1-retention-beta01")
    RUN_SCRIPTS=(
      "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/code_task/run_s1_code_onpolicy_sft_beta_0_retention.sh"
      "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/code_task/run_s1_code_onpolicy_sft_beta_01_retention.sh"
    )
    RUN_PREFIXES=(
      "${RETENTION_BETA0_PREFIX:-ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V2-RETENTION}"
      "${RETENTION_BETA01_PREFIX:-ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V2-RETENTION}"
    )
    FINAL_STEPS=(150 150)
    TMUX_NAMES=("code_task_s1_kodcode_beta0_retention" "code_task_s1_kodcode_beta01_retention")
elif [ "$QUEUE_MODE" = "stage2_retention" ]; then
    : "${STAGE2_BETA0_HANDOFF_STEP:?set STAGE2_BETA0_HANDOFF_STEP after selecting the beta=0.0 Stage1 handoff checkpoint}"
    : "${STAGE2_BETA01_HANDOFF_STEP:?set STAGE2_BETA01_HANDOFF_STEP after selecting the beta=0.1 Stage1 handoff checkpoint}"
    STAGE2_TOTAL_TRAINING_STEPS=${STAGE2_TOTAL_TRAINING_STEPS:-40}
    RUN_LABELS=("s2-retention-beta0-beta0" "s2-retention-beta01-beta01")
    RUN_SCRIPTS=(
      "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/code_task/run_s2_code_retention_beta0_beta0.sh"
      "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/code_task/run_s2_code_retention_beta01_beta01.sh"
    )
    RUN_PREFIXES=(
      "${STAGE2_BETA0_PREFIX:-CODE-S2-RETENTION-BETA0-BETA0}"
      "${STAGE2_BETA01_PREFIX:-CODE-S2-RETENTION-BETA01-BETA01}"
    )
    FINAL_STEPS=("$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS")
    TMUX_NAMES=("code_task_s2_retention_beta0_beta0" "code_task_s2_retention_beta01_beta01")
elif [ "$QUEUE_MODE" = "pilot" ]; then
    RUN_LABELS=("s1-pilot" "s2-pilot")
    RUN_SCRIPTS=(
      "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/code_task/run_s1_code_pilot_beta_0.sh"
      "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/code_task/run_s2_code_pilot_beta0_beta0.sh"
    )
    RUN_PREFIXES=("CODE-S1-PILOT-BETA0" "CODE-S2-PILOT-BETA0-BETA0")
    FINAL_STEPS=(40 20)
    TMUX_NAMES=("code_task_s1_pilot_beta0" "code_task_s2_pilot_beta0_beta0")
else
    RUN_LABELS=("s1-smoke" "s2-smoke")
    RUN_SCRIPTS=(
      "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/code_task/run_s1_code_smoke_beta_0.sh"
      "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/code_task/run_s2_code_smoke_beta0_beta0.sh"
    )
    RUN_PREFIXES=("CODE-S1-SMOKE-BETA0" "CODE-S2-SMOKE-BETA0-BETA0")
    FINAL_STEPS=(5 5)
    TMUX_NAMES=("code_task_s1_smoke_beta0" "code_task_s2_smoke_beta0_beta0")
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
    if [ ! -f "$WXPUSHER_SCRIPT" ]; then
        log "WARNING: WxPusher script not found: $WXPUSHER_SCRIPT"
        return
    fi
    if ! python3 "$WXPUSHER_SCRIPT" --title "$title" --body "$body" >>"$LOG_FILE" 2>&1; then
        log "WARNING: WxPusher notification failed: $title"
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

precheck_disk_or_skip() {
    local label="$1" free_gb
    if [ "${DRY_RUN:-0}" = "1" ]; then
        return 0
    fi
    free_gb=$(get_free_gb "$CKPT_ROOT")
    if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
        log "SKIP ${label}: CKPT_ROOT=${CKPT_ROOT} free=${free_gb}G need>=${MIN_FREE_GB}G"
        record_item_status "${CURRENT_IDX:-NA}" "$label" "${CURRENT_PREFIX:-NA}" "skipped_disk" "free=${free_gb}G need>=${MIN_FREE_GB}G"
        notify "Code-task queue item skipped" "Status: skipped
What happened: ${label} was not launched because checkpoint disk space is below the configured threshold.
Evidence: CKPT_ROOT=${CKPT_ROOT}; free=${free_gb}G; need>=${MIN_FREE_GB}G
Next action: Monitor Agent should free/archive disk or lower retention only if the runbook allows it; queue will move on when possible."
        return 1
    fi
    return 0
}

latest_ckpt_dir() { find "$CKPT_ROOT" -maxdepth 1 -type d -name "$1_*" 2>/dev/null | sort | tail -1; }

latest_ckpt_step() {
    local ckpt_dir="$1"
    if [ -z "$ckpt_dir" ] || [ ! -d "$ckpt_dir" ]; then
        echo 0
        return
    fi
    if [ -f "${ckpt_dir}/latest_checkpointed_iteration.txt" ]; then
        local step
        step=$(tr -dc '0-9' < "${ckpt_dir}/latest_checkpointed_iteration.txt")
        echo "${step:-0}"
        return
    fi
    local step
    step=$(find "$ckpt_dir" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null \
        | sed 's/.*global_step_//' | sort -n | tail -1 | awk '{print $1 + 0}')
    echo "${step:-0}"
}

latest_metrics_file() {
    local ckpt_dir="$1" run_name root candidate
    if [ -z "$ckpt_dir" ]; then
        return
    fi
    run_name=$(basename "$ckpt_dir")
    candidate="${METRICS_ROOT}/${WANDB_PROJECT}/${run_name}.jsonl"
    if [ -f "$candidate" ]; then
        printf '%s\n' "$candidate"
        return
    fi
    for root in $EXTRA_METRICS_ROOTS; do
        candidate="${root}/${WANDB_PROJECT}/${run_name}.jsonl"
        if [ -f "$candidate" ]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
}

is_run_complete() {
    local prefix="$1" final_step="$2" ckpt step metrics
    ckpt=$(latest_ckpt_dir "$prefix" || true)
    step=$(latest_ckpt_step "$ckpt")
    metrics=$(latest_metrics_file "$ckpt" || true)
    [ -n "$ckpt" ] && [ "$step" -ge "$final_step" ] && [ -n "$metrics" ]
}

wait_for_tmux_run() {
    local tmux_name="$1" prefix="$2" final_step="$3" idx="$4" label="$5"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        return
    fi
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
        log "ERROR ${tmux_name} exited before final checkpoint+metrics: prefix=${prefix} step=${step} final=${final_step} ckpt=${ckpt:-none} metrics=${metrics:-none}"
        record_item_status "$idx" "$label" "$prefix" "failed" "step=${step};final=${final_step};ckpt=${ckpt:-none};metrics=${metrics:-none}"
        notify "Code-task run stopped early" "Status: failed
What happened: ${tmux_name} exited before final checkpoint and metrics were both available.
Evidence: prefix=${prefix}; latest_step=${step}; final_step=${final_step}; checkpoint=${ckpt:-none}; metrics=${metrics:-none}
Next action: Queue will move to the next item if configured. Monitor Agent must debug and repair before resuming this failed item."
        if [ "$QUEUE_CONTINUE_ON_FAILURE" = "1" ]; then
            return 1
        fi
        exit 1
    fi
    log "completed ${tmux_name}: prefix=${prefix} step=${step} final=${final_step} ckpt=${ckpt} metrics=${metrics}"
    record_item_status "$idx" "$label" "$prefix" "completed" "step=${step};ckpt=${ckpt};metrics=${metrics}"
    notify "Code-task run complete" "Status: completed
What happened: ${prefix} reached final_step=${final_step} and has metrics evidence.
Evidence: checkpoint=${ckpt}; latest_step=${step}; metrics=${metrics}
Next action: Queue will continue to the next item or finish."
    return 0
}

check_merged_model2_collision() {
    local model_dir="$1"
    [ -e "$model_dir" ] || return 0
    [ "${ALLOW_OVERWRITE_MERGED_MODEL2:-0}" = "1" ] && return 0
    if [ -f "${model_dir}/stage1_source.json" ]; then
        log "existing merged Model2 has provenance and will be checked by Stage2 wrapper: $model_dir"
        return 0
    fi
    log "ERROR stale merged Model2 dir exists without code provenance: $model_dir"
    exit 1
}

launch_container() {
    local container_script="$1" host_script="${1/#$REPO_CONTAINER/$REPO_HOST}" tmux_name="$2" extra_env="$3"
    [ -f "$host_script" ] || { log "ERROR missing script: $host_script"; exit 1; }
    if [ "${DRY_RUN:-0}" = "1" ]; then
        log "DRY_RUN would launch ${tmux_name}: ${host_script} extra_env=${extra_env}"
        if [ "${QUEUE_DRY_RUN_VALIDATE_WRAPPERS:-0}" = "1" ]; then
            log "DRY_RUN validating wrapper ${host_script}"
            (cd "$REPO_HOST" && eval "${extra_env} DRY_RUN=1 bash '${host_script}'")
        fi
        return
    fi
    log "launching ${tmux_name}: ${host_script}"
    if [ -x "$LAUNCHER" ]; then
        tmux new-session -d -s "$tmux_name" "cd '$REPO_HOST' && ${extra_env} bash '$LAUNCHER' '$container_script' 2>&1 | tee -a '$LOG_FILE'"
    else
        tmux new-session -d -s "$tmux_name" "docker run --rm --gpus all --ipc=host --network=host --shm-size=64g -v /data-1:/data-1 -v '$REPO_HOST':'$REPO_CONTAINER' -w '$REPO_CONTAINER' '$DOCKER_IMAGE' bash -lc \"${extra_env} bash '$container_script'\" 2>&1 | tee -a '$LOG_FILE'"
    fi
}

log "code-task ${QUEUE_MODE} queue start START_INDEX=${START_INDEX} END_INDEX=${END_INDEX} DRY_RUN=${DRY_RUN:-0} QUEUE_CONTINUE_ON_FAILURE=${QUEUE_CONTINUE_ON_FAILURE} STATUS_FILE=${QUEUE_STATUS_FILE}"
notify "Code-task queue started" "Status: started
What happened: Code-task ${QUEUE_MODE} queue started.
Evidence: range=${START_INDEX}-${END_INDEX}; continue_on_failure=${QUEUE_CONTINUE_ON_FAILURE}; min_free_gb=${MIN_FREE_GB}; status_file=${QUEUE_STATUS_FILE}
Next action: Queue will launch eligible items sequentially."
for idx in "${!RUN_LABELS[@]}"; do
    [ "$idx" -lt "$START_INDEX" ] && continue
    [ "$idx" -gt "$END_INDEX" ] && continue
    prefix="${RUN_PREFIXES[$idx]}"
    label="${RUN_LABELS[$idx]}"
    tmux_name="${TMUX_NAMES[$idx]}"
    final_step="${FINAL_STEPS[$idx]}"
    CURRENT_IDX="$idx"
    CURRENT_PREFIX="$prefix"
    ckpt=$(latest_ckpt_dir "$prefix" || true)
    if [ "${DRY_RUN:-0}" != "1" ] && is_run_complete "$prefix" "$final_step"; then
        metrics=$(latest_metrics_file "$ckpt" || true)
        step=$(latest_ckpt_step "$ckpt")
        log "already complete ${label}: prefix=${prefix} step=${step} final=${final_step} ckpt=${ckpt} metrics=${metrics}"
        record_item_status "$idx" "$label" "$prefix" "already_complete" "step=${step};ckpt=${ckpt};metrics=${metrics}"
        continue
    fi
    if [ "${DRY_RUN:-0}" != "1" ] && tmux has-session -t "$tmux_name" 2>/dev/null; then
        log "adopting active ${tmux_name}: prefix=${prefix} ckpt=${ckpt:-none}"
        wait_for_tmux_run "$tmux_name" "$prefix" "$final_step" "$idx" "$label" || continue
        continue
    fi
    if [ -n "$ckpt" ] && [ "$ALLOW_RESUME" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
        log "FAILED ${label}: existing partial checkpoint and no active tmux: $ckpt"
        record_item_status "$idx" "$label" "$prefix" "failed_partial_checkpoint" "ckpt=${ckpt}"
        notify "Code-task queue blocked" "Status: blocked
What happened: Existing partial checkpoint found with no active run and ALLOW_RESUME is not enabled.
Evidence: prefix=${prefix}; checkpoint=${ckpt}
Next action: Queue will move on if configured. Monitor Agent must debug and repair before resuming this item."
        if [ "$QUEUE_CONTINUE_ON_FAILURE" = "1" ]; then
            continue
        fi
        exit 1
    fi
    if [ "$QUEUE_MODE" = "stage2_retention" ]; then
        if [ "$idx" = "0" ]; then
            s1_prefix="${STAGE2_BETA0_STAGE1_PREFIX:-ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V2-RETENTION-R2}"
            s1_ckpt="${STAGE2_BETA0_STAGE1_CKPT_DIR:-$(latest_ckpt_dir "$s1_prefix" || true)}"
            handoff_step="$STAGE2_BETA0_HANDOFF_STEP"
            train_file="${STAGE2_BETA0_TRAIN_FILE:-/data-1/dataset/code/verl_rl/kodcode_stage2_after_s1_seed20260604_beta0_handoff.parquet}"
            model_label="beta0"
            merged_dir="${STAGE2_BETA0_MERGED_MODEL2_DIR:-${MODEL2_ROOT}/stage2_retention/${model_label}/step_${handoff_step}}"
        else
            s1_prefix="${STAGE2_BETA01_STAGE1_PREFIX:-ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V2-RETENTION-R2}"
            s1_ckpt="${STAGE2_BETA01_STAGE1_CKPT_DIR:-$(latest_ckpt_dir "$s1_prefix" || true)}"
            handoff_step="$STAGE2_BETA01_HANDOFF_STEP"
            train_file="${STAGE2_BETA01_TRAIN_FILE:-/data-1/dataset/code/verl_rl/kodcode_stage2_after_s1_seed20260604_beta01_handoff.parquet}"
            model_label="beta01"
            merged_dir="${STAGE2_BETA01_MERGED_MODEL2_DIR:-${MODEL2_ROOT}/stage2_retention/${model_label}/step_${handoff_step}}"
        fi
        if [ -z "$s1_ckpt" ] && [ "${DRY_RUN:-0}" != "1" ]; then
            log "SKIP ${label}: missing Stage1 checkpoint for ${s1_prefix}"
            record_item_status "$idx" "$label" "$prefix" "skipped_missing_dependency" "required_prefix=${s1_prefix}"
            notify "Code-task Stage2 item skipped" "Status: skipped
What happened: ${label} was not launched because its selected Stage1 checkpoint dependency is missing.
Evidence: required_prefix=${s1_prefix}
Next action: Select or restore the Stage1 handoff checkpoint before launching Stage2."
            continue
        fi
        check_merged_model2_collision "$merged_dir"
        extra="RUN_PREFIX=${prefix} STAGE1_RUN_PREFIX=${s1_prefix} STAGE1_CKPT_DIR=${s1_ckpt:-/missing/stage1} STAGE2_HANDOFF_STEP=${handoff_step} CODE_TRAIN_FILE=${train_file} TRAIN_FILE=${train_file} MODEL2_LABEL=${model_label} MERGED_MODEL2_DIR=${merged_dir} TOTAL_TRAINING_STEPS=${STAGE2_TOTAL_TRAINING_STEPS} ALLOW_OVERWRITE_MERGED_MODEL2=${ALLOW_OVERWRITE_MERGED_MODEL2}"
    elif [ "$QUEUE_MODE" != "full" ] && [ "$QUEUE_MODE" != "retention" ] && [ "$idx" = "1" ]; then
        s1_ckpt=$(latest_ckpt_dir "${RUN_PREFIXES[0]}" || true)
        if [ -z "$s1_ckpt" ] && [ "${DRY_RUN:-0}" != "1" ]; then
            log "SKIP ${label}: missing Stage1 checkpoint for ${RUN_PREFIXES[0]}"
            record_item_status "$idx" "$label" "$prefix" "skipped_missing_dependency" "required_prefix=${RUN_PREFIXES[0]}"
            notify "Code-task queue item skipped" "Status: skipped
What happened: ${label} was not launched because its Stage1 checkpoint dependency is missing.
Evidence: required_prefix=${RUN_PREFIXES[0]}
Next action: Queue will move on. Monitor Agent should inspect why Stage1 did not produce a checkpoint."
            continue
        fi
        merged_dir="${MODEL2_ROOT}/model2-from-s1-${QUEUE_MODE}-step${STAGE2_HANDOFF_STEP}"
        check_merged_model2_collision "$merged_dir"
        extra="RUN_PREFIX=${prefix} STAGE1_CKPT_DIR=${s1_ckpt:-/missing/stage1} STAGE2_HANDOFF_STEP=${STAGE2_HANDOFF_STEP} MERGED_MODEL2_DIR=${merged_dir} ALLOW_OVERWRITE_MERGED_MODEL2=${ALLOW_OVERWRITE_MERGED_MODEL2}"
    else
        extra="RUN_PREFIX=${prefix}"
    fi
    if ! precheck_disk_or_skip "${RUN_LABELS[$idx]}"; then
        continue
    fi
    notify "Code-task run starting" "Status: started
What happened: Launching ${label}.
Evidence: prefix=${prefix}; script=${RUN_SCRIPTS[$idx]}; final_step=${final_step}
Next action: Queue will notify on completion, skip, or failure."
    launch_container "${RUN_SCRIPTS[$idx]}" "$tmux_name" "$extra"
    wait_for_tmux_run "$tmux_name" "$prefix" "$final_step" "$idx" "$label" || continue
done
log "code-task ${QUEUE_MODE} queue complete"
notify "Code-task queue complete" "Status: completed
What happened: Code-task ${QUEUE_MODE} queue reached the end of its item list.
Evidence: log=${LOG_FILE}; status_file=${QUEUE_STATUS_FILE}
Next action: Review queue log, monitor log, and validation metrics before deciding the next experiment."
