#!/usr/bin/env bash
# Queue monitor for the 2H single-model wdl_group_adv_is MATH ablations.
#
# Runs, in order:
#   1. run_2h_math_base.sh
#   2. run_2h_math_sft.sh
#
# Designed to run on the host in a detached tmux session. It launches each
# training script through /data-1/verl07/run_train.sh and waits for the final
# step before starting the next run.

set -euo pipefail

REPO_HOST=${REPO_HOST:-/data-1/verl07/verl}
REPO_CONTAINER=${REPO_CONTAINER:-/workspace/verl}
LAUNCHER=${LAUNCHER:-/data-1/verl07/run_train.sh}
DOCKER_IMAGE=${DOCKER_IMAGE:-verl-harness}
CKPT_ROOT=${CKPT_ROOT:-/data-1/checkpoints}
MIN_FREE_GB=${MIN_FREE_GB:-160}
MAX_GPU_UTIL=${MAX_GPU_UTIL:-50}
FINAL_STEP=${FINAL_STEP:-115}
POLL_SEC=${POLL_SEC:-300}
LOG_FILE=${LOG_FILE:-"${REPO_HOST}/recipe/on_policy_wdl_sft/ablation_single_model/monitor_2h_math_queue.log"}

RUN_PREFIXES=(
    "WDL-GROUP-ADV-IS-Qwen3-4B-MATH-2H-MATHDATA-BASE-E1"
    "WDL-GROUP-ADV-IS-Qwen3-4B-MATH-2H-MATHDATA-SFT-E1"
)
RUN_SCRIPTS=(
    "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/ablation_single_model/run_2h_math_base.sh"
    "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/ablation_single_model/run_2h_math_sft.sh"
)
TMUX_NAMES=(
    "ablation_2h_math_base"
    "ablation_2h_math_sft"
)

log() {
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

get_free_gb() {
    df -BG /data-1 | awk 'NR==2 {sub("G","",$4); print $4}'
}

get_gpu_util_total() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
        | awk '{s+=$1} END {print s+0}'
}

host_script_path() {
    local script="$1"
    printf '%s\n' "${script/#$REPO_CONTAINER/$REPO_HOST}"
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

is_complete() {
    local prefix="$1"
    local ckpt_dir
    ckpt_dir=$(latest_ckpt_dir "$prefix")
    [ -n "$ckpt_dir" ] || return 1
    [ -d "$ckpt_dir/global_step_${FINAL_STEP}" ] && return 0
    local step
    step=$(latest_step "$ckpt_dir" || true)
    [ -n "$step" ] && [ "$step" -ge "$FINAL_STEP" ]
}

wait_for_resources() {
    while true; do
        local free_gb gpu_util
        free_gb=$(get_free_gb)
        gpu_util=$(get_gpu_util_total)
        if [ "$free_gb" -ge "$MIN_FREE_GB" ] && [ "$gpu_util" -lt "$MAX_GPU_UTIL" ]; then
            log "resources ok: /data-1 free=${free_gb}G, gpu_util_total=${gpu_util}"
            return
        fi
        log "waiting resources: /data-1 free=${free_gb}G need>=${MIN_FREE_GB}G; gpu_util_total=${gpu_util} need<${MAX_GPU_UTIL}; sleep ${POLL_SEC}s"
        sleep "$POLL_SEC"
    done
}

launch_run() {
    local script="$1"
    local tmux_name="$2"
    local host_script
    host_script=$(host_script_path "$script")

    [ -f "$host_script" ] || { log "ERROR: missing host script $host_script"; exit 1; }
    if tmux has-session -t "$tmux_name" 2>/dev/null; then
        log "tmux session already exists: $tmux_name; attaching monitor to it"
        return
    fi

    local launch_log="${LOG_FILE%.log}_${tmux_name}.log"
    log "launching $script in tmux $tmux_name"
    if [ -x "$LAUNCHER" ]; then
        tmux new-session -d -s "$tmux_name" \
            "bash $LAUNCHER $script 2>&1 | tee -a $launch_log"
    else
        log "launcher not found/executable ($LAUNCHER); falling back to docker image $DOCKER_IMAGE"
        tmux new-session -d -s "$tmux_name" \
            "docker run --rm --gpus all --ipc=host --shm-size=64g -v /data-1:/data-1 -v $REPO_HOST:$REPO_CONTAINER -w $REPO_CONTAINER $DOCKER_IMAGE bash $script 2>&1 | tee -a $launch_log"
    fi
    sleep 5
    tmux has-session -t "$tmux_name" 2>/dev/null || {
        log "ERROR: tmux session failed to start: $tmux_name; see $launch_log"
        exit 1
    }
}

wait_for_completion() {
    local prefix="$1"
    local tmux_name="$2"

    while true; do
        if is_complete "$prefix"; then
            local ckpt_dir step
            ckpt_dir=$(latest_ckpt_dir "$prefix")
            step=$(latest_step "$ckpt_dir" || echo "$FINAL_STEP")
            log "complete: prefix=$prefix ckpt_dir=$ckpt_dir latest_step=$step"
            return
        fi

        local ckpt_dir step tmux_state
        ckpt_dir=$(latest_ckpt_dir "$prefix" || true)
        if [ -n "$ckpt_dir" ]; then
            step=$(latest_step "$ckpt_dir" || echo "none")
        else
            step="none"
        fi
        if tmux has-session -t "$tmux_name" 2>/dev/null; then
            tmux_state="alive"
        else
            tmux_state="missing"
        fi

        if [ "$tmux_state" = "missing" ]; then
            log "ERROR: tmux $tmux_name exited before final step; latest_step=$step, need=$FINAL_STEP"
            exit 1
        fi
        log "waiting completion: prefix=$prefix tmux=$tmux_state latest_step=$step sleep=${POLL_SEC}s"
        sleep "$POLL_SEC"
    done
}

log "=========================================================="
log "2H MATH queue monitor started (PID $$)"
log "REPO_HOST=$REPO_HOST"
log "CKPT_ROOT=$CKPT_ROOT"
log "FINAL_STEP=$FINAL_STEP"
log "MIN_FREE_GB=$MIN_FREE_GB"
log "MAX_GPU_UTIL=$MAX_GPU_UTIL"
log "LAUNCHER=$LAUNCHER"
log "DOCKER_IMAGE=$DOCKER_IMAGE"
log "=========================================================="

for idx in "${!RUN_PREFIXES[@]}"; do
    prefix="${RUN_PREFIXES[$idx]}"
    script="${RUN_SCRIPTS[$idx]}"
    tmux_name="${TMUX_NAMES[$idx]}"

    if is_complete "$prefix"; then
        log "skip already complete: $prefix"
        continue
    fi

    wait_for_resources
    launch_run "$script" "$tmux_name"
    wait_for_completion "$prefix" "$tmux_name"
done

log "2H MATH queue complete."
