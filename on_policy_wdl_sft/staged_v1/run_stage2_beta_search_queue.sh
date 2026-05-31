#!/usr/bin/env bash
# Local host queue for Stage 2 beta search. Run this in tmux after Stage 1 completes.

set -euo pipefail

REPO_HOST=${REPO_HOST:-/data-1/verl07/verl}
REPO_CONTAINER=${REPO_CONTAINER:-/workspace/verl}
LAUNCHER=${LAUNCHER:-/data-1/verl07/run_train.sh}
DOCKER_IMAGE=${DOCKER_IMAGE:-verl-harness}
CKPT_ROOT=${CKPT_ROOT:-/data-1/checkpoints}
MIN_FREE_GB=${MIN_FREE_GB:-160}
MAX_GPU_UTIL=${MAX_GPU_UTIL:-50}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-150}
FINAL_STEP=${FINAL_STEP:-$TOTAL_TRAINING_STEPS}
POLL_SEC=${POLL_SEC:-300}
LOG_FILE=${LOG_FILE:-"${REPO_HOST}/recipe/on_policy_wdl_sft/staged_v1/run_stage2_beta_search_queue.log"}
export TOTAL_TRAINING_STEPS

RUN_PREFIXES=(
    "WDL-SFT-STAGED-V1-Qwen3-4B-MATH-S2-BETA0"
    "WDL-SFT-STAGED-V1-Qwen3-4B-MATH-S2-BETA01"
    "WDL-SFT-STAGED-V1-Qwen3-4B-MATH-S2-BETA02"
    "WDL-SFT-STAGED-V1-Qwen3-4B-MATH-S2-BETA03"
    "WDL-SFT-STAGED-V1-Qwen3-4B-MATH-S2-BETA04"
    "WDL-SFT-STAGED-V1-Qwen3-4B-MATH-S2-BETA05"
)
RUN_SCRIPTS=(
    "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/staged_v1/run_s2_beta_0.sh"
    "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/staged_v1/run_s2_beta_01.sh"
    "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/staged_v1/run_s2_beta_02.sh"
    "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/staged_v1/run_s2_beta_03.sh"
    "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/staged_v1/run_s2_beta_04.sh"
    "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/staged_v1/run_s2_beta_05.sh"
)
TMUX_NAMES=(
    "staged_v1_beta0"
    "staged_v1_beta01"
    "staged_v1_beta02"
    "staged_v1_beta03"
    "staged_v1_beta04"
    "staged_v1_beta05"
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
    local ckpt_dir step
    ckpt_dir=$(latest_ckpt_dir "$prefix")
    [ -n "$ckpt_dir" ] || return 1
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
    local script="$1" tmux_name="$2" host_script launch_log
    host_script=$(host_script_path "$script")
    launch_log="${LOG_FILE%.log}_${tmux_name}.log"

    [ -f "$host_script" ] || { log "ERROR: missing host script $host_script"; exit 1; }
    if tmux has-session -t "$tmux_name" 2>/dev/null; then
        log "tmux session already exists: $tmux_name"
        return
    fi

    log "launching $script in tmux $tmux_name"
    if [ -x "$LAUNCHER" ]; then
        tmux new-session -d -s "$tmux_name" \
            "TOTAL_TRAINING_STEPS=$TOTAL_TRAINING_STEPS bash $LAUNCHER $script 2>&1 | tee -a $launch_log"
    else
        tmux new-session -d -s "$tmux_name" \
            "docker run --rm --gpus all --ipc=host --shm-size=64g -e TOTAL_TRAINING_STEPS=$TOTAL_TRAINING_STEPS -v /data-1:/data-1 -v $REPO_HOST:$REPO_CONTAINER -w $REPO_CONTAINER $DOCKER_IMAGE bash $script 2>&1 | tee -a $launch_log"
    fi
    sleep 5
    tmux has-session -t "$tmux_name" 2>/dev/null || {
        log "ERROR: tmux session failed to start: $tmux_name; see $launch_log"
        exit 1
    }
}

wait_for_completion() {
    local prefix="$1" tmux_name="$2"
    while true; do
        if is_complete "$prefix"; then
            log "complete: prefix=$prefix"
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
        log "waiting completion: prefix=$prefix latest_step=$step sleep=${POLL_SEC}s"
        sleep "$POLL_SEC"
    done
}

log "staged v1 Stage 2 beta queue started"
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
log "staged v1 Stage 2 beta queue complete"
