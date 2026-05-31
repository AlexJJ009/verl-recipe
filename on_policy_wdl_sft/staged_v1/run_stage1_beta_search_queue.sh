#!/usr/bin/env bash
# Local host queue for Stage 1 beta search. Run this in tmux.

set -euo pipefail

REPO_HOST=${REPO_HOST:-/data-1/verl07/verl}
REPO_CONTAINER=${REPO_CONTAINER:-/workspace/verl}
LAUNCHER=${LAUNCHER:-/data-1/verl07/run_train.sh}
DOCKER_IMAGE=${DOCKER_IMAGE:-verl-harness}
CKPT_ROOT=${CKPT_ROOT:-/data-1/checkpoints}
WANDB_ROOT=${WANDB_ROOT:-/data-1/wandb_runs}
MIN_FREE_GB=${MIN_FREE_GB:-100}
MIN_WANDB_FREE_GB=${MIN_WANDB_FREE_GB:-10}
MAX_GPU_UTIL=${MAX_GPU_UTIL:-50}
ALLOW_RESUME=${ALLOW_RESUME:-0}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-150}
FINAL_STEP=${FINAL_STEP:-$TOTAL_TRAINING_STEPS}
POLL_SEC=${POLL_SEC:-300}
LOG_FILE=${LOG_FILE:-"${REPO_HOST}/recipe/on_policy_wdl_sft/staged_v1/run_stage1_beta_search_queue.log"}

export TOTAL_TRAINING_STEPS
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export TEST_FREQ=${TEST_FREQ:-5}
export SAVE_FREQ=${SAVE_FREQ:-5}
export VAL_N=${VAL_N:-3}
export DATA_SEED=${DATA_SEED:-20260528}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
export WANDB_PROJECT=${WANDB_PROJECT:-OnPolicySFT-Then-WDLSFT-StagedV1}
export WANDB_MODE=${WANDB_MODE:-offline}

RUN_PREFIXES=(
    "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BOXED-BETA0-V1"
    "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BOXED-BETA01-V1"
)
RUN_SCRIPTS=(
    "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/staged_v1/run_s1_beta_0.sh"
    "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/staged_v1/run_s1_beta_01.sh"
)
TMUX_NAMES=(
    "staged_v1_s1_boxed_beta0"
    "staged_v1_s1_boxed_beta01"
)

log() {
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
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
    # Match the training wrapper guardrail: floor KiB to GiB so the queue does
    # not launch a child that then exits because df -BG rounded up.
    df -Pk "$target" | awk 'NR==2 {print int($4 / 1024 / 1024)}'
}

get_gpu_util_total() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
        | awk '{s+=$1} END {print s+0}'
}

has_conflicting_training() {
    local session_count container_count
    session_count=$(tmux list-sessions -F '#S' 2>/dev/null | grep -Ec '(^staged_v1_s1_beta|^staged_v1_s2_beta|^ablation_|^wdl_|^train_)' || true)
    container_count=$(docker ps --format '{{.Names}} {{.Image}} {{.Command}}' 2>/dev/null \
        | grep -Eci 'verl-harness|main_ppo|run_s1_beta|run_s2_beta|ONPOLICY|WDL-SFT' || true)
    [ "$session_count" -eq 0 ] && [ "$container_count" -eq 0 ]
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

assert_no_unapproved_collision() {
    local prefix="$1"
    local ckpt_dir
    ckpt_dir=$(latest_ckpt_dir "$prefix" || true)
    [ -z "$ckpt_dir" ] && return
    if is_complete "$prefix"; then
        return
    fi
    if [ "$ALLOW_RESUME" = "1" ]; then
        log "resume allowed for existing incomplete checkpoint: $ckpt_dir"
        return
    fi
    log "ERROR: existing incomplete checkpoint would trigger auto-resume: $ckpt_dir; set ALLOW_RESUME=1 to resume explicitly"
    exit 1
}

wait_for_resources() {
    while true; do
        local ckpt_free_gb wandb_free_gb gpu_util
        ckpt_free_gb=$(get_free_gb "$CKPT_ROOT")
        wandb_free_gb=$(get_free_gb "$WANDB_ROOT")
        gpu_util=$(get_gpu_util_total)
        if [ "$ckpt_free_gb" -ge "$MIN_FREE_GB" ] \
            && [ "$wandb_free_gb" -ge "$MIN_WANDB_FREE_GB" ] \
            && [ "$gpu_util" -lt "$MAX_GPU_UTIL" ] \
            && has_conflicting_training; then
            log "resources ok: ckpt_free=${ckpt_free_gb}G, wandb_free=${wandb_free_gb}G, gpu_util_total=${gpu_util}"
            return
        fi
        log "waiting resources: ckpt_free=${ckpt_free_gb}G need>=${MIN_FREE_GB}G; wandb_free=${wandb_free_gb}G need>=${MIN_WANDB_FREE_GB}G; gpu_util_total=${gpu_util} need<${MAX_GPU_UTIL}; no_conflict=$(has_conflicting_training && echo yes || echo no); sleep ${POLL_SEC}s"
        sleep "$POLL_SEC"
    done
}

launch_run() {
    local script="$1" tmux_name="$2" host_script launch_log
    host_script=$(host_script_path "$script")
    launch_log="${LOG_FILE%.log}_${tmux_name}.log"

    [ -f "$host_script" ] || { log "ERROR: missing host script $host_script"; exit 1; }
    if tmux has-session -t "$tmux_name" 2>/dev/null; then
        log "ERROR: tmux session already exists: $tmux_name"
        exit 1
    fi

    log "launching $script in tmux $tmux_name"
    if [ -x "$LAUNCHER" ]; then
        tmux new-session -d -s "$tmux_name" \
            "TOTAL_TRAINING_STEPS=$TOTAL_TRAINING_STEPS VAL_BEFORE_TRAIN=$VAL_BEFORE_TRAIN TEST_FREQ=$TEST_FREQ SAVE_FREQ=$SAVE_FREQ VAL_N=$VAL_N DATA_SEED=$DATA_SEED TRAIN_MAX_SAMPLES=$TRAIN_MAX_SAMPLES WANDB_PROJECT=$WANDB_PROJECT WANDB_MODE=$WANDB_MODE MIN_FREE_GB_FOR_CKPT=$MIN_FREE_GB bash $LAUNCHER $script 2>&1 | tee -a $launch_log"
    else
        tmux new-session -d -s "$tmux_name" \
            "docker run --rm --gpus all --ipc=host --shm-size=64g -e TOTAL_TRAINING_STEPS=$TOTAL_TRAINING_STEPS -e VAL_BEFORE_TRAIN=$VAL_BEFORE_TRAIN -e TEST_FREQ=$TEST_FREQ -e SAVE_FREQ=$SAVE_FREQ -e VAL_N=$VAL_N -e DATA_SEED=$DATA_SEED -e TRAIN_MAX_SAMPLES=$TRAIN_MAX_SAMPLES -e WANDB_PROJECT=$WANDB_PROJECT -e WANDB_MODE=$WANDB_MODE -e MIN_FREE_GB_FOR_CKPT=$MIN_FREE_GB -v /data-1:/data-1 -v $REPO_HOST:$REPO_CONTAINER -w $REPO_CONTAINER $DOCKER_IMAGE bash $script 2>&1 | tee -a $launch_log"
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

log "staged v1 Stage 1 beta queue started"
for idx in "${!RUN_PREFIXES[@]}"; do
    prefix="${RUN_PREFIXES[$idx]}"
    script="${RUN_SCRIPTS[$idx]}"
    tmux_name="${TMUX_NAMES[$idx]}"

    if is_complete "$prefix"; then
        log "skip already complete: $prefix"
        continue
    fi

    assert_no_unapproved_collision "$prefix"
    wait_for_resources
    launch_run "$script" "$tmux_name"
    wait_for_completion "$prefix" "$tmux_name"
done
log "staged v1 Stage 1 beta queue complete"
