#!/usr/bin/env bash
# Local host queue for Stage 2 fast validation. Run this in tmux.

set -euo pipefail

REPO_HOST=${REPO_HOST:-/data-1/verl07/verl}
REPO_CONTAINER=${REPO_CONTAINER:-/workspace/verl}
LAUNCHER=${LAUNCHER:-/data-1/verl07/run_train.sh}
DOCKER_IMAGE=${DOCKER_IMAGE:-verl-harness}
CKPT_ROOT=${CKPT_ROOT:-/data-1/checkpoints}
WANDB_ROOT=${WANDB_ROOT:-/data-1/wandb_runs}
METRICS_ROOT=${METRICS_ROOT:-"${REPO_HOST}/recipe/on_policy_wdl_sft/staged_v1/metrics"}
MIN_FREE_GB=${MIN_FREE_GB:-160}
MIN_WANDB_FREE_GB=${MIN_WANDB_FREE_GB:-10}
MAX_GPU_UTIL=${MAX_GPU_UTIL:-50}
ALLOW_RESUME=${ALLOW_RESUME:-0}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-75}
FINAL_STEP=${FINAL_STEP:-$TOTAL_TRAINING_STEPS}
POLL_SEC=${POLL_SEC:-300}
LOG_FILE=${LOG_FILE:-"${REPO_HOST}/recipe/on_policy_wdl_sft/staged_v1/run_stage2_fast_validation_queue.log"}
WXPUSHER_NOTIFY=${WXPUSHER_NOTIFY:-1}
WXPUSHER_SCRIPT=${WXPUSHER_SCRIPT:-/root/agent-core/skills/wxpusher-notify/scripts/wxpusher_notify.py}

export TOTAL_TRAINING_STEPS
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export TEST_FREQ=${TEST_FREQ:-5}
export SAVE_FREQ=${SAVE_FREQ:-5}
export VAL_N=${VAL_N:-3}
export DATA_SEED=${DATA_SEED:-20260528}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
export WANDB_PROJECT=${WANDB_PROJECT:-OnPolicySFT-Then-WDLSFT-StagedV1}
export WANDB_MODE=${WANDB_MODE:-offline}
export JOINT_TRAINING_ROLLOUT_SOURCE=${JOINT_TRAINING_ROLLOUT_SOURCE:-model2}
export CALCULATE_ENTROPY=${CALCULATE_ENTROPY:-False}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}
export TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-64}
export ROLLOUT_N=${ROLLOUT_N:-8}
export TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ:-$((TRAIN_PROMPT_BSZ * ROLLOUT_N))}
export ACTOR_PPO_EPOCHS=${ACTOR_PPO_EPOCHS:-1}
export ACTOR_SHUFFLE=${ACTOR_SHUFFLE:-false}

RUN_PREFIXES=(
    "WDL-SFT-STAGED-V1-S2-FROM-S1-BETA0-BETA0"
    "WDL-SFT-STAGED-V1-S2-FROM-S1-BETA01-BETA01"
)
RUN_SCRIPTS=(
    "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/staged_v1/run_s2_from_s1_beta0_beta0.sh"
    "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/staged_v1/run_s2_from_s1_beta01_beta01.sh"
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
    if [ "$WXPUSHER_NOTIFY" != "1" ]; then
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

get_gpu_util_total() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
        | awk '{s+=$1} END {print s+0}'
}

has_conflicting_training() {
    local session_count container_count
    session_count=$(tmux list-sessions -F '#S' 2>/dev/null | grep -Ec '(^staged_v1_s2_from_|^staged_v1_s1_|^ablation_|^wdl_|^train_)' || true)
    container_count=$(docker ps --format '{{.Names}} {{.Image}} {{.Command}}' 2>/dev/null \
        | grep -Eci 'verl-harness|main_ppo|run_s[12]_|ONPOLICY|WDL-SFT' || true)
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

metrics_path_for_ckpt() {
    local ckpt_dir="$1"
    printf '%s/%s/%s.jsonl\n' "$METRICS_ROOT" "$WANDB_PROJECT" "$(basename "$ckpt_dir")"
}

has_final_metrics() {
    local prefix="$1" ckpt_dir metrics_path
    ckpt_dir=$(latest_ckpt_dir "$prefix")
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
    notify "Stage2 WDL-SFT launch starting" "Status: started
What happened: Starting Stage2 training for ${tmux_name}.
Evidence: script=${script}; final_step=${FINAL_STEP}
Config: train_prompt_bsz=${TRAIN_PROMPT_BSZ}; rollout_n=${ROLLOUT_N}; train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ}; actor_ppo_epochs=${ACTOR_PPO_EPOCHS}; actor_shuffle=${ACTOR_SHUFFLE}
Next action: The queue monitor will keep watching this run."
    if [ -x "$LAUNCHER" ]; then
        tmux new-session -d -s "$tmux_name" \
            "TOTAL_TRAINING_STEPS=$TOTAL_TRAINING_STEPS VAL_BEFORE_TRAIN=$VAL_BEFORE_TRAIN TEST_FREQ=$TEST_FREQ SAVE_FREQ=$SAVE_FREQ VAL_N=$VAL_N DATA_SEED=$DATA_SEED TRAIN_MAX_SAMPLES=$TRAIN_MAX_SAMPLES WANDB_PROJECT=$WANDB_PROJECT WANDB_MODE=$WANDB_MODE MIN_FREE_GB_FOR_CKPT=$MIN_FREE_GB JOINT_TRAINING_ROLLOUT_SOURCE=$JOINT_TRAINING_ROLLOUT_SOURCE CALCULATE_ENTROPY=$CALCULATE_ENTROPY ROLLOUT_GPU_MEMORY_UTILIZATION=$ROLLOUT_GPU_MEMORY_UTILIZATION TRAIN_PROMPT_BSZ=$TRAIN_PROMPT_BSZ ROLLOUT_N=$ROLLOUT_N TRAIN_PROMPT_MINI_BSZ=$TRAIN_PROMPT_MINI_BSZ ACTOR_PPO_EPOCHS=$ACTOR_PPO_EPOCHS ACTOR_SHUFFLE=$ACTOR_SHUFFLE bash $LAUNCHER $script 2>&1 | tee -a $launch_log"
    else
        tmux new-session -d -s "$tmux_name" \
            "docker run --rm --gpus all --ipc=host --shm-size=64g -e TOTAL_TRAINING_STEPS=$TOTAL_TRAINING_STEPS -e VAL_BEFORE_TRAIN=$VAL_BEFORE_TRAIN -e TEST_FREQ=$TEST_FREQ -e SAVE_FREQ=$SAVE_FREQ -e VAL_N=$VAL_N -e DATA_SEED=$DATA_SEED -e TRAIN_MAX_SAMPLES=$TRAIN_MAX_SAMPLES -e WANDB_PROJECT=$WANDB_PROJECT -e WANDB_MODE=$WANDB_MODE -e MIN_FREE_GB_FOR_CKPT=$MIN_FREE_GB -e JOINT_TRAINING_ROLLOUT_SOURCE=$JOINT_TRAINING_ROLLOUT_SOURCE -e CALCULATE_ENTROPY=$CALCULATE_ENTROPY -e ROLLOUT_GPU_MEMORY_UTILIZATION=$ROLLOUT_GPU_MEMORY_UTILIZATION -e TRAIN_PROMPT_BSZ=$TRAIN_PROMPT_BSZ -e ROLLOUT_N=$ROLLOUT_N -e TRAIN_PROMPT_MINI_BSZ=$TRAIN_PROMPT_MINI_BSZ -e ACTOR_PPO_EPOCHS=$ACTOR_PPO_EPOCHS -e ACTOR_SHUFFLE=$ACTOR_SHUFFLE -v /data-1:/data-1 -v $REPO_HOST:$REPO_CONTAINER -w $REPO_CONTAINER $DOCKER_IMAGE bash $script 2>&1 | tee -a $launch_log"
    fi
    sleep 5
    tmux has-session -t "$tmux_name" 2>/dev/null || {
        log "ERROR: tmux session failed to start: $tmux_name; see $launch_log"
        notify "Stage2 WDL-SFT launch failed" "Status: failed
What happened: tmux session failed to start for ${tmux_name}.
Evidence: launch_log=${launch_log}
Next action: Inspect the launch log before relaunching."
        exit 1
    }
    notify "Stage2 WDL-SFT launch complete" "Status: started
What happened: Stage2 training tmux session is alive for ${tmux_name}.
Evidence: launch_log=${launch_log}
Next action: Waiting for final_step=${FINAL_STEP}."
}

wait_for_completion() {
    local prefix="$1" tmux_name="$2"
    while true; do
        if is_complete "$prefix" && has_final_metrics "$prefix"; then
            log "complete: prefix=$prefix"
            local ckpt_dir
            ckpt_dir=$(latest_ckpt_dir "$prefix" || true)
            notify "Stage2 WDL-SFT run complete" "Status: completed
What happened: ${prefix} reached final_step=${FINAL_STEP}.
Evidence: checkpoint=${ckpt_dir}
Next action: Queue will continue to the next run or finish."
            return
        fi

        local ckpt_dir step tmux_state
        ckpt_dir=$(latest_ckpt_dir "$prefix" || true)
        if [ -n "$ckpt_dir" ]; then
            step=$(latest_step "$ckpt_dir" || echo "none")
        else
            step="none"
        fi
        metrics_state="missing"
        if [ -n "$ckpt_dir" ]; then
            metrics_file=$(metrics_path_for_ckpt "$ckpt_dir")
            if has_final_metrics "$prefix"; then
                metrics_state="final"
            elif [ -f "$metrics_file" ]; then
                metrics_state="present-no-final"
            fi
        fi
        if tmux has-session -t "$tmux_name" 2>/dev/null; then
            tmux_state="alive"
        else
            tmux_state="missing"
        fi

        if [ "$tmux_state" = "missing" ]; then
            log "ERROR: tmux $tmux_name exited before final checkpoint+metrics; latest_step=$step, metrics=${metrics_state}, need=$FINAL_STEP"
            notify "Stage2 WDL-SFT run failed" "Status: failed
What happened: ${tmux_name} exited before final checkpoint+metrics were verified for final_step=${FINAL_STEP}.
Evidence: latest_step=${step}; metrics=${metrics_state}; checkpoint=${ckpt_dir:-none}
Next action: Inspect ${LOG_FILE%.log}_${tmux_name}.log."
            exit 1
        fi
        log "waiting completion: prefix=$prefix latest_step=$step metrics=${metrics_state} sleep=${POLL_SEC}s"
        sleep "$POLL_SEC"
    done
}

log "staged v1 Stage 2 fast-validation queue started"
notify "Stage2 WDL-SFT queue started" "Status: started
What happened: Stage2 fast-validation queue started.
Evidence: runs=${RUN_PREFIXES[*]}; final_step=${FINAL_STEP}
Config: train_prompt_bsz=${TRAIN_PROMPT_BSZ}; rollout_n=${ROLLOUT_N}; train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ}; actor_ppo_epochs=${ACTOR_PPO_EPOCHS}; actor_shuffle=${ACTOR_SHUFFLE}
Next action: The queue will run candidates sequentially."
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
log "staged v1 Stage 2 fast-validation queue complete"
notify "Stage2 WDL-SFT queue complete" "Status: completed
What happened: All Stage2 fast-validation queue runs completed.
Evidence: final_step=${FINAL_STEP}; log=${LOG_FILE}
Next action: Review best checkpoint metrics against Stage1 baselines."
