#!/usr/bin/env bash
# Sequential boxed Stage1 -> fixed merge -> Stage2 matched-beta queue.

set -euo pipefail

REPO_HOST=${REPO_HOST:-/data-1/verl07/verl}
REPO_CONTAINER=${REPO_CONTAINER:-/workspace/verl}
LAUNCHER=${LAUNCHER:-/data-1/verl07/run_train.sh}
DOCKER_IMAGE=${DOCKER_IMAGE:-verl-harness}
CKPT_ROOT=${CKPT_ROOT:-/data-1/checkpoints}
WANDB_ROOT=${WANDB_ROOT:-/data-1/wandb_runs}
MODEL2_ROOT=${MODEL2_ROOT:-/data-1/model_weights/staged_v1/boxed_matched}
METRICS_ROOT=${METRICS_ROOT:-"${REPO_HOST}/recipe/on_policy_wdl_sft/staged_v1/metrics"}
MIN_FREE_GB=${MIN_FREE_GB:-100}
MIN_WANDB_FREE_GB=${MIN_WANDB_FREE_GB:-10}
MIN_MODEL2_FREE_GB=${MIN_MODEL2_FREE_GB:-20}
MAX_GPU_UTIL=${MAX_GPU_UTIL:-50}
ALLOW_RESUME=${ALLOW_RESUME:-0}
ALLOW_OVERWRITE_MERGED_MODEL2=${ALLOW_OVERWRITE_MERGED_MODEL2:-0}
POLL_SEC=${POLL_SEC:-300}
LOG_FILE=${LOG_FILE:-"${REPO_HOST}/recipe/on_policy_wdl_sft/staged_v1/run_boxed_matched_chain_queue.log"}
QUEUE_TMUX=${QUEUE_TMUX:-staged_v1_boxed_matched_chain_queue}
WXPUSHER_NOTIFY=${WXPUSHER_NOTIFY:-1}
WXPUSHER_SCRIPT=${WXPUSHER_SCRIPT:-/root/agent-core/skills/wxpusher-notify/scripts/wxpusher_notify.py}

STAGE1_TOTAL_TRAINING_STEPS=${STAGE1_TOTAL_TRAINING_STEPS:-150}
STAGE2_TOTAL_TRAINING_STEPS=${STAGE2_TOTAL_TRAINING_STEPS:-75}

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

BETA_LABELS=("beta0" "beta01")
BETA_VALUES=("0.0" "0.1")
STAGE1_PREFIXES=(
    "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BOXED-BETA0-V1"
    "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BOXED-BETA01-V1"
)
STAGE1_SCRIPTS=(
    "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/staged_v1/run_s1_beta_0.sh"
    "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/staged_v1/run_s1_beta_01.sh"
)
STAGE1_TMUX_NAMES=(
    "staged_v1_chain_s1_boxed_beta0"
    "staged_v1_chain_s1_boxed_beta01"
)
STAGE2_PREFIXES=(
    "WDL-SFT-STAGED-V1-S2-BOXED-FROM-S1-BETA0-BETA0"
    "WDL-SFT-STAGED-V1-S2-BOXED-FROM-S1-BETA01-BETA01"
)
STAGE2_SCRIPTS=(
    "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/staged_v1/run_s2_from_s1_beta0_beta0.sh"
    "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/staged_v1/run_s2_from_s1_beta01_beta01.sh"
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
    session_count=$(tmux list-sessions -F '#S' 2>/dev/null \
        | grep -Ev "^${QUEUE_TMUX}$" \
        | grep -Ec '(^staged_v1_chain_s[12]_|^staged_v1_s[12]_|^ablation_|^wdl_|^train_)' || true)
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

best_step() {
    local ckpt_dir="$1"
    if [ -f "$ckpt_dir/best_checkpoint.json" ]; then
        python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "$ckpt_dir/best_checkpoint.json"
        return
    fi
    latest_step "$ckpt_dir"
}

is_complete() {
    local prefix="$1" final_step="$2" ckpt_dir step
    ckpt_dir=$(latest_ckpt_dir "$prefix")
    [ -n "$ckpt_dir" ] || return 1
    step=$(latest_step "$ckpt_dir" || true)
    [ -n "$step" ] && [ "$step" -ge "$final_step" ]
}

metrics_path_for_ckpt() {
    local ckpt_dir="$1"
    printf '%s/%s/%s.jsonl\n' "$METRICS_ROOT" "$WANDB_PROJECT" "$(basename "$ckpt_dir")"
}

has_stage2_final_metrics() {
    local prefix="$1" final_step="$2" ckpt_dir metrics_path
    ckpt_dir=$(latest_ckpt_dir "$prefix")
    [ -n "$ckpt_dir" ] || return 1
    metrics_path=$(metrics_path_for_ckpt "$ckpt_dir")
    [ -f "$metrics_path" ] || return 1
    python3 - "$metrics_path" "$final_step" <<'PY'
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

has_merged_model2() {
    local model_dir="$1"
    [ -f "${model_dir}/model.safetensors" ] || [ -f "${model_dir}/model.safetensors.index.json" ]
}

assert_no_unapproved_collision() {
    local prefix="$1" final_step="$2" ckpt_dir
    ckpt_dir=$(latest_ckpt_dir "$prefix" || true)
    [ -z "$ckpt_dir" ] && return
    if is_complete "$prefix" "$final_step"; then
        return
    fi
    if [ "$ALLOW_RESUME" = "1" ]; then
        log "resume allowed for existing incomplete checkpoint: $ckpt_dir"
        return
    fi
    log "ERROR: existing incomplete checkpoint would trigger auto-resume: $ckpt_dir; set ALLOW_RESUME=1 to resume explicitly"
    notify "Boxed WDL-SFT chain blocked" "Status: blocked
What happened: Existing incomplete checkpoint would trigger auto-resume.
Evidence: checkpoint=${ckpt_dir}
Next action: Set ALLOW_RESUME=1 only if resuming is intended."
    exit 1
}

wait_for_resources() {
    while true; do
        local ckpt_free_gb wandb_free_gb model2_free_gb gpu_util
        ckpt_free_gb=$(get_free_gb "$CKPT_ROOT")
        wandb_free_gb=$(get_free_gb "$WANDB_ROOT")
        model2_free_gb=$(get_free_gb "$MODEL2_ROOT")
        gpu_util=$(get_gpu_util_total)
        if [ "$ckpt_free_gb" -ge "$MIN_FREE_GB" ] \
            && [ "$wandb_free_gb" -ge "$MIN_WANDB_FREE_GB" ] \
            && [ "$model2_free_gb" -ge "$MIN_MODEL2_FREE_GB" ] \
            && [ "$gpu_util" -lt "$MAX_GPU_UTIL" ] \
            && has_conflicting_training; then
            log "resources ok: ckpt_free=${ckpt_free_gb}G, wandb_free=${wandb_free_gb}G, model2_free=${model2_free_gb}G, gpu_util_total=${gpu_util}"
            return
        fi
        log "waiting resources: ckpt_free=${ckpt_free_gb}G need>=${MIN_FREE_GB}G; wandb_free=${wandb_free_gb}G need>=${MIN_WANDB_FREE_GB}G; model2_free=${model2_free_gb}G need>=${MIN_MODEL2_FREE_GB}G; gpu_util_total=${gpu_util} need<${MAX_GPU_UTIL}; no_conflict=$(has_conflicting_training && echo yes || echo no); sleep ${POLL_SEC}s"
        sleep "$POLL_SEC"
    done
}

launch_run() {
    local script="$1" tmux_name="$2" final_step="$3" extra_env="$4" host_script launch_log
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
            "TOTAL_TRAINING_STEPS=$final_step VAL_BEFORE_TRAIN=$VAL_BEFORE_TRAIN TEST_FREQ=$TEST_FREQ SAVE_FREQ=$SAVE_FREQ VAL_N=$VAL_N DATA_SEED=$DATA_SEED TRAIN_MAX_SAMPLES=$TRAIN_MAX_SAMPLES WANDB_PROJECT=$WANDB_PROJECT WANDB_MODE=$WANDB_MODE MIN_FREE_GB_FOR_CKPT=$MIN_FREE_GB JOINT_TRAINING_ROLLOUT_SOURCE=$JOINT_TRAINING_ROLLOUT_SOURCE CALCULATE_ENTROPY=$CALCULATE_ENTROPY ROLLOUT_GPU_MEMORY_UTILIZATION=$ROLLOUT_GPU_MEMORY_UTILIZATION TRAIN_PROMPT_BSZ=$TRAIN_PROMPT_BSZ ROLLOUT_N=$ROLLOUT_N TRAIN_PROMPT_MINI_BSZ=$TRAIN_PROMPT_MINI_BSZ ACTOR_PPO_EPOCHS=$ACTOR_PPO_EPOCHS ACTOR_SHUFFLE=$ACTOR_SHUFFLE $extra_env bash $LAUNCHER $script 2>&1 | tee -a $launch_log"
    else
        tmux new-session -d -s "$tmux_name" \
            "docker run --rm --gpus all --ipc=host --shm-size=64g -e TOTAL_TRAINING_STEPS=$final_step -e VAL_BEFORE_TRAIN=$VAL_BEFORE_TRAIN -e TEST_FREQ=$TEST_FREQ -e SAVE_FREQ=$SAVE_FREQ -e VAL_N=$VAL_N -e DATA_SEED=$DATA_SEED -e TRAIN_MAX_SAMPLES=$TRAIN_MAX_SAMPLES -e WANDB_PROJECT=$WANDB_PROJECT -e WANDB_MODE=$WANDB_MODE -e MIN_FREE_GB_FOR_CKPT=$MIN_FREE_GB -e JOINT_TRAINING_ROLLOUT_SOURCE=$JOINT_TRAINING_ROLLOUT_SOURCE -e CALCULATE_ENTROPY=$CALCULATE_ENTROPY -e ROLLOUT_GPU_MEMORY_UTILIZATION=$ROLLOUT_GPU_MEMORY_UTILIZATION -e TRAIN_PROMPT_BSZ=$TRAIN_PROMPT_BSZ -e ROLLOUT_N=$ROLLOUT_N -e TRAIN_PROMPT_MINI_BSZ=$TRAIN_PROMPT_MINI_BSZ -e ACTOR_PPO_EPOCHS=$ACTOR_PPO_EPOCHS -e ACTOR_SHUFFLE=$ACTOR_SHUFFLE -v /data-1:/data-1 -v $REPO_HOST:$REPO_CONTAINER -w $REPO_CONTAINER $DOCKER_IMAGE bash -lc \"$extra_env bash $script\" 2>&1 | tee -a $launch_log"
    fi
    sleep 5
    tmux has-session -t "$tmux_name" 2>/dev/null || {
        log "ERROR: tmux session failed to start: $tmux_name; see $launch_log"
        notify "Boxed WDL-SFT chain launch failed" "Status: failed
What happened: tmux session failed to start.
Evidence: tmux=${tmux_name}; log=${launch_log}
Next action: Inspect the launch log before relaunching."
        exit 1
    }
}

wait_for_checkpoint_completion() {
    local prefix="$1" tmux_name="$2" final_step="$3"
    while true; do
        if is_complete "$prefix" "$final_step"; then
            log "complete: prefix=$prefix final_step=$final_step"
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
            log "ERROR: tmux $tmux_name exited before final checkpoint; latest_step=$step, need=$final_step"
            notify "Boxed WDL-SFT chain run failed" "Status: failed
What happened: ${tmux_name} exited before final checkpoint.
Evidence: prefix=${prefix}; latest_step=${step}; need=${final_step}
Next action: Inspect ${LOG_FILE%.log}_${tmux_name}.log."
            exit 1
        fi
        log "waiting checkpoint: prefix=$prefix latest_step=$step need=$final_step sleep=${POLL_SEC}s"
        sleep "$POLL_SEC"
    done
}

wait_for_stage2_completion() {
    local prefix="$1" tmux_name="$2" final_step="$3"
    while true; do
        if is_complete "$prefix" "$final_step" && has_stage2_final_metrics "$prefix" "$final_step"; then
            log "complete: prefix=$prefix final_step=$final_step with metrics"
            return
        fi

        local ckpt_dir step metrics_state tmux_state metrics_file
        ckpt_dir=$(latest_ckpt_dir "$prefix" || true)
        if [ -n "$ckpt_dir" ]; then
            step=$(latest_step "$ckpt_dir" || echo "none")
            metrics_file=$(metrics_path_for_ckpt "$ckpt_dir")
            if has_stage2_final_metrics "$prefix" "$final_step"; then
                metrics_state="final"
            elif [ -f "$metrics_file" ]; then
                metrics_state="present-no-final"
            else
                metrics_state="missing"
            fi
        else
            step="none"
            metrics_state="missing"
        fi
        if tmux has-session -t "$tmux_name" 2>/dev/null; then
            tmux_state="alive"
        else
            tmux_state="missing"
        fi
        if [ "$tmux_state" = "missing" ]; then
            log "ERROR: tmux $tmux_name exited before final checkpoint+metrics; latest_step=$step, metrics=$metrics_state, need=$final_step"
            notify "Boxed WDL-SFT Stage2 failed" "Status: failed
What happened: ${tmux_name} exited before final checkpoint+metrics.
Evidence: prefix=${prefix}; latest_step=${step}; metrics=${metrics_state}
Next action: Inspect ${LOG_FILE%.log}_${tmux_name}.log."
            exit 1
        fi
        log "waiting Stage2: prefix=$prefix latest_step=$step metrics=$metrics_state sleep=${POLL_SEC}s"
        sleep "$POLL_SEC"
    done
}

merge_stage1_model2() {
    local beta_label="$1" stage1_prefix="$2" merged_dir="$3" ckpt_dir step merge_log
    ckpt_dir=$(latest_ckpt_dir "$stage1_prefix")
    [ -n "$ckpt_dir" ] || { log "ERROR: cannot merge; missing Stage1 checkpoint for $stage1_prefix"; exit 1; }
    step=$(best_step "$ckpt_dir")
    merge_log="${LOG_FILE%.log}_merge_${beta_label}.log"

    log "merging Stage1 best to fixed Model2 dir: beta=${beta_label} ckpt=${ckpt_dir} step=${step} target=${merged_dir}"
    notify "Boxed WDL-SFT merge starting" "Status: started
What happened: Merging Stage1 best checkpoint to fixed Model2 dir for ${beta_label}.
Evidence: checkpoint=${ckpt_dir}; step=${step}; target=${merged_dir}
Next action: Stage2 will start after load check passes."

    mkdir -p "$MODEL2_ROOT"
    if [ -x "$LAUNCHER" ]; then
        REQUIRE_MERGED_MODEL2_PROVENANCE=True \
        ALLOW_OVERWRITE_MERGED_MODEL2="$ALLOW_OVERWRITE_MERGED_MODEL2" \
        STAGE1_RUN_PREFIX="$stage1_prefix" \
        STAGE1_CKPT_DIR="$ckpt_dir" \
        STAGE1_STEP="$step" \
        MERGED_MODEL2_DIR="$merged_dir" \
        bash "$LAUNCHER" "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/staged_v1/merge_stage1_model2_fixed.sh" \
            2>&1 | tee -a "$merge_log"
    else
        docker run --rm --gpus all --ipc=host --shm-size=64g \
            -e REQUIRE_MERGED_MODEL2_PROVENANCE=True \
            -e ALLOW_OVERWRITE_MERGED_MODEL2="$ALLOW_OVERWRITE_MERGED_MODEL2" \
            -e STAGE1_RUN_PREFIX="$stage1_prefix" \
            -e STAGE1_CKPT_DIR="$ckpt_dir" \
            -e STAGE1_STEP="$step" \
            -e MERGED_MODEL2_DIR="$merged_dir" \
            -v /data-1:/data-1 -v "$REPO_HOST:$REPO_CONTAINER" -w "$REPO_CONTAINER" \
            "$DOCKER_IMAGE" bash "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/staged_v1/merge_stage1_model2_fixed.sh" \
            2>&1 | tee -a "$merge_log"
    fi

    if ! has_merged_model2 "$merged_dir"; then
        log "ERROR: merged Model2 weights missing after merge: $merged_dir"
        notify "Boxed WDL-SFT merge failed" "Status: failed
What happened: Merge command finished but Model2 weights are missing.
Evidence: target=${merged_dir}; log=${merge_log}
Next action: Inspect the merge log."
        exit 1
    fi

    notify "Boxed WDL-SFT merge complete" "Status: completed
What happened: Fixed Model2 dir is ready for ${beta_label}.
Evidence: target=${merged_dir}; source=${ckpt_dir}; step=${step}
Next action: Launching matched Stage2."
}

log "boxed matched Stage1->Stage2 chain queue started"
notify "Boxed WDL-SFT chain started" "Status: started
What happened: Starting sequential boxed matched chains: beta=0.0 then beta=0.1.
Evidence: stage1_steps=${STAGE1_TOTAL_TRAINING_STEPS}; stage2_steps=${STAGE2_TOTAL_TRAINING_STEPS}; model2_root=${MODEL2_ROOT}
Next action: The queue will run Stage1, fixed merge, then Stage2 for each beta."

for idx in "${!BETA_LABELS[@]}"; do
    beta_label="${BETA_LABELS[$idx]}"
    beta_value="${BETA_VALUES[$idx]}"
    s1_prefix="${STAGE1_PREFIXES[$idx]}"
    s1_script="${STAGE1_SCRIPTS[$idx]}"
    s1_tmux="${STAGE1_TMUX_NAMES[$idx]}"
    s2_prefix="${STAGE2_PREFIXES[$idx]}"
    s2_script="${STAGE2_SCRIPTS[$idx]}"
    s2_tmux="${STAGE2_TMUX_NAMES[$idx]}"
    merged_dir="${MERGED_MODEL2_DIRS[$idx]}"

    log "chain start: beta=${beta_value}"
    if is_complete "$s1_prefix" "$STAGE1_TOTAL_TRAINING_STEPS"; then
        log "skip Stage1 already complete: $s1_prefix"
    else
        assert_no_unapproved_collision "$s1_prefix" "$STAGE1_TOTAL_TRAINING_STEPS"
        wait_for_resources
        notify "Boxed WDL-SFT Stage1 starting" "Status: started
What happened: Starting Stage1 for beta=${beta_value}.
Evidence: prefix=${s1_prefix}; final_step=${STAGE1_TOTAL_TRAINING_STEPS}
Next action: Merge will run after Stage1 checkpoint completes."
        launch_run "$s1_script" "$s1_tmux" "$STAGE1_TOTAL_TRAINING_STEPS" ""
        wait_for_checkpoint_completion "$s1_prefix" "$s1_tmux" "$STAGE1_TOTAL_TRAINING_STEPS"
        notify "Boxed WDL-SFT Stage1 complete" "Status: completed
What happened: Stage1 completed for beta=${beta_value}.
Evidence: prefix=${s1_prefix}; checkpoint=$(latest_ckpt_dir "$s1_prefix")
Next action: Merging best checkpoint to fixed Model2 dir."
    fi

    wait_for_resources
    merge_stage1_model2 "$beta_label" "$s1_prefix" "$merged_dir"

    if is_complete "$s2_prefix" "$STAGE2_TOTAL_TRAINING_STEPS" && has_stage2_final_metrics "$s2_prefix" "$STAGE2_TOTAL_TRAINING_STEPS"; then
        log "skip Stage2 already complete with metrics: $s2_prefix"
    else
        assert_no_unapproved_collision "$s2_prefix" "$STAGE2_TOTAL_TRAINING_STEPS"
        wait_for_resources
        notify "Boxed WDL-SFT Stage2 starting" "Status: started
What happened: Starting matched Stage2 for beta=${beta_value}.
Evidence: prefix=${s2_prefix}; model2=${merged_dir}; final_step=${STAGE2_TOTAL_TRAINING_STEPS}
Next action: Queue waits for final checkpoint and metrics."
        extra_env="MODEL2_PATH=$merged_dir MERGED_MODEL2_DIR=$merged_dir REQUIRE_MERGED_MODEL2_PROVENANCE=True STAGE1_RUN_PREFIX=$s1_prefix"
        launch_run "$s2_script" "$s2_tmux" "$STAGE2_TOTAL_TRAINING_STEPS" "$extra_env"
        wait_for_stage2_completion "$s2_prefix" "$s2_tmux" "$STAGE2_TOTAL_TRAINING_STEPS"
        notify "Boxed WDL-SFT Stage2 complete" "Status: completed
What happened: Stage2 completed for beta=${beta_value}.
Evidence: prefix=${s2_prefix}; checkpoint=$(latest_ckpt_dir "$s2_prefix"); model2=${merged_dir}
Next action: Queue will continue or finish."
    fi
    log "chain complete: beta=${beta_value}"
done

log "boxed matched Stage1->Stage2 chain queue complete"
notify "Boxed WDL-SFT chain complete" "Status: completed
What happened: Both boxed matched chains completed.
Evidence: beta=0.0 and beta=0.1; log=${LOG_FILE}
Next action: Review Stage1 and Stage2 validation metrics."
