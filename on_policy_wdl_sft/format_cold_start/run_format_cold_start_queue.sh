#!/usr/bin/env bash
# Host-side queue: prepare format SFT data, run SFT, merge final HF model.
set -euo pipefail

REPO_HOST=${REPO_HOST:-/data-1/verl07/verl}
REPO_CONTAINER=${REPO_CONTAINER:-/workspace/verl}
DOCKER_IMAGE=${DOCKER_IMAGE:-verl-harness:latest}
CKPT_ROOT=${CKPT_ROOT:-/data-1/checkpoints/format_cold_start}
MERGED_ROOT=${MERGED_ROOT:-/data-1/model_weights/format_cold_start}
if [ -z "${DATA_ROOT:-}" ]; then
    if [ "${DRY_RUN:-0}" = "1" ]; then
        DATA_ROOT=/data-1/tmp/verl_agent_scratch/format_cold_start_queue_dry_run
    else
        DATA_ROOT=/data-1/dataset/format_cold_start
    fi
fi
LOG_FILE=${LOG_FILE:-"${REPO_HOST}/recipe/on_policy_wdl_sft/format_cold_start/run_format_cold_start_queue.log"}
QUEUE_STATUS_FILE=${QUEUE_STATUS_FILE:-"${LOG_FILE%.log}_status.tsv"}
QUEUE_POLL_SEC=${QUEUE_POLL_SEC:-60}
START_INDEX=${START_INDEX:-0}
END_INDEX=${END_INDEX:-1}
ALLOW_RESUME=${ALLOW_RESUME:-0}
ALLOW_OVERWRITE_MERGED=${ALLOW_OVERWRITE_MERGED:-0}

RUN_LABELS=("code" "math")
DATA_SCRIPTS=(
  "${REPO_HOST}/recipe/on_policy_wdl_sft/format_cold_start/prepare_code_kodcode_sft_dataset.py"
  "${REPO_HOST}/recipe/on_policy_wdl_sft/format_cold_start/prepare_math_sft_dataset.py"
)
TRAIN_SCRIPTS=(
  "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/format_cold_start/run_sft_code_qwen3_1p7b_kodcode_format.sh"
  "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/format_cold_start/run_sft_math_qwen3_1p7b_format.sh"
)
TRAIN_FILES=(
  "${DATA_ROOT}/code/kodcode_light_sft_messages.parquet"
  "${DATA_ROOT}/math/math_sft_messages.parquet"
)
RUN_PREFIXES=(
  "SFT-FORMAT-COLDSTART-Qwen3-1P7B-CODE-KODCODE-V1"
  "SFT-FORMAT-COLDSTART-Qwen3-1P7B-MATH-V1"
)
TMUX_NAMES=("format_cold_start_sft_code" "format_cold_start_sft_math")
FINAL_STEPS=("${CODE_TOTAL_TRAINING_STEPS:-120}" "${MATH_TOTAL_TRAINING_STEPS:-80}")
MERGED_DIRS=(
  "${MERGED_ROOT}/qwen3-1p7b-kodcode-format-sft"
  "${MERGED_ROOT}/qwen3-1p7b-math-format-sft"
)

if [ "${DRY_RUN:-0}" != "1" ] && [ "${ALLOW_FORMAT_COLD_START_QUEUE:-0}" != "1" ]; then
    echo "[format-queue] ERROR: non-dry-run requires ALLOW_FORMAT_COLD_START_QUEUE=1" >&2
    exit 1
fi

log() { mkdir -p "$(dirname "$LOG_FILE")"; echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

record_status() {
    mkdir -p "$(dirname "$QUEUE_STATUS_FILE")"
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" "$2" "$3" "$4" >>"$QUEUE_STATUS_FILE"
}

latest_run_dir() {
    [ -d "$CKPT_ROOT" ] || return 0
    find "$CKPT_ROOT" -maxdepth 1 -type d -name "$1*" 2>/dev/null | sort | tail -1
}

latest_step() {
    local run_dir="$1"
    [ -n "$run_dir" ] && [ -d "$run_dir" ] || { echo 0; return; }
    find "$run_dir" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1 | awk '{print $1 + 0}'
}

prepare_dataset() {
    local idx="$1" label="$2" output="$3"
    local script="${DATA_SCRIPTS[$idx]}"
    [ -f "$script" ] || { log "ERROR missing data script: $script"; exit 1; }
    log "prepare dataset ${label}: ${output}"
    local args=(--output "$output" --seed "${DATA_SEED:-20260706}")
    if [ -n "${DATA_MAX_SAMPLES:-}" ]; then
        args+=(--max-samples "$DATA_MAX_SAMPLES")
    fi
    python3 "$script" "${args[@]}"
    python3 "$script" --output "$output" --verify-only
}

launch_train() {
    local idx="$1" label="$2" script="${TRAIN_SCRIPTS[$idx]}" tmux_name="${TMUX_NAMES[$idx]}" train_file="${TRAIN_FILES[$idx]}" prefix="${RUN_PREFIXES[$idx]}"
    local host_script="${script/#$REPO_CONTAINER/$REPO_HOST}"
    [ -f "$host_script" ] || { log "ERROR missing train script: $host_script"; exit 1; }
    local run_dir step final
    run_dir=$(latest_run_dir "$prefix")
    step=$(latest_step "$run_dir")
    final="${FINAL_STEPS[$idx]}"
    if [ -n "$run_dir" ] && [ "$step" -ge "$final" ]; then
        log "skip launch ${label}: existing final checkpoint step=${step} final=${final} run_dir=${run_dir}"
        return
    fi
    if [ "${DRY_RUN:-0}" = "1" ]; then
        log "DRY_RUN validate launcher ${label}: ${host_script}"
        DRY_RUN=1 TRAIN_FILE="$train_file" RUN_PREFIX="$prefix" bash "$host_script"
        return
    fi
    if tmux has-session -t "$tmux_name" 2>/dev/null; then
        log "adopt existing training ${label}: ${tmux_name}"
        return
    fi
    log "launch training ${label}: ${tmux_name}"
    tmux new-session -d -s "$tmux_name" \
        "docker run --rm --gpus all --ipc=host --network=host --shm-size=64g -v /data-1:/data-1 -v '$REPO_HOST':'$REPO_CONTAINER' -w '$REPO_CONTAINER' '$DOCKER_IMAGE' bash -lc \"ALLOW_FORMAT_COLD_START_SFT=1 TRAIN_FILE='$train_file' RUN_PREFIX='$prefix' CKPT_ROOT='$CKPT_ROOT' bash '$script'\" 2>&1 | tee -a '$LOG_FILE'"
}

wait_train() {
    local idx="$1" label="$2" tmux_name="${TMUX_NAMES[$idx]}" prefix="${RUN_PREFIXES[$idx]}" final="${FINAL_STEPS[$idx]}"
    [ "${DRY_RUN:-0}" = "1" ] && return 0
    while tmux has-session -t "$tmux_name" 2>/dev/null; do
        local run_dir step
        run_dir=$(latest_run_dir "$prefix")
        step=$(latest_step "$run_dir")
        log "waiting ${label}: step=${step} final=${final} run_dir=${run_dir:-none}"
        sleep "$QUEUE_POLL_SEC"
    done
    local run_dir step
    run_dir=$(latest_run_dir "$prefix")
    step=$(latest_step "$run_dir")
    if [ -z "$run_dir" ] || [ "$step" -lt "$final" ]; then
        log "ERROR ${label} stopped before final checkpoint: step=${step} final=${final} run_dir=${run_dir:-none}"
        record_status "$idx" "$label" failed "step=${step};final=${final};run_dir=${run_dir:-none}"
        exit 1
    fi
    record_status "$idx" "$label" trained "step=${step};run_dir=${run_dir}"
}

merge_train() {
    local idx="$1" label="$2" prefix="${RUN_PREFIXES[$idx]}" target="${MERGED_DIRS[$idx]}" run_dir
    [ "${DRY_RUN:-0}" = "1" ] && { log "DRY_RUN would merge ${label} -> ${target}"; return 0; }
    run_dir=$(latest_run_dir "$prefix")
    [ -n "$run_dir" ] || { log "ERROR missing run dir for merge: $prefix"; exit 1; }
    local overwrite=()
    [ "$ALLOW_OVERWRITE_MERGED" = "1" ] && overwrite+=(--overwrite)
    docker run --rm --ipc=host --network=host --shm-size=64g \
        -v /data-1:/data-1 \
        -v "$REPO_HOST":"$REPO_CONTAINER" \
        -w "$REPO_CONTAINER" \
        "$DOCKER_IMAGE" \
        bash "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/format_cold_start/merge_sft_checkpoint.sh" \
            --checkpoint-dir "$run_dir" \
            --target-dir "$target" \
            "${overwrite[@]}"
    record_status "$idx" "$label" merged "target=${target}"
}

log "format cold-start queue start DRY_RUN=${DRY_RUN:-0} START_INDEX=${START_INDEX} END_INDEX=${END_INDEX}"
for idx in "${!RUN_LABELS[@]}"; do
    [ "$idx" -lt "$START_INDEX" ] && continue
    [ "$idx" -gt "$END_INDEX" ] && continue
    label="${RUN_LABELS[$idx]}"
    prepare_dataset "$idx" "$label" "${TRAIN_FILES[$idx]}"
    launch_train "$idx" "$label"
    wait_train "$idx" "$label"
    merge_train "$idx" "$label"
done
log "format cold-start queue complete; merged roots under ${MERGED_ROOT}"
