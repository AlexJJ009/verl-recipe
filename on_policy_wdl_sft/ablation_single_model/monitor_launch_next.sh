#!/usr/bin/env bash
# Monitor 2Z-SFT completion + /data-1 disk, then launch the next ablation run.
#
# Conditions for launch (all must hold):
#   1. Current training docker container has exited.
#   2. Final checkpoint (global_step_300) exists — rules out crash exits.
#   3. /data-1 free space >= MIN_FREE_GB.
#   4. GPU total utilization < 50% (no other job is occupying the box).
#
# Launches:
#   tmux new-session -d -s $NEXT_TMUX_NAME "bash /data-1/verl07/run_train.sh $NEXT_SCRIPT"
#
# Designed to run on the host (not inside the docker container).

set -euo pipefail

NEXT_SCRIPT="${NEXT_SCRIPT:-/workspace/verl/recipe/on_policy_wdl_sft/ablation_single_model/run_2a_sft.sh}"
NEXT_TMUX_NAME="${NEXT_TMUX_NAME:-ablation_2a_sft}"
CURRENT_CKPT_DIR="${CURRENT_CKPT_DIR:-/data-1/checkpoints/MINIRL-Qwen3-4B-MATH-2Z-SFT_1776855436}"
CURRENT_DOCKER="${CURRENT_DOCKER:-brave_gates}"
MIN_FREE_GB="${MIN_FREE_GB:-700}"
FINAL_STEP="${FINAL_STEP:-300}"
POLL_TRAINING_SEC="${POLL_TRAINING_SEC:-300}"
POLL_IDLE_SEC="${POLL_IDLE_SEC:-600}"
LOG_FILE="${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/ablation_single_model/monitor_launch_next.log}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

get_free_gb() {
    df -BG /data-1 | awk 'NR==2 {sub("G","",$4); print $4}'
}

get_gpu_util_total() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
        | awk '{s+=$1} END {print s+0}'
}

latest_step() {
    ls -d "$CURRENT_CKPT_DIR"/global_step_* 2>/dev/null \
        | sed 's/.*global_step_//' | sort -n | tail -1
}

container_running() {
    docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CURRENT_DOCKER"
}

log "=========================================================="
log "Monitor started (PID $$)."
log "  NEXT_SCRIPT        = $NEXT_SCRIPT"
log "  NEXT_TMUX_NAME     = $NEXT_TMUX_NAME"
log "  CURRENT_CKPT_DIR   = $CURRENT_CKPT_DIR"
log "  CURRENT_DOCKER     = $CURRENT_DOCKER"
log "  FINAL_STEP         = $FINAL_STEP"
log "  MIN_FREE_GB        = $MIN_FREE_GB"
log "  POLL_TRAINING_SEC  = $POLL_TRAINING_SEC"
log "  POLL_IDLE_SEC      = $POLL_IDLE_SEC"
log "=========================================================="

# -------- Phase 1: wait for current training to exit --------
while container_running; do
    STEP=$(latest_step || echo "?")
    FREE=$(get_free_gb)
    log "[phase=training-running] container=$CURRENT_DOCKER alive; latest_ckpt=step_${STEP}; /data-1 free=${FREE}G"
    sleep "$POLL_TRAINING_SEC"
done

log "Container '$CURRENT_DOCKER' is no longer running. Verifying clean completion..."

# -------- Phase 2: verify clean exit (final checkpoint present) --------
if [ ! -d "$CURRENT_CKPT_DIR/global_step_${FINAL_STEP}" ]; then
    LAST=$(latest_step || echo "none")
    log "ERROR: expected checkpoint '$CURRENT_CKPT_DIR/global_step_${FINAL_STEP}' not found (latest=${LAST})."
    log "ERROR: 2Z-SFT may have crashed or exited early. Refusing to auto-launch next run."
    log "ACTION: inspect tmux session 'ablation_2z_sft' and the log file, then launch next run manually."
    exit 1
fi
log "OK: $CURRENT_CKPT_DIR/global_step_${FINAL_STEP} exists. 2Z-SFT completed cleanly."

# -------- Phase 3: wait for disk + GPU idle --------
while true; do
    FREE=$(get_free_gb)
    GPU_UTIL=$(get_gpu_util_total)
    if [ "$FREE" -ge "$MIN_FREE_GB" ] && [ "$GPU_UTIL" -lt 50 ]; then
        log "OK: /data-1 free=${FREE}G >= ${MIN_FREE_GB}G; GPU total util=${GPU_UTIL} < 50."
        break
    fi
    log "[phase=waiting-resources] /data-1 free=${FREE}G (need >=${MIN_FREE_GB}G); GPU total util=${GPU_UTIL} (need <50). Sleeping ${POLL_IDLE_SEC}s..."
    sleep "$POLL_IDLE_SEC"
done

# -------- Phase 4: safety checks before launch --------
if tmux has-session -t "$NEXT_TMUX_NAME" 2>/dev/null; then
    log "ERROR: tmux session '$NEXT_TMUX_NAME' already exists. Refusing to stomp. Exiting."
    exit 1
fi

if ! docker image inspect verl-harness >/dev/null 2>&1; then
    log "ERROR: docker image 'verl-harness' not found. Refusing to launch. Exiting."
    exit 1
fi

# Verify the script path is present on host (maps into /workspace/verl inside container)
HOST_SCRIPT_PATH="${NEXT_SCRIPT/#\/workspace\/verl/\/data-1\/verl07\/verl}"
if [ ! -f "$HOST_SCRIPT_PATH" ]; then
    log "ERROR: script not found on host: $HOST_SCRIPT_PATH (from in-container path $NEXT_SCRIPT). Exiting."
    exit 1
fi

# -------- Phase 5: launch --------
log "All conditions met. Launching '$NEXT_SCRIPT' in tmux session '$NEXT_TMUX_NAME'..."
LAUNCH_LOG="${LOG_FILE%.log}_launched.log"
tmux new-session -d -s "$NEXT_TMUX_NAME" \
    "bash /data-1/verl07/run_train.sh $NEXT_SCRIPT 2>&1 | tee -a $LAUNCH_LOG"

sleep 5
if tmux has-session -t "$NEXT_TMUX_NAME" 2>/dev/null; then
    log "SUCCESS: tmux session '$NEXT_TMUX_NAME' created. Attach with: tmux attach -t $NEXT_TMUX_NAME"
    log "Launch stdout: $LAUNCH_LOG"
else
    log "ERROR: tmux session '$NEXT_TMUX_NAME' failed to start. Check $LAUNCH_LOG."
    exit 1
fi

log "Monitor complete. Exiting."
