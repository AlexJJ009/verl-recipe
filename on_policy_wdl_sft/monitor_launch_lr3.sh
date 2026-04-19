#!/usr/bin/env bash
# ==============================================================================
# Monitor: Wait for DPO-4B-Code-M1 pipeline to FULLY finish, then launch
# On-Policy WDL-SFT LR-3 (lr=1e-6) inside the verl-harness Docker container.
#
# Multi-layer safety gates before launching:
#
#   Gate A — DPO pipeline done:
#     A1. Eval done marker exists: ${EVAL_DONE_DPO}
#     A2. Training summary exists: ${TRAIN_SUMMARY_DPO}
#     A3. DPO run log quiescent (mtime older than ${QUIESCENT_SECS}s)
#     A4. No dpo-harness docker container running
#     A5. No accelerate / train_dpo / batch_rollout / eval_code processes alive
#     A6. All 8 GPUs idle (memory.used <= GPU_IDLE_MEM_MIB, util <= GPU_IDLE_UTIL_PCT)
#
#   Gate B — tmux pull-m1 session gone (rollout pipeline exited):
#     B1. tmux session 'pull-m1' no longer exists
#
# Run inside a dedicated tmux session:
#   tmux new-session -d -s monitor-wdl-lr3 \
#     'bash /data-1/verl07/verl/recipe/on_policy_wdl_sft/monitor_launch_lr3.sh'
# ==============================================================================
set -uo pipefail

# --- DPO completion markers ---
CKPT_DPO="/data-1/checkpoints/qwen3-4b-code-m1-dpo-code"
TRAIN_SUMMARY_DPO="${CKPT_DPO}/training_logs/training_summary.json"
EVAL_DONE_DPO="${CKPT_DPO}/eval_code/.done"
RUN_LOG_DPO="/data-1/dpo-experiment/logs/4b-code-m1-dpo/run_4b_code_m1_dpo.log"

# --- WDL-SFT launch target ---
SESSION_WDL="wdl-sft-lr3"
LAUNCH_SCRIPT="/data-1/verl07/run_train.sh"
TRAIN_SCRIPT="/workspace/verl/recipe/on_policy_wdl_sft/run_on_policy_wdl_sft_qwen3_4b_math_lr3.sh"

# --- Thresholds ---
POLL_SECS=60
QUIESCENT_SECS=600        # DPO log must be untouched for 10 min
GPU_IDLE_MEM_MIB=1024     # per-GPU used-memory threshold (MiB)
GPU_IDLE_UTIL_PCT=5        # per-GPU utilization threshold (%)

# --- Helpers ---
log()  { echo "[$(date '+%F %T')] $*"; }
file_age_secs() {
  local f="$1"
  [ -f "$f" ] || { echo "nofile"; return; }
  echo $(( $(date +%s) - $(stat -c %Y "$f") ))
}

check_dpo_done() {
  # A1. Eval done marker
  [ -f "${EVAL_DONE_DPO}" ] || { echo "no_eval_done"; return 1; }
  # A2. Training summary
  [ -f "${TRAIN_SUMMARY_DPO}" ] || { echo "no_train_summary"; return 1; }
  # A3. Log quiescence
  local age; age=$(file_age_secs "${RUN_LOG_DPO}")
  if [ "${age}" = "nofile" ] || [ "${age}" -lt "${QUIESCENT_SECS}" ]; then
    echo "log_still_active(age=${age}s)"; return 1
  fi
  # A4. No DPO docker container
  if docker ps --format '{{.Image}}' 2>/dev/null | grep -q '^dpo-harness'; then
    echo "dpo_container_running"; return 1
  fi
  # A5. No DPO-related processes
  if pgrep -af 'accelerate launch|train_dpo|batch_rollout|eval_code' >/dev/null 2>&1; then
    echo "dpo_procs_alive"; return 1
  fi
  # A6. All GPUs idle
  local idx mem util
  while IFS=',' read -r idx mem util; do
    mem=${mem// /}; util=${util// /}
    if [ "${mem}" -gt "${GPU_IDLE_MEM_MIB}" ] || [ "${util}" -gt "${GPU_IDLE_UTIL_PCT}" ]; then
      echo "gpu${idx}_busy(mem=${mem}MiB,util=${util}%)"; return 1
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
           --format=csv,noheader,nounits)
  echo "ok"; return 0
}

check_pull_session_gone() {
  # B1. tmux session 'pull-m1' should no longer exist
  if tmux has-session -t pull-m1 2>/dev/null; then
    echo "pull-m1_session_alive"; return 1
  fi
  echo "ok"; return 0
}

# ==============================================================================
# Main
# ==============================================================================
log "Monitor started. Poll every ${POLL_SECS}s."
log "Gate A (DPO done)  — eval_done, train_summary, log quiescent ${QUIESCENT_SECS}s, no container, no procs, GPUs idle"
log "Gate B (pull done)  — tmux 'pull-m1' session gone"
log "Target             — tmux '${SESSION_WDL}' : run_train.sh ${TRAIN_SCRIPT}"

# Gate A: DPO pipeline complete
while true; do
  reason=$(check_dpo_done) && break
  log "[Gate A] blocked: ${reason}"
  sleep "${POLL_SECS}"
done
log "[Gate A] PASS — DPO pipeline fully complete and machine idle."

# Gate B: pull-m1 session gone
while true; do
  reason=$(check_pull_session_gone) && break
  log "[Gate B] blocked: ${reason}"
  sleep "${POLL_SECS}"
done
log "[Gate B] PASS — pull-m1 tmux session gone."

# Extra safety: wait 30s after all gates pass for transient cleanup
log "Cooling down 30s..."
sleep 30

# Final GPU check
while true; do
  local_reason=""
  while IFS=',' read -r idx mem util; do
    mem=${mem// /}; util=${util// /}
    if [ "${mem}" -gt "${GPU_IDLE_MEM_MIB}" ] || [ "${util}" -gt "${GPU_IDLE_UTIL_PCT}" ]; then
      local_reason="gpu${idx}_busy(mem=${mem}MiB,util=${util}%)"
      break
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
           --format=csv,noheader,nounits)
  [ -z "${local_reason}" ] && break
  log "[Final GPU check] blocked: ${local_reason}"
  sleep "${POLL_SECS}"
done
log "[Final GPU check] PASS — all GPUs idle."

# Idempotency: refuse to double-launch
if tmux has-session -t "${SESSION_WDL}" 2>/dev/null; then
  log "ABORT — tmux session '${SESSION_WDL}' already exists; refusing to double-launch."
  log "       Inspect with: tmux attach -t ${SESSION_WDL}"
  exit 1
fi

# ==============================================================================
# Launch WDL-SFT LR-3 via run_train.sh (docker run --rm + 8 GPU)
# ==============================================================================
log "LAUNCH — tmux '${SESSION_WDL}' : ${LAUNCH_SCRIPT} ${TRAIN_SCRIPT}"

tmux new-session -d -s "${SESSION_WDL}"
tmux send-keys -t "${SESSION_WDL}" \
  "bash ${LAUNCH_SCRIPT} ${TRAIN_SCRIPT}" C-m

sleep 5
if tmux has-session -t "${SESSION_WDL}" 2>/dev/null; then
  log "LAUNCH — tmux session '${SESSION_WDL}' up. Monitor exiting cleanly."
  log "         Attach with: tmux attach -t ${SESSION_WDL}"
  exit 0
else
  log "LAUNCH — tmux session failed to start."
  exit 1
fi
