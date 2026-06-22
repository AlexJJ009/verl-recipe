#!/usr/bin/env bash
# Host-side code-task Stage1 -> merge -> Stage2 pilot queue.
set -euo pipefail
export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/run_code_task_pilot_queue.log}
export QUEUE_TMUX=${QUEUE_TMUX:-code_task_pilot_queue}
export STAGE2_HANDOFF_STEP=${STAGE2_HANDOFF_STEP:-20}
export QUEUE_MODE=pilot
if [ "${DRY_RUN:-0}" != "1" ] && [ "${ALLOW_G2_TRAINING_SMOKE:-0}" != "1" ]; then
    echo "[code-task pilot queue] ERROR: pilot queue requires explicit approval env ALLOW_G2_TRAINING_SMOKE=1" >&2
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_code_task_smoke_queue.sh" "$@"
