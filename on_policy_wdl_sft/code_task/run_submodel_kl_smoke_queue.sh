#!/usr/bin/env bash
# Host-side queue for the required submodel-KL Stage2 smoke matrix.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export QUEUE_MODE=submodel_kl_smoke
export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/run_submodel_kl_smoke_queue.log}
export QUEUE_TMUX=${QUEUE_TMUX:-submodel_kl_smoke_queue}
export QUEUE_STATUS_FILE=${QUEUE_STATUS_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/run_submodel_kl_smoke_queue_status.tsv}
export START_INDEX=${START_INDEX:-0}
export END_INDEX=${END_INDEX:-9}
export QUEUE_CONTINUE_ON_FAILURE=${QUEUE_CONTINUE_ON_FAILURE:-0}
export STAGE2_TOTAL_TRAINING_STEPS=${STAGE2_TOTAL_TRAINING_STEPS:-${TOTAL_TRAINING_STEPS:-5}}
export MIN_FREE_GB=${MIN_FREE_GB:-80}
export WXPUSHER_NOTIFY=${WXPUSHER_NOTIFY:-0}

if [ "${DRY_RUN:-0}" != "1" ] && [ "${ALLOW_SUBMODEL_KL_SMOKE:-0}" != "1" ]; then
    echo "[submodel-kl smoke queue] ERROR: non-dry-run requires ALLOW_SUBMODEL_KL_SMOKE=1" >&2
    exit 1
fi
if [ "${ALLOW_SUBMODEL_KL_SMOKE:-0}" = "1" ]; then
    export ALLOW_G2_TRAINING_SMOKE=1
fi

exec bash "${SCRIPT_DIR}/run_code_task_smoke_queue.sh" "$@"
