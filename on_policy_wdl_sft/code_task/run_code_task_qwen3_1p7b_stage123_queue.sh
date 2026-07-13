#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
if [ "${DRY_RUN:-1}" = 1 ]; then
    export STAGE123_EXECUTION_STATE_ROOT=${STAGE123_EXECUTION_STATE_ROOT:-${STAGE123_SCRATCH_ROOT:-/data-1/tmp/verl_agent_scratch/qwen3_1p7b_stage123}/execution-state}
else
    export STAGE123_EXECUTION_STATE_ROOT=${STAGE123_EXECUTION_STATE_ROOT:-/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/state}
fi
export STAGE123_QUEUE_DEADLINE_SECONDS=${STAGE123_QUEUE_DEADLINE_SECONDS:-604800}
command_json=$(python3 - "${SCRIPT_DIR}/run_code_task_qwen3_1p7b_stage123_queue_impl.sh" <<'PY'
import json,sys
print(json.dumps(["bash",sys.argv[1]],separators=(",",":")))
PY
)
args=(queue --run-id stage123-primary-queue --state-root "$STAGE123_EXECUTION_STATE_ROOT" --timeout-seconds "$STAGE123_QUEUE_DEADLINE_SECONDS" --recovery-policy "${REPO_ROOT}/config/experiment_execution/stage123_recovery_policy_v1.json" --command-json "$command_json")
[ "${1:-}" = --resume ] && args+=(--resume)
exec python3 "${REPO_ROOT}/scripts/experiment_execution_core.py" "${args[@]}"
