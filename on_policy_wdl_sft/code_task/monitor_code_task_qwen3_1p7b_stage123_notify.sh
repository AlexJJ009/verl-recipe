#!/usr/bin/env bash
# Monitor and release the FRAC25/FRAC50 Qwen3-1.7B Stage123 chains.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export STAGE123_MANIFEST=${STAGE123_MANIFEST:-${REPO_ROOT}/recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml}
export STAGE123_MANIFEST_TOOL=${STAGE123_MANIFEST_TOOL:-${REPO_ROOT}/scripts/experiment_manifest.py}
export STAGE123_MANIFEST_PYTHON=${STAGE123_MANIFEST_PYTHON:-python3}
manifest_json=${EXPERIMENT_BATCH_MANIFEST:-}
if [ -z "$manifest_json" ]; then
    manifest_json=$(mktemp)
    trap 'rm -f "$manifest_json"' EXIT
    "$STAGE123_MANIFEST_PYTHON" "$STAGE123_MANIFEST_TOOL" render "$STAGE123_MANIFEST" --format json > "$manifest_json"
fi

manifest_value() { "$STAGE123_MANIFEST_PYTHON" - "$manifest_json" "$1" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))
for key in sys.argv[2].split('.'): value=value[key]
print(value)
PY
}

export MONITOR_NAME=${MONITOR_NAME:-$(manifest_value monitor.name)}
export WANDB_PROJECT=${WANDB_PROJECT:-$(manifest_value release.project)}
export POLL_SEC=${POLL_SEC:-$(manifest_value monitor.poll_seconds)}
export STAGE123_EXECUTION_STATE_ROOT=${STAGE123_EXECUTION_STATE_ROOT:-/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/state}
export NOTIFICATION_LEDGER=${NOTIFICATION_LEDGER:-/data-2/experiment_registry/stage123_notification_events.jsonl}
sender_args=()
if [ "${WXPUSHER_NOTIFY:-0}" = 1 ]; then
    sender_args=(--sender "${REPO_ROOT}/scripts/wxpusher_event_sender.sh")
fi
exec python3 "${REPO_ROOT}/scripts/stage123_manifest_monitor.py" \
    --manifest "$manifest_json" --state-root "$STAGE123_EXECUTION_STATE_ROOT" \
    --poll-seconds "$POLL_SEC" --ledger "$NOTIFICATION_LEDGER" \
    --policy "${REPO_ROOT}/scripts/experiment_notification_policy.py" "${sender_args[@]}"
