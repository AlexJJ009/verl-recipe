#!/usr/bin/env bash
# Monitor and release the FRAC25/FRAC50 Qwen3-1.7B Stage123 chains.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export STAGE123_MANIFEST=${STAGE123_MANIFEST:-${REPO_ROOT}/recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml}
export STAGE123_MANIFEST_TOOL=${STAGE123_MANIFEST_TOOL:-${REPO_ROOT}/scripts/experiment_manifest.py}
manifest_json=$(mktemp)
trap 'rm -f "$manifest_json"' EXIT
python3 "$STAGE123_MANIFEST_TOOL" render "$STAGE123_MANIFEST" --format json > "$manifest_json"

export MONITOR_NAME=${MONITOR_NAME:-$(jq -r .monitor.name "$manifest_json")}
export QUEUE_TMUX=${QUEUE_TMUX:-$(jq -r .monitor.queue_tmux "$manifest_json")}
export CKPT_ROOT=${CKPT_ROOT:-/data-1/checkpoints}
export METRICS_ROOT=${METRICS_ROOT:-${SCRIPT_DIR}/metrics}
export EXTRA_METRICS_ROOTS=${EXTRA_METRICS_ROOTS:-${REPO_ROOT}/recipe/on_policy_wdl_sft/staged_v1/metrics}
export WANDB_PROJECT=${WANDB_PROJECT:-$(jq -r .release.project "$manifest_json")}
export POLL_SEC=${POLL_SEC:-$(jq -r .monitor.poll_seconds "$manifest_json")}
export NOTIFICATION_LEDGER=${NOTIFICATION_LEDGER:-/data-2/experiment_registry/stage123_notification_events.jsonl}
sender_args=()
if [ "${WXPUSHER_NOTIFY:-0}" = 1 ]; then
    sender_args=(--sender "${REPO_ROOT}/scripts/wxpusher_event_sender.sh")
fi
exec python3 "${REPO_ROOT}/scripts/stage123_manifest_monitor.py" \
    --manifest "$manifest_json" --checkpoint-root "$CKPT_ROOT" --queue-tmux "$QUEUE_TMUX" \
    --poll-seconds "$POLL_SEC" --ledger "$NOTIFICATION_LEDGER" \
    --policy "${REPO_ROOT}/scripts/experiment_notification_policy.py" "${sender_args[@]}"
