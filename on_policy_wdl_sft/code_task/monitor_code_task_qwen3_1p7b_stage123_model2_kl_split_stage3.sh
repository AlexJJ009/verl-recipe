#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MANIFEST=${STAGE123_MANIFEST:-${REPO_ROOT}/recipe/on_policy_wdl_sft/experiment_manifest/stage123_model2_kl_split_stage3.yaml}
STATE_ROOT=${STAGE123_EXECUTION_STATE_ROOT:-/data-2/model_weights/code_task/qwen3_1p7b_stage123_model2_kl_split_stage3_v1/state}
POLL_SEC=${POLL_SEC:-60}
LEDGER=${NOTIFICATION_LEDGER:-/data-2/experiment_registry/stage123_model2_kl_split_stage3_events.jsonl}
rendered=$(mktemp)
trap 'rm -f "$rendered"' EXIT
python3 "${REPO_ROOT}/scripts/stage123_matrix_manifest.py" render "$MANIFEST" > "$rendered"
exec python3 "${REPO_ROOT}/scripts/stage123_manifest_monitor.py" \
    --manifest "$rendered" \
    --state-root "$STATE_ROOT" \
    --poll-seconds "$POLL_SEC" \
    --ledger "$LEDGER" \
    --policy "${REPO_ROOT}/scripts/experiment_notification_policy.py"
