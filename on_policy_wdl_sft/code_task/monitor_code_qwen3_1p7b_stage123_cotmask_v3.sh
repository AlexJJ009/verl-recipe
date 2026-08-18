#!/usr/bin/env bash
set -euo pipefail
REPO_HOST=${REPO_HOST:-/data-1/code/verl}
MANIFEST=${CODE_STAGE123_MANIFEST:-${REPO_HOST}/recipe/on_policy_wdl_sft/experiment_manifest/code_qwen3_1p7b_stage123_cotmask_v3.yaml}
EVENT_LOG=${STAGE123_EVENT_LOG:-$(python3 - "$MANIFEST" <<'PY'
import sys,yaml
print(yaml.safe_load(open(sys.argv[1],encoding="utf-8"))["paths"]["event_log"])
PY
)}
LEDGER=${CODE_STAGE123_NOTIFICATION_LEDGER:-$(dirname "$EVENT_LOG")/notification_ledger.jsonl}
QUEUE_SESSION=${CODE_STAGE123_QUEUE_SESSION:-code-qwen3-1p7b-stage123-author-signature-v2-step20}
POLL_SEC=${POLL_SEC:-60}
exec python3 "${REPO_HOST}/scripts/code_stage123_monitor.py" \
    --event-log "$EVENT_LOG" --ledger "$LEDGER" --queue-session "$QUEUE_SESSION" --poll-seconds "$POLL_SEC"
