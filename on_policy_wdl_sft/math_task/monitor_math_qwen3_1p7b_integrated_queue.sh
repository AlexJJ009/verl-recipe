#!/usr/bin/env bash
set -euo pipefail
REPO_HOST=${REPO_HOST:-/data-1/code/verl}
EVENT_LOG=${MATH_EVENT_LOG:-/data-2/model_weights/math_task/qwen3_1p7b_integrated_cotmask_v3/events.jsonl}
LEDGER=${MATH_NOTIFICATION_LEDGER:-/data-2/model_weights/math_task/qwen3_1p7b_integrated_cotmask_v3/notification_ledger.jsonl}
QUEUE_SESSION=${MATH_QUEUE_SESSION:-math-qwen3-1p7b-coldstart-cotmask-v3}
POLL_SEC=${POLL_SEC:-60}
exec python3 "${REPO_HOST}/scripts/math_experiment_monitor.py" \
  --event-log "$EVENT_LOG" \
  --ledger "$LEDGER" \
  --queue-session "$QUEUE_SESSION" \
  --poll-seconds "$POLL_SEC"
