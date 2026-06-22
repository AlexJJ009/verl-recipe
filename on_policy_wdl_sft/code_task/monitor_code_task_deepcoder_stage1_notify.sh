#!/usr/bin/env bash
# Thin DeepCoder Stage1 monitor entry point.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QUEUE_MODE=deepcoder_stage1
exec bash "${SCRIPT_DIR}/monitor_code_task_queue_notify.sh" "$@"
