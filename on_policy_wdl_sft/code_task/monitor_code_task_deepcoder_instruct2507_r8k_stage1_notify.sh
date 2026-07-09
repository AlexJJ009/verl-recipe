#!/usr/bin/env bash
# Thin DeepCoder Instruct-2507 response-8K Stage1 monitor entry point.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QUEUE_MODE=deepcoder_instruct2507_r8k_stage1
exec bash "${SCRIPT_DIR}/monitor_code_task_queue_notify.sh" "$@"
