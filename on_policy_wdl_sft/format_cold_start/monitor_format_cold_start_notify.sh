#!/usr/bin/env bash
# Thin monitor for format cold-start SFT queue.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export MONITOR_NAME=${MONITOR_NAME:-format_cold_start_sft}
export QUEUE_TMUX=${QUEUE_TMUX:-format_cold_start_queue}
export CKPT_ROOT=${CKPT_ROOT:-/data-1/checkpoints/format_cold_start}
export METRICS_ROOT=${METRICS_ROOT:-"${SCRIPT_DIR}/metrics"}
export WANDB_PROJECT=${WANDB_PROJECT:-FormatColdStartSFT}
export LOG_FILE=${LOG_FILE:-"${SCRIPT_DIR}/monitor_format_cold_start_notify.log"}
export TRAINING_RELEASE_GATE=${TRAINING_RELEASE_GATE:-0}

RUN_PREFIXES=(
  "SFT-FORMAT-COLDSTART-Qwen3-1P7B-CODE-KODCODE-V1"
  "SFT-FORMAT-COLDSTART-Qwen3-1P7B-MATH-V1"
)
TMUX_NAMES=(
  "format_cold_start_sft_code"
  "format_cold_start_sft_math"
)
FINAL_STEPS=(
  "${CODE_TOTAL_TRAINING_STEPS:-120}"
  "${MATH_TOTAL_TRAINING_STEPS:-80}"
)

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/training_queue_monitor.sh"
training_queue_monitor_main "$@"
