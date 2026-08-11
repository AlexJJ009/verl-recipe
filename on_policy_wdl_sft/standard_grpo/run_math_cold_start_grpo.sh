#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TASK=math PIPELINE=cold_start_grpo
exec bash "${SCRIPT_DIR}/run_qwen3_1p7b_standard_grpo.sh" "$@"
