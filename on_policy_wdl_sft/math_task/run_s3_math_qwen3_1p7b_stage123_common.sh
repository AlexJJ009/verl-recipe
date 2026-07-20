#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/qwen3_1p7b_math_stage123_resource_profile.sh"
: "${STAGE2_MODEL_PATH:?extracted Stage2 model path required}"
export INIT_MODEL_PATH="$STAGE2_MODEL_PATH"
exec bash "${SCRIPT_DIR}/run_s1_math_qwen3_1p7b_stage123_common.sh" "$@"
