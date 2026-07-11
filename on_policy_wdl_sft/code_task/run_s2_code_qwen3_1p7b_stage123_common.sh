#!/usr/bin/env bash
# Stage2 wrapper for the Stage123 family; all resource values come from one profile.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/qwen3_1p7b_stage123_resource_profile.sh"
: "${STAGE123_RUN_ID:?STAGE123_RUN_ID required}"
source "${SCRIPT_DIR}/stage123_manifest_gate.sh"
stage123_require_preflight_receipt "$STAGE123_RUN_ID"
stage123_print_profile STAGE2
exec bash "${SCRIPT_DIR}/run_s2_code_kodcode_qwen3_1p7b_instruct_ctx8k_p40_common.sh" "$@"
