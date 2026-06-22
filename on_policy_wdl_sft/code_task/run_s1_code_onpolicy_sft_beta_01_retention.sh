#!/usr/bin/env bash
# Stage 1 code-task On-Policy WDL-SFT rerun, beta=0.1, retaining plateau handoff checkpoints.
set -euo pipefail

export RUN_PREFIX=${RUN_PREFIX:-ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V2-RETENTION}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_s1_code_onpolicy_sft_beta_0_retention.sh" "$@"
