#!/usr/bin/env bash
# DeepCoder Stage1 code-task On-Policy WDL-SFT, beta=0.1, dense handoff retention.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_PREFIX=${RUN_PREFIX:-ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA01-V1-RETENTION}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.1}

exec bash "${SCRIPT_DIR}/run_s1_code_deepcoder_beta_0_retention.sh" "$@"
