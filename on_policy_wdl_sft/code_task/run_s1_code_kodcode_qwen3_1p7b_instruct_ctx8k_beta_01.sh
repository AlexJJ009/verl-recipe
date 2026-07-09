#!/usr/bin/env bash
# KodCode Stage1 code-task On-Policy WDL-SFT from Qwen3-1.7B chat/instruct model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_PREFIX=${RUN_PREFIX:-ONPOLICY-SFT-Qwen3-1P7B-INSTRUCT-CODE-KODCODE-CTX8K-S1-BETA01-V1}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.1}

exec bash "${SCRIPT_DIR}/run_s1_code_kodcode_qwen3_1p7b_instruct_ctx8k_beta_0.sh" "$@"
