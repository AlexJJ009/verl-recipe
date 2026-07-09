#!/usr/bin/env bash
# DeepCoder Stage1 code-task On-Policy WDL-SFT from Qwen3-4B-Instruct-2507, response 8K.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_PREFIX=${RUN_PREFIX:-ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-DEEPCODER-R8K-S1-BETA01-V1}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.1}

exec bash "${SCRIPT_DIR}/run_s1_code_deepcoder_instruct2507_r8k_beta_0.sh" "$@"
