#!/usr/bin/env bash
# Stage 1 single-model On-Policy SFT beta grid wrapper, beta=0.8.
set -xeuo pipefail

export RUN_PREFIX=${RUN_PREFIX:-"ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA08-V1"}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.8}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_s1_base_sft.sh" "$@"
