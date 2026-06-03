#!/usr/bin/env bash
# Plateau handoff Stage 1 wrapper, beta=0.1, fixed 50-step source.
set -xeuo pipefail

export RUN_PREFIX=${RUN_PREFIX:-"ONPOLICY-SFT-Qwen3-4B-MATH-S1-PLATEAU-P50-BETA01-V1"}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.1}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-50}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_s1_base_sft.sh" "$@"
