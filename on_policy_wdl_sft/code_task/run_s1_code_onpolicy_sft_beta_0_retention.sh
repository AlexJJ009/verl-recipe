#!/usr/bin/env bash
# Stage 1 code-task On-Policy SFT rerun, beta=0.0, retaining plateau handoff checkpoints.
set -euo pipefail

export RUN_PREFIX=${RUN_PREFIX:-ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V2-RETENTION}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.0}

# Keep the baseline comparison clean: rerun the same 150-step Stage1, but protect
# plateau handoff candidates for later Stage2 launches.
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-150}
export SAVE_FREQ=${SAVE_FREQ:-5}
export TEST_FREQ=${TEST_FREQ:-5}
export PROTECTED_CKPT_STEPS=${PROTECTED_CKPT_STEPS:-[70,80,90,100,110,120]}
export PROTECTED_CKPT_STRIP_OPTIMIZER=${PROTECTED_CKPT_STRIP_OPTIMIZER:-True}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_s1_code_onpolicy_sft_beta_0.sh" "$@"
