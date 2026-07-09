#!/usr/bin/env bash
# KodCode Qwen3-1.7B Stage1 beta=0.1 from code format cold-start SFT weights.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_PREFIX=${RUN_PREFIX:-ONPOLICY-SFT-Qwen3-1P7B-COLDSTART-CODE-KODCODE-CTX8K-S1-BETA01-V1}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.1}
export INIT_MODEL_PATH=${INIT_MODEL_PATH:-/data-1/model_weights/format_cold_start/qwen3-1p7b-kodcode-format-sft}
export PROTECTED_CKPT_STEPS=${PROTECTED_CKPT_STEPS:-[40,60,80,100,120,150]}

exec bash "${SCRIPT_DIR}/run_s1_code_kodcode_qwen3_1p7b_instruct_ctx8k_beta_01.sh" "$@"
