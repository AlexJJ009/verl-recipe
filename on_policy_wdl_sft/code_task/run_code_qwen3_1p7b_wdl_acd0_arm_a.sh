#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_code_qwen3_1p7b_wdl_acd0_common.sh"
: "${INIT_MODEL_PATH:?INIT_MODEL_PATH required}"
export RUN_PREFIX=${RUN_PREFIX:-CODE-WDL-ACD0-P60-ARM-A-QWEN3-1P7B}
export BEST_CKPT_METRIC_KEY=val-core/code3_macro/acc/mean@3
exec bash "${SCRIPT_DIR}/run_s1_code_base.sh" \
  +trainer.validation_macro_average_sources='[HumanEval+,MBPP+,LiveCodeBench]' \
  +trainer.validation_macro_average_name=code3_macro \
  +trainer.validation_macro_average_metric=acc/mean@3 "$@"
