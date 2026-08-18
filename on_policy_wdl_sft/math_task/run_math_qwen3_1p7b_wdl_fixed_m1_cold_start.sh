#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Same fixed-M1 treatment as the Stage1 arm, starting Model2 directly from CS0.
export RUN_PREFIX=${RUN_PREFIX:-MATH-WDL-FIXED-M1-COLD-START-P60-QWEN3-1P7B}
export FUSION_LAMBDA=0.8
export FUSION_MODE=mixture
export FREEZE_MODEL1=true
export JOINT_VALIDATION_VIEWS="[model1,model2]"
export TRACK_JOINT_SUBMODEL_LOSSES=true
export BEST_CKPT_METRIC_KEY=val-core/model2/math7_macro/acc/mean@3
export WANDB_PROJECT=${WANDB_PROJECT:-OnPolicyWDLSFT-Math-1P7B-Fixed-M1-P60}
export CAUSAL_ARTIFACT_ROOT=${CAUSAL_ARTIFACT_ROOT:-/data-2/model_weights/math_task/qwen3_1p7b_wdl_fixed_m1_p60}
export WDL_MANIPULATION_RECEIPT=${WDL_MANIPULATION_RECEIPT:-/data-2/model_weights/math_task/qwen3_1p7b_wdl_causal_p60/admission/manipulation_receipt.json}
export BASE_MODEL_PATH=${BASE_MODEL_PATH:-/data-2/model_weights/math_task/qwen3_1p7b_cold_start_cotmask_v3/candidates/step_20}
export MODEL2_PATH=${MODEL2_PATH:-$BASE_MODEL_PATH}
export STAGE1_MODEL2_PROVENANCE_FILE=${STAGE1_MODEL2_PROVENANCE_FILE:-$MODEL2_PATH/format_cold_start_source.json}
export STAGE1_RUN_PREFIX=${STAGE1_RUN_PREFIX:-MATH-CS-QWEN3-1P7B-SEED20260719-COTMASK-V3}
export STAGE1_STEP=${STAGE1_STEP:-20}
export MODEL_PATH=${MODEL_PATH:-/data-1/.cache/huggingface/math-wdl-fixed-m1-cold-start-p60}

exec bash "${SCRIPT_DIR}/run_math_qwen3_1p7b_wdl_causal_p60_common.sh" "$@"
