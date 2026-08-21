#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Matched to causal arm C. Only the Model1 update state and run/cache identity differ.
export RUN_PREFIX=${RUN_PREFIX:-MATH-WDL-FIXED-M1-STAGE1-P60-QWEN3-1P7B}
export FUSION_LAMBDA=0.8
export FUSION_MODE=mixture
export FREEZE_MODEL1=true
export WDL_ARM_ID=fixed-m1-stage1
export JOINT_VALIDATION_VIEWS="[model1,model2]"
export TRACK_JOINT_SUBMODEL_LOSSES=true
export BEST_CKPT_METRIC_KEY=val-core/model2/math7_macro/acc/mean@3
export WANDB_PROJECT=${WANDB_PROJECT:-OnPolicyWDLSFT-Math-1P7B-Fixed-M1-P60}
export CAUSAL_ARTIFACT_ROOT=${CAUSAL_ARTIFACT_ROOT:-/data-2/model_weights/math_task/qwen3_1p7b_wdl_fixed_m1_p60}
export WDL_MANIPULATION_RECEIPT=${WDL_MANIPULATION_RECEIPT:-/data-2/model_weights/math_task/qwen3_1p7b_wdl_causal_p60/admission/manipulation_receipt.json}
export MODEL2_PATH=${MODEL2_PATH:-/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/restored_from_causal_p60_joint_20260812/final_model}
export STAGE1_MODEL2_PROVENANCE_FILE=${STAGE1_MODEL2_PROVENANCE_FILE:-$MODEL2_PATH/model_input_provenance.json}
export MODEL_PATH=${MODEL_PATH:-/data-1/.cache/huggingface/math-wdl-fixed-m1-stage1-p60}

exec bash "${SCRIPT_DIR}/run_math_qwen3_1p7b_wdl_causal_p60_common.sh" "$@"
