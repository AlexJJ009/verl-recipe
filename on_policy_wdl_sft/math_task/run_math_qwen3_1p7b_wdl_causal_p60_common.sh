#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAUSAL_WANDB_PROJECT=${WANDB_PROJECT:-OnPolicyWDLSFT-Math-1P7B-Causal-P60}
source "${SCRIPT_DIR}/qwen3_1p7b_math_stage123_resource_profile.sh"

: "${RUN_PREFIX:?RUN_PREFIX required}"
: "${FUSION_LAMBDA:?FUSION_LAMBDA required}"
: "${FUSION_MODE:?FUSION_MODE required}"

export BASE_MODEL_PATH=${BASE_MODEL_PATH:-/data-2/model_weights/math_task/qwen3_1p7b_cold_start_cotmask_v3/candidates/step_20}
export EXPECTED_MODEL1_PATH=${EXPECTED_MODEL1_PATH:-$BASE_MODEL_PATH}
export MODEL2_PATH=${MODEL2_PATH:-/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/launches/20260720T091917Z/artifacts/b0-stage1/final_model}
export STAGE1_MODEL2_PROVENANCE_FILE=${STAGE1_MODEL2_PROVENANCE_FILE:-/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/launches/20260720T091917Z/artifacts/b0-stage1/provenance.json}
export STAGE1_RUN_PREFIX=${STAGE1_RUN_PREFIX:-MATH-B0_STAGE1-QWEN3-1P7B-COTMASK-V3}
export EXPECTED_STAGE1_RUN_PREFIX=${EXPECTED_STAGE1_RUN_PREFIX:-$STAGE1_RUN_PREFIX}
export STAGE1_STEP=${STAGE1_STEP:-40}
export STAGE2_HANDOFF_STEP=${STAGE2_HANDOFF_STEP:-40}
export EXPECTED_STAGE1_BETA=0.0
export ALLOW_EXTERNAL_MODEL2=1
export TRAIN_FILE=${TRAIN_FILE:-/data-1/dataset/math/qwen3_1p7b_stage123_seed20260719/stage1_control_stage2_then_stage3.parquet}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-60}
export WDL_SFT_BETA=0.0
export LOSS_MODE=wdl_sft
export LR=${LR:-1e-6}
export LR_WARMUP_STEPS=0
export DATA_SEED=${DATA_SEED:-20260719}
export DATA_SHUFFLE=False
export JOINT_TRAINING_ROLLOUT_SOURCE=model2
export TRACK_JOINT_SUBMODEL_LOSSES=true
export JOINT_VALIDATION_VIEWS="[model1,model2]"
export PROTECTED_CKPT_STEPS="[20,40,45,50,60]"
export TEST_FREQ=5
export SAVE_FREQ=5
export MAX_ACTOR_CKPTS_TO_KEEP=${MAX_ACTOR_CKPTS_TO_KEEP:-1}
export KEEP_BEST_CKPT=True
export BEST_CKPT_METRIC_KEY=val-core/model2/math7_macro/acc/mean@3
export BEST_CKPT_STRIP_OPTIMIZER=False
export SUBMODEL_KL_ENABLED=false
export SUBMODEL_KL_MODEL1_ENABLED=false
export SUBMODEL_KL_MODEL2_ENABLED=false
export CUSTOM_REWARD_FN_PATH="${SCRIPT_DIR}/../../joint_training/custom_reward_function_latex_verify.py"
export CUSTOM_REWARD_FN_NAME=compute_score_latex_verify
export WANDB_PROJECT=$CAUSAL_WANDB_PROJECT
export CAUSAL_ARTIFACT_ROOT=${CAUSAL_ARTIFACT_ROOT:-/data-2/model_weights/math_task/qwen3_1p7b_wdl_causal_p60}
export LOG_DIR=${LOG_DIR:-$CAUSAL_ARTIFACT_ROOT/logs}
export WDL_MANIPULATION_RECEIPT=${WDL_MANIPULATION_RECEIPT:-$CAUSAL_ARTIFACT_ROOT/admission/manipulation_receipt.json}

for required in "$BASE_MODEL_PATH" "$MODEL2_PATH" "$TRAIN_FILE" "$STAGE1_MODEL2_PROVENANCE_FILE"; do
    if [ ! -e "$required" ]; then
        echo "ERROR: required causal-P60 input missing: $required" >&2
        exit 1
    fi
done
python3 "${SCRIPT_DIR}/../../../scripts/check_math_reward_contract.py" \
    --reward-path "$CUSTOM_REWARD_FN_PATH" \
    --function "$CUSTOM_REWARD_FN_NAME"
python3 - "$WDL_MANIPULATION_RECEIPT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"ERROR: manipulation receipt missing: {path}")
receipt = json.loads(path.read_text())
if receipt.get("status") != "pass" or not all(receipt.get("checks", {}).values()):
    raise SystemExit(f"ERROR: manipulation receipt is not a complete pass: {path}")
print(f"Manipulation receipt PASS: {path}")
PY

exec bash "${SCRIPT_DIR}/run_s2_math_qwen3_1p7b_stage123_common.sh" "$@"
