#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAUSAL_WANDB_PROJECT=${WANDB_PROJECT:-OnPolicyWDLSFT-Math-1P7B-Causal-P60}
source "${SCRIPT_DIR}/qwen3_1p7b_math_stage123_resource_profile.sh"

: "${RUN_PREFIX:?RUN_PREFIX required}"
: "${FUSION_LAMBDA:?FUSION_LAMBDA required}"
: "${FUSION_MODE:?FUSION_MODE required}"

export DYNPERM_ENABLED=${DYNPERM_ENABLED:-false}
export DYNPERM_RHO=${DYNPERM_RHO:-0.0}
DYNPERM_FORMAL_OVERRIDES=()
case "${DYNPERM_ENABLED,,}" in
    true|1) DYNPERM_ENABLED=true ;;
    false|0) DYNPERM_ENABLED=false ;;
    *) echo "ERROR: DYNPERM_ENABLED must be true or false" >&2; exit 1 ;;
esac
DYNPERM_RHO_AND_TAG="$(python3 - "$DYNPERM_RHO" <<'PY'
import math
import sys

rho = float(sys.argv[1])
if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
    raise SystemExit("ERROR: DYNPERM_RHO must be finite and in [0, 1]")
canonical = format(rho, ".12g")
print(f"{canonical} rho{canonical.replace('.', 'p')}")
PY
)"
read -r DYNPERM_RHO DYNPERM_DOSE_TAG <<<"$DYNPERM_RHO_AND_TAG"
export DYNPERM_ENABLED DYNPERM_RHO DYNPERM_DOSE_TAG
if [ "$DYNPERM_ENABLED" = false ] && [ "$DYNPERM_RHO" != "0" ]; then
    echo "ERROR: DYNPERM_RHO must be 0 when DYNPERM_ENABLED=false" >&2
    exit 1
fi
if [ "$DYNPERM_ENABLED" = true ]; then
    if [ "${TOTAL_TRAINING_STEPS:-60}" != "60" ]; then
        echo "ERROR: formal DynPerm experiments are P60-only" >&2
        exit 1
    fi
fi

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
export PROTECTED_CKPT_STEPS=${PROTECTED_CKPT_STEPS:-"[20,40,45,50,60]"}
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

if [ "$DYNPERM_ENABLED" = true ]; then
    # Apply the formal experiment identity after all legacy C/fixed-M1 defaults
    # so the two DynPerm knobs cannot inherit a colliding project or cache root.
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/run_math_qwen3_1p7b_wdl_dynperm_common.sh"
fi

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

exec bash "${SCRIPT_DIR}/run_s2_math_qwen3_1p7b_stage123_common.sh" "$@" \
    "${DYNPERM_FORMAL_OVERRIDES[@]}" \
    actor_rollout_ref.actor.weak_logit_permutation.enabled="$DYNPERM_ENABLED" \
    actor_rollout_ref.actor.weak_logit_permutation.rho="$DYNPERM_RHO" \
    actor_rollout_ref.actor.weak_logit_permutation.seed=42 \
    actor_rollout_ref.actor.weak_logit_permutation.row_chunk_size=16 \
    actor_rollout_ref.actor.weak_logit_permutation.audit_invariants=true \
    actor_rollout_ref.actor.weak_logit_permutation.audit_frequency=1 \
    actor_rollout_ref.actor.weak_logit_permutation.audit_rows=4 \
    actor_rollout_ref.actor.weak_logit_permutation.entropy_atol=2.0e-6 \
    actor_rollout_ref.actor.weak_logit_permutation.multiset_atol=0.0 \
    actor_rollout_ref.actor.weak_logit_permutation.log_telemetry=true
