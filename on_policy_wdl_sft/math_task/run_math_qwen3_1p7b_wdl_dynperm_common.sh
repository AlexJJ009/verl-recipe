#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${DYNPERM_RHO:?DYNPERM_RHO must be 0.0 or 1.0}"
: "${TOTAL_TRAINING_STEPS:?TOTAL_TRAINING_STEPS must be 20, 30, or 60}"

case "$DYNPERM_RHO" in
    0|0.0) DYNPERM_RHO=0.0; DYNPERM_DOSE=rho000 ;;
    1|1.0) DYNPERM_RHO=1.0; DYNPERM_DOSE=rho100 ;;
    *) echo "ERROR: canonical DynPerm endpoint requires DYNPERM_RHO=0.0 or 1.0" >&2; exit 1 ;;
esac
case "$TOTAL_TRAINING_STEPS" in
    20) PROTECTED_CKPT_STEPS="[20]" ;;
    30) PROTECTED_CKPT_STEPS="[20,30]" ;;
    60) PROTECTED_CKPT_STEPS="[20,40,45,50,60]" ;;
    *) echo "ERROR: DynPerm horizon must be P20, P30, or P60" >&2; exit 1 ;;
esac

# Exact C-joint contract. The only scientific treatment is the weak-logit
# permutation block appended as the final Hydra overrides below.
export FUSION_LAMBDA=0.8
export FUSION_MODE=mixture
export FREEZE_MODEL1=false
export BASE_MODEL_PATH=/data-2/model_weights/math_task/qwen3_1p7b_cold_start_cotmask_v3/candidates/step_20
export EXPECTED_MODEL1_PATH="$BASE_MODEL_PATH"
export MODEL2_PATH=/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/restored_from_causal_p60_joint_20260812/final_model
export STAGE1_MODEL2_PROVENANCE_FILE="$MODEL2_PATH/model_input_provenance.json"
export STAGE1_RUN_PREFIX=MATH-B0_STAGE1-QWEN3-1P7B-COTMASK-V3
export EXPECTED_STAGE1_RUN_PREFIX="$STAGE1_RUN_PREFIX"
export STAGE1_STEP=40
export STAGE2_HANDOFF_STEP=40
export TRAIN_FILE=/data-1/dataset/math/qwen3_1p7b_stage123_seed20260719/stage1_control_stage2_then_stage3.parquet
export WDL_SFT_BETA=0.0
export LOSS_MODE=wdl_sft
export LR=1e-6
export LR_WARMUP_STEPS=0
export DATA_SEED=20260719
export DATA_SHUFFLE=False
export JOINT_TRAINING_ROLLOUT_SOURCE=model2
export TRACK_JOINT_SUBMODEL_LOSSES=true
export JOINT_VALIDATION_VIEWS="[model1,model2]"
export TEST_FREQ=5
export SAVE_FREQ=5
export MAX_ACTOR_CKPTS_TO_KEEP=1
export KEEP_BEST_CKPT=True
export BEST_CKPT_METRIC_KEY=val-core/model2/math7_macro/acc/mean@3
export BEST_CKPT_STRIP_OPTIMIZER=False
export SUBMODEL_KL_ENABLED=false
export SUBMODEL_KL_MODEL1_ENABLED=false
export SUBMODEL_KL_MODEL2_ENABLED=false
export MAX_PROMPT_LENGTH=500
export MAX_RESPONSE_LENGTH=4096
export ROLLOUT_MAX_MODEL_LEN=4596
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=32768
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4596
export ACTOR_PPO_MAX_TOKEN_LEN=4596
export GENERATION_MICRO_BATCH_SIZE=32
export LOG_PROB_MICRO_BATCH_SIZE=8
export REF_LOG_PROB_MICRO_BATCH_SIZE=1
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.55
export ACTOR_CALCULATE_ENTROPY=False
export CALCULATE_ENTROPY=False
export ROLLOUT_TP_SIZE=1
export TRAIN_PROMPT_BSZ=64
export ROLLOUT_N=8
export TRAIN_PROMPT_MINI_BSZ=512
export ACTOR_PPO_EPOCHS=1
export ACTOR_SHUFFLE=false
export VAL_N=3
export VAL_TEMPERATURE=0.2
export VAL_TOP_P=0.95
export VAL_DO_SAMPLE=True
export VAL_BEFORE_TRAIN=True
export NGPUS_PER_NODE=8
export NNODES=1
export MATH7_VALIDATION_ROOT=/data-1/dataset/math/qwen3_1p7b_math7_validation_v1
export TEST_FILES="['${MATH7_VALIDATION_ROOT}/aime-2025_with_system_prompt_schema_aligned.parquet','${MATH7_VALIDATION_ROOT}/math500-test_with_system_prompt_schema_aligned.parquet','${MATH7_VALIDATION_ROOT}/amc23-test_with_system_prompt_schema_aligned.parquet','${MATH7_VALIDATION_ROOT}/aqua-test_with_system_prompt_schema_aligned.parquet','${MATH7_VALIDATION_ROOT}/gsm8k-test_with_system_prompt_schema_aligned.parquet','${MATH7_VALIDATION_ROOT}/mawps-test_with_system_prompt_schema_aligned.parquet','${MATH7_VALIDATION_ROOT}/svamp-test_with_system_prompt_schema_aligned.parquet']"
export MATH7_MACRO_SOURCES="[aime25,HuggingFaceH4/MATH-500,zwhe99/amc23,deepmind/aqua_rat,openai/gsm8k,mwpt5/MAWPS,ChilleD/SVAMP]"
export WANDB_MODE=offline
export RUN_PREFIX="MATH-WDL-DYNPERM-${DYNPERM_DOSE^^}-P${TOTAL_TRAINING_STEPS}-QWEN3-1P7B"
export WANDB_PROJECT=OnPolicyWDLSFT-Math-1P7B-DynPerm
export CAUSAL_ARTIFACT_ROOT="/data-2/model_weights/math_task/qwen3_1p7b_wdl_dynperm/${DYNPERM_DOSE}-p${TOTAL_TRAINING_STEPS}"
export WDL_MANIPULATION_RECEIPT=/data-2/model_weights/math_task/qwen3_1p7b_wdl_causal_p60/admission/manipulation_receipt.json
export MODEL_PATH="/data-1/.cache/huggingface/math-wdl-dynperm-${DYNPERM_DOSE}-p${TOTAL_TRAINING_STEPS}"
export PROTECTED_CKPT_STEPS

ENGINEERING_RECEIPT=/data-1/code/_artifacts/verl-v0.7/linear-gon-34-dynperm-mvp/slurm-job-146/runtime-output/gpu_fsdp_smoke_receipt.json
ENGINEERING_RECEIPT_SHA256=3c757ffe6eaed509019bc8fd1b338ed8ab8244803f1f7d32510d7b7fc1eb89a2
ENGINEERING_CANDIDATE_SHA=686f3ee1f190387581e38847cb0e75f055021caa
python3 - "$ENGINEERING_RECEIPT" "$ENGINEERING_RECEIPT_SHA256" "$ENGINEERING_CANDIDATE_SHA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_sha = sys.argv[2]
expected_candidate = sys.argv[3]
if not path.is_file():
    raise SystemExit(f"ERROR: candidate-bound engineering receipt missing: {path}")
raw = path.read_bytes()
if hashlib.sha256(raw).hexdigest() != expected_sha:
    raise SystemExit(f"ERROR: engineering receipt hash mismatch: {path}")
receipt = json.loads(raw)
if receipt.get("result") != "PASS" or receipt.get("world_size") != 8:
    raise SystemExit(f"ERROR: engineering receipt is not the admitted 8-GPU PASS: {path}")
if receipt.get("candidate_sha") != expected_candidate:
    raise SystemExit(f"ERROR: engineering receipt candidate mismatch: {path}")
if receipt.get("formal_experiment") is not False:
    raise SystemExit(f"ERROR: engineering evidence boundary changed: {path}")
print(f"DynPerm engineering receipt PASS: {path}")
PY

# Dry-run validates sources/config without granting a formal launch. Real runs
# require a separate, exact-horizon human admission receipt.
if [ "${STAGE2_DRY_RUN:-0}" != "1" ]; then
    : "${DYNPERM_LAUNCH_RECEIPT:?real DynPerm runs require DYNPERM_LAUNCH_RECEIPT}"
    PARENT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
    RECIPE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
    if [ -n "$(git -C "$PARENT_ROOT" status --porcelain)" ] \
        || [ -n "$(git -C "$RECIPE_ROOT" status --porcelain)" ]; then
        echo "ERROR: formal DynPerm launch requires clean parent and recipe worktrees" >&2
        exit 1
    fi
    PARENT_SHA="$(git -C "$PARENT_ROOT" rev-parse HEAD)"
    RECIPE_SHA="$(git -C "$RECIPE_ROOT" rev-parse HEAD)"
    python3 - "$DYNPERM_LAUNCH_RECEIPT" "$DYNPERM_RHO" "$TOTAL_TRAINING_STEPS" "$PARENT_SHA" "$RECIPE_SHA" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
rho = float(sys.argv[2])
horizon = int(sys.argv[3])
parent_sha = sys.argv[4]
recipe_sha = sys.argv[5]
if not path.is_file():
    raise SystemExit(f"ERROR: launch receipt missing: {path}")
receipt = json.loads(path.read_text())
expected = {
    "status": "authorized",
    "experiment_id": "math_qwen3_1p7b_wdl_dynperm",
    "rho": rho,
    "max_training_steps": horizon,
    "parent_candidate_sha": parent_sha,
    "recipe_candidate_sha": recipe_sha,
}
for key, value in expected.items():
    if receipt.get(key) != value:
        raise SystemExit(f"ERROR: launch receipt {key}={receipt.get(key)!r}; expected {value!r}")
print(f"DynPerm launch authorization PASS: {path}")
PY
    if [ "$TOTAL_TRAINING_STEPS" = "60" ]; then
        : "${DYNPERM_PILOT_ADMISSION_RECEIPT:?P60 DynPerm endpoint requires DYNPERM_PILOT_ADMISSION_RECEIPT}"
        python3 - "$DYNPERM_PILOT_ADMISSION_RECEIPT" "$DYNPERM_RHO" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
rho = float(sys.argv[2])
if not path.is_file():
    raise SystemExit(f"ERROR: pilot admission receipt missing: {path}")
receipt = json.loads(path.read_text())
if receipt.get("status") != "pass":
    raise SystemExit(f"ERROR: pilot admission did not pass: {path}")
if receipt.get("experiment_id") != "math_qwen3_1p7b_wdl_dynperm":
    raise SystemExit(f"ERROR: pilot admission experiment_id mismatch: {path}")
if float(receipt.get("rho", "nan")) != rho:
    raise SystemExit(f"ERROR: pilot admission rho mismatch: {path}")
horizon = int(receipt.get("max_training_steps", 0))
if horizon not in (20, 30):
    raise SystemExit(f"ERROR: pilot admission must be P20/P30, got P{horizon}: {path}")
if receipt.get("material_curve_validity") is not True:
    raise SystemExit(f"ERROR: pilot admission lacks material_curve_validity=true: {path}")
print(f"DynPerm pilot admission PASS: {path}")
PY
    fi
fi

exec bash "${SCRIPT_DIR}/run_math_qwen3_1p7b_wdl_causal_p60_common.sh" "$@" \
    data.seed=20260719 \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILES" \
    data.shuffle=False \
    data.max_prompt_length=500 \
    data.max_response_length=4096 \
    data.train_batch_size=64 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.shuffle=false \
    actor_rollout_ref.actor.track_joint_submodel_losses=true \
    actor_rollout_ref.actor.policy_loss.loss_mode=wdl_sft \
    actor_rollout_ref.actor.policy_loss.wdl_sft_beta=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.submodel_kl.enabled=false \
    actor_rollout_ref.actor.submodel_kl.model1.enabled=false \
    actor_rollout_ref.actor.submodel_kl.model2.enabled=false \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.joint_training_rollout_source=model2 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.response_length=4096 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.2 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.n=3 \
    algorithm.use_kl_in_reward=False \
    trainer.validation_macro_average_sources="$MATH7_MACRO_SOURCES" \
    trainer.validation_macro_average_name=math7_macro \
    trainer.validation_macro_average_metric=acc/mean@3 \
    trainer.test_freq=5 \
    trainer.save_freq=5 \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.joint_validation_views="[model1,model2]" \
    custom_reward_function.path="${SCRIPT_DIR}/../../joint_training/custom_reward_function_latex_verify.py" \
    custom_reward_function.name=compute_score_latex_verify \
    actor_rollout_ref.actor.weak_logit_permutation.enabled=true \
    actor_rollout_ref.actor.weak_logit_permutation.rho="$DYNPERM_RHO" \
    actor_rollout_ref.actor.weak_logit_permutation.seed=42 \
    actor_rollout_ref.actor.weak_logit_permutation.row_chunk_size=16 \
    actor_rollout_ref.actor.weak_logit_permutation.audit_invariants=true \
    actor_rollout_ref.actor.weak_logit_permutation.audit_frequency=1 \
    actor_rollout_ref.actor.weak_logit_permutation.audit_rows=4 \
    actor_rollout_ref.actor.weak_logit_permutation.entropy_atol=2.0e-6 \
    actor_rollout_ref.actor.weak_logit_permutation.multiset_atol=0.0 \
    actor_rollout_ref.actor.weak_logit_permutation.log_telemetry=true
