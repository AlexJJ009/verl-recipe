#!/usr/bin/env bash
# Source-only admission and identity contract for Math DynPerm P60 runs.

: "${WDL_ARM_ID:?WDL_ARM_ID must identify standard-c or fixed-m1-stage1}"
if [ "${STAGE2_DRY_RUN:-0}" != "1" ]; then
    : "${DYNPERM_LAUNCH_RECEIPT:?formal DynPerm P60 requires DYNPERM_LAUNCH_RECEIPT}"
    : "${DYNPERM_IMAGE_ID:?formal DynPerm P60 requires the exact container image id}"
fi

case "$WDL_ARM_ID" in
    standard-c)
        DYNPERM_ARM_TAG=standard-c
        DYNPERM_EXPECTED_FREEZE_MODEL1=false
        ;;
    fixed-m1-stage1)
        DYNPERM_ARM_TAG=fixed-m1-stage1
        DYNPERM_EXPECTED_FREEZE_MODEL1=true
        ;;
    *)
        echo "ERROR: unsupported DynPerm P60 arm: $WDL_ARM_ID" >&2
        return 1
        ;;
esac

if [ "${FREEZE_MODEL1:-false}" != "$DYNPERM_EXPECTED_FREEZE_MODEL1" ]; then
    echo "ERROR: $WDL_ARM_ID requires FREEZE_MODEL1=$DYNPERM_EXPECTED_FREEZE_MODEL1" >&2
    return 1
fi

# Keep the non-treatment contract identical for Standard C and fixed-M1.
export TOTAL_TRAINING_STEPS=60
export PROTECTED_CKPT_STEPS="[20,40,45,50,60]"
export FUSION_LAMBDA=0.8
export FUSION_MODE=mixture
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
export CHECKPOINT_SAVE_CONTENTS="[model,optimizer,extra]"
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
# Both vLLM sleep levels fail in CUDA cuMem wake_up on the cluster's L40S
# runtime. Keep the reduced rollout allocation, but disable sleep transitions.
export ROLLOUT_FREE_CACHE_ENGINE=False
export ROLLOUT_ENABLE_SLEEP_MODE=False
export ACTOR_CALCULATE_ENTROPY=False
export CALCULATE_ENTROPY=False
export ROLLOUT_TP_SIZE=1
export TRAIN_PROMPT_BSZ=64
export ROLLOUT_N=8
export TEMPERATURE=1.0
export TOP_P=1.0
export TOP_K=-1
export ROLLOUT_DO_SAMPLE=True
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
export RUN_PREFIX="MATH-WDL-DYNPERM-${DYNPERM_DOSE_TAG^^}-${DYNPERM_ARM_TAG^^}-P60-QWEN3-1P7B"
export WANDB_PROJECT=OnPolicyWDLSFT-Math-1P7B-DynPerm-P60
export CAUSAL_ARTIFACT_ROOT="/data-2/model_weights/math_task/qwen3_1p7b_wdl_dynperm/${DYNPERM_DOSE_TAG}/${DYNPERM_ARM_TAG}-p60"
export LOG_DIR="${CAUSAL_ARTIFACT_ROOT}/logs"
export WDL_MANIPULATION_RECEIPT=/data-2/model_weights/math_task/qwen3_1p7b_wdl_causal_p60/admission/manipulation_receipt.json
export MODEL_PATH="/data-1/.cache/huggingface/math-wdl-dynperm-${DYNPERM_DOSE_TAG}-${DYNPERM_ARM_TAG}-p60"

# Appended after caller arguments by the shared causal entry. This makes the
# non-treatment contract candidate-bound rather than merely a set of defaults.
DYNPERM_FORMAL_OVERRIDES=(
    hydra.run.dir="${LOG_DIR}/hydra/${SLURM_JOB_ID:-manual}"
    data.seed=20260719
    data.train_files="$TRAIN_FILE"
    data.val_files="$TEST_FILES"
    data.shuffle=False
    data.max_prompt_length=500
    data.max_response_length=4096
    data.train_batch_size=64
    actor_rollout_ref.actor.fsdp_config.seed=42
    actor_rollout_ref.actor.data_loader_seed=42
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=0
    actor_rollout_ref.actor.ppo_mini_batch_size=512
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.shuffle=false
    actor_rollout_ref.actor.track_joint_submodel_losses=true
    actor_rollout_ref.actor.policy_loss.loss_mode=wdl_sft
    actor_rollout_ref.actor.policy_loss.wdl_sft_beta=0.0
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.calculate_entropy=False
    actor_rollout_ref.actor.checkpoint.save_contents="[model,optimizer,extra]"
    actor_rollout_ref.actor.checkpoint.load_contents="[model,optimizer,extra]"
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.submodel_kl.enabled=false
    actor_rollout_ref.actor.submodel_kl.model1.enabled=false
    actor_rollout_ref.actor.submodel_kl.model2.enabled=false
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.joint_training_rollout_source=model2
    actor_rollout_ref.rollout.n=8
    +actor_rollout_ref.rollout.seed=0
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.do_sample=True
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.response_length=4096
    actor_rollout_ref.rollout.val_kwargs.temperature=0.2
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    actor_rollout_ref.rollout.val_kwargs.n=3
    algorithm.use_kl_in_reward=False
    trainer.validation_macro_average_sources="$MATH7_MACRO_SOURCES"
    trainer.validation_macro_average_name=math7_macro
    trainer.validation_macro_average_metric=acc/mean@3
    trainer.test_freq=5
    trainer.save_freq=5
    trainer.total_training_steps=60
    trainer.joint_validation_views="[model1,model2]"
    custom_reward_function.path="${SCRIPT_DIR}/../../joint_training/custom_reward_function_latex_verify.py"
    custom_reward_function.name=compute_score_latex_verify
)

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

if [ "${STAGE2_DRY_RUN:-0}" != "1" ]; then
    PARENT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
    RECIPE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
    if [ -n "$(git -C "$PARENT_ROOT" status --porcelain)" ] \
        || [ -n "$(git -C "$RECIPE_ROOT" status --porcelain)" ]; then
        echo "ERROR: formal DynPerm launch requires clean parent and recipe worktrees" >&2
        return 1
    fi
    PARENT_SHA="$(git -C "$PARENT_ROOT" rev-parse HEAD)"
    RECIPE_SHA="$(git -C "$RECIPE_ROOT" rev-parse HEAD)"
    python3 - "$DYNPERM_LAUNCH_RECEIPT" "$DYNPERM_RHO" "$WDL_ARM_ID" "$PARENT_SHA" "$RECIPE_SHA" "$DYNPERM_IMAGE_ID" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
rho = float(sys.argv[2])
arm_id = sys.argv[3]
parent_sha = sys.argv[4]
recipe_sha = sys.argv[5]
image_id = sys.argv[6]
if not path.is_file():
    raise SystemExit(f"ERROR: launch receipt missing: {path}")
receipt = json.loads(path.read_text())
expected = {
    "status": "authorized",
    "experiment_id": "math_qwen3_1p7b_wdl_dynperm_p60",
    "rho": rho,
    "max_training_steps": 60,
    "parent_candidate_sha": parent_sha,
    "recipe_candidate_sha": recipe_sha,
    "image_id": image_id,
}
for key, value in expected.items():
    if receipt.get(key) != value:
        raise SystemExit(f"ERROR: launch receipt {key}={receipt.get(key)!r}; expected {value!r}")
if arm_id not in receipt.get("arms", []):
    raise SystemExit(f"ERROR: launch receipt does not authorize arm {arm_id!r}")
print(f"DynPerm P60 launch authorization PASS for {arm_id}: {path}")
PY
fi
