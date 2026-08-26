#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -ne 0 ]; then
    echo "ERROR: formal lambda matrix accepts no positional/Hydra overrides" >&2
    exit 64
fi
: "${LAMBDA_ARM:?set LAMBDA_ARM to standard-c, fixed-m1, or d0}"
: "${FUSION_LAMBDA:?set FUSION_LAMBDA to the arm-authorized value}"
: "${TRAINING_LR:?set TRAINING_LR to the run-authorized value}"

case "$FUSION_LAMBDA" in
    0.5) lambda_tag=lambda05 ;;
    0.7) lambda_tag=lambda07 ;;
    0.8) lambda_tag=lambda08 ;;
    0.9) lambda_tag=lambda09 ;;
    *)
        echo "ERROR: formal lambda follow-up permits only fusion lambda 0.5, 0.7, 0.8, or 0.9" >&2
        exit 64
        ;;
esac

case "$TRAINING_LR" in
    1e-6) lr_tag=lr1e6 ;;
    5e-7) lr_tag=lr5e7 ;;
    *)
        echo "ERROR: formal lambda follow-up permits only TRAINING_LR=1e-6 or 5e-7" >&2
        exit 64
        ;;
esac

case "$LAMBDA_ARM:$FUSION_LAMBDA:$TRAINING_LR" in
    fixed-m1:0.7:1e-6|fixed-m1:0.9:1e-6)
        arm_tag=FIXED-M1
        artifact_arm=fixed-m1
        expected_model1_gradient=zero
        export FUSION_MODE=mixture
        export FREEZE_MODEL1=true
        export WDL_ARM_ID=fixed-m1-stage1
        ;;
    d0:0.7:1e-6|d0:0.9:1e-6)
        arm_tag=D0
        artifact_arm=d0
        expected_model1_gradient=zero
        export FUSION_MODE=strong_scaled
        export FREEZE_MODEL1=false
        export WDL_ARM_ID=d0
        ;;
    standard-c:0.5:5e-7|standard-c:0.8:5e-7)
        arm_tag=C
        artifact_arm=standard-c
        expected_model1_gradient=nonzero
        export FUSION_MODE=mixture
        export FREEZE_MODEL1=false
        export WDL_ARM_ID=standard-c
        ;;
    *)
        echo "ERROR: unauthorized lambda follow-up triple LAMBDA_ARM=$LAMBDA_ARM FUSION_LAMBDA=$FUSION_LAMBDA TRAINING_LR=$TRAINING_LR" >&2
        exit 64
        ;;
esac

export FUSION_LAMBDA
export LAMBDA_EXPECTED_MODEL1_GRADIENT="$expected_model1_gradient"
export RUN_PREFIX="MATH-WDL-${lambda_tag^^}-ARM-${arm_tag}-${lr_tag^^}-P60-QWEN3-1P7B"
export DYNPERM_ENABLED=false
export DYNPERM_RHO=0

# Everything below is pinned to the admitted Math causal-P60 contract. The
# scientific treatments are exactly (arm, FUSION_LAMBDA, TRAINING_LR).
export BASE_MODEL_PATH=/data-2/model_weights/math_task/qwen3_1p7b_cold_start_cotmask_v3/candidates/step_20
export EXPECTED_MODEL1_PATH="$BASE_MODEL_PATH"
export MODEL2_PATH=/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/restored_from_causal_p60_joint_20260812/final_model
export STAGE1_MODEL2_PROVENANCE_FILE="$MODEL2_PATH/model_input_provenance.json"
export STAGE1_RUN_PREFIX=MATH-B0_STAGE1-QWEN3-1P7B-COTMASK-V3
export EXPECTED_STAGE1_RUN_PREFIX="$STAGE1_RUN_PREFIX"
export STAGE1_STEP=40
export STAGE2_HANDOFF_STEP=40
export MODEL_PATH="/data-1/.cache/huggingface/math-wdl-${lambda_tag}-${artifact_arm}-p60"
export TRAIN_FILE=/data-1/dataset/math/qwen3_1p7b_stage123_seed20260719/stage1_control_stage2_then_stage3.parquet

export TOTAL_TRAINING_STEPS=60
export WDL_SFT_BETA=0.0
export LOSS_MODE=wdl_sft
export LR="$TRAINING_LR"
export LR_WARMUP_STEPS=0
export DATA_SEED=20260719
export DATA_SHUFFLE=False
export JOINT_TRAINING_ROLLOUT_SOURCE=model2
export TRACK_JOINT_SUBMODEL_LOSSES=true
export JOINT_VALIDATION_VIEWS="[model1,model2]"
export TEST_FREQ=5
export SAVE_FREQ=5
export PROTECTED_CKPT_STEPS="[20,40,45,50,55,60]"
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
export ROLLOUT_FREE_CACHE_ENGINE=False
export ROLLOUT_ENABLE_SLEEP_MODE=False
export TRAIN_PROMPT_BSZ=64
export ROLLOUT_N=8
export TRAIN_PROMPT_MINI_BSZ=512
export ACTOR_PPO_EPOCHS=1
export ACTOR_SHUFFLE=false
export TEMPERATURE=1.0
export TOP_P=1.0
export TOP_K=-1
export ROLLOUT_DO_SAMPLE=True
export VAL_N=3
export VAL_TEMPERATURE=0.2
export VAL_TOP_P=0.95
export VAL_DO_SAMPLE=True
export VAL_BEFORE_TRAIN=True

export WANDB_PROJECT=OnPolicyWDLSFT-Math-1P7B-Lambda-Followup-P60
export CAUSAL_ARTIFACT_ROOT="/data-2/model_weights/math_task/qwen3_1p7b_wdl_lambda_followup/${lambda_tag}/${artifact_arm}-${lr_tag}-p60"
export LOG_DIR="${CAUSAL_ARTIFACT_ROOT}/logs"
export WDL_MANIPULATION_RECEIPT=/data-2/model_weights/math_task/qwen3_1p7b_wdl_causal_p60/admission/manipulation_receipt.json

exec bash "${SCRIPT_DIR}/run_math_qwen3_1p7b_wdl_causal_p60_common.sh" \
    hydra.run.dir="${LOG_DIR}/hydra/${SLURM_JOB_ID:-manual}" \
    data.seed=20260719 \
    data.shuffle=False \
    actor_rollout_ref.actor.fsdp_config.seed=42 \
    actor_rollout_ref.actor.data_loader_seed=42 \
    +actor_rollout_ref.rollout.seed=0
