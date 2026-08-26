#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -ne 0 ]; then
    echo "ERROR: formal strict-scorer A accepts no positional/Hydra overrides" >&2
    exit 64
fi

# Experiment A is the single-model On-Policy SFT anchor.  It has no fusion
# lambda.  The initial Model2 tensors and ordered data stream are the same as
# the causal-P60 C/D0 matrix; the treatment is the single-model objective.
export RUN_PREFIX=${RUN_PREFIX:-MATH-A-STRICT-SCORER-P60-QWEN3-1P7B}
export INIT_MODEL_PATH=/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/restored_from_causal_p60_joint_20260812/final_model
export TRAIN_FILE=/data-1/dataset/math/qwen3_1p7b_stage123_seed20260719/stage1_control_stage2_then_stage3.parquet
export TOTAL_TRAINING_STEPS=60
export WDL_SFT_BETA=0.0
export LOSS_MODE=wdl_sft
export LR=1e-6
export LR_WARMUP_STEPS=0
export DATA_SEED=${DATA_SEED:-20260719}
export DATA_SHUFFLE=False
export TRAINING_SEED=${TRAINING_SEED:-42}
export ROLLOUT_SEED=${ROLLOUT_SEED:-0}
export JOINT_TRAINING=False
export ROLLOUT_CALCULATE_LOG_PROBS=False
export ROLLOUT_IS=null
export ROLLOUT_RS=null
export NORM_ADV_BY_STD_IN_GRPO=False
export USE_KL_IN_REWARD=False
export KL_COEF=0.0
export USE_KL_LOSS=False
export KL_LOSS_COEF=0.0

export MAX_PROMPT_LENGTH=500
export MAX_RESPONSE_LENGTH=4096
export ROLLOUT_MAX_MODEL_LEN=4596
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=32768
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4596
export ACTOR_PPO_MAX_TOKEN_LEN=4596
export GENERATION_MICRO_BATCH_SIZE=32
export LOG_PROB_MICRO_BATCH_SIZE=8
export REF_LOG_PROB_MICRO_BATCH_SIZE=8
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.55
export ROLLOUT_MAX_NUM_SEQS=256
export ROLLOUT_FREE_CACHE_ENGINE=False
export ROLLOUT_ENABLE_SLEEP_MODE=False
export TRAIN_PROMPT_BSZ=64
export ROLLOUT_N=8
export TRAIN_PROMPT_MINI_BSZ=512
export PPO_EPOCHS=1
export ACTOR_SHUFFLE=False
export TEMPERATURE=1.0
export TOP_P=1.0
export TOP_K=-1
export ROLLOUT_DO_SAMPLE=True

export TEST_FREQ=5
export SAVE_FREQ=5
export VAL_N=3
export VAL_TEMPERATURE=0.2
export VAL_TOP_P=0.95
export VAL_DO_SAMPLE=True
export VAL_BEFORE_TRAIN=True
export PROTECTED_CKPT_STEPS="[20,40,45,50,55,60]"
export MAX_ACTOR_CKPTS_TO_KEEP=1
export KEEP_BEST_CKPT=True
export BEST_CKPT_METRIC_KEY=val-core/math7_macro/acc/mean@3
export BEST_CKPT_STRIP_OPTIMIZER=False
export CHECKPOINT_SAVE_CONTENTS="[model,optimizer,extra]"

export CUSTOM_REWARD_FN_PATH="${SCRIPT_DIR}/../../joint_training/custom_reward_function_latex_verify.py"
export CUSTOM_REWARD_FN_NAME=compute_score_latex_verify
export WANDB_PROJECT=${WANDB_PROJECT:-OnPolicyWDLSFT-Math-1P7B-Strict-A-P60}
export STRICT_A_ARTIFACT_ROOT=${STRICT_A_ARTIFACT_ROOT:-/data-2/model_weights/math_task/qwen3_1p7b_opsft_a_strict_p60}
export LOG_DIR="${STRICT_A_ARTIFACT_ROOT}/logs"
export RESUME_MODE=disable

python3 "${SCRIPT_DIR}/../../../scripts/check_math_reward_contract.py" \
    --reward-path "$CUSTOM_REWARD_FN_PATH" \
    --function "$CUSTOM_REWARD_FN_NAME" >/dev/null

exec bash "${SCRIPT_DIR}/run_s1_math_qwen3_1p7b_stage123_common.sh" \
    hydra.run.dir="${LOG_DIR}/hydra/${SLURM_JOB_ID:-manual}" \
    data.seed="${DATA_SEED}" \
    data.shuffle=False \
    actor_rollout_ref.actor.fsdp_config.seed="${TRAINING_SEED}" \
    actor_rollout_ref.actor.data_loader_seed="${TRAINING_SEED}"
