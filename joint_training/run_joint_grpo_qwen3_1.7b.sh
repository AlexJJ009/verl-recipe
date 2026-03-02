#!/usr/bin/env bash
# ==============================================================================
# Joint Training with GRPO on GSM8K (HuggingFace Rollout)
# Model: Qwen3-1.7B Joint Model
# Algorithm: GRPO with vanilla loss, token-mean aggregation
#
# Usage: bash recipe/joint_training/run_joint_grpo_qwen3_1.7b.sh
# ==============================================================================

set -xeuo pipefail

# ===================== Section 1: Environment Activation ======================
echo "Activating verl07 conda environment..."
eval "$(conda shell.bash hook)"
conda deactivate
conda activate verl07

export PYTHONUNBUFFERED=1

# ===================== Section 2: Cache & Temp Directories ====================
export HF_HOME=/data-1/.cache/huggingface
export RAY_TMPDIR=/data-1/ray_tmp
mkdir -p "$RAY_TMPDIR"

export LD_LIBRARY_PATH=/data-1/.cache/conda/envs/verl07/lib:${LD_LIBRARY_PATH:-}

# NCCL / networking settings
export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_TIMEOUT=3600
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_LOGGING_LEVEL=WARNING
export HYDRA_FULL_ERROR=1

# ===================== Section 3: W&B Configuration ===========================
RUN_PREFIX="Joint-GRPO-Qwen3-1.7B-GSM8K"
export WANDB_PROJECT="JointTraining"
export WANDB_RUN_NAME="${RUN_PREFIX}_$(date +%s)"

# Increase W&B timeout/retries for unreliable networks
export WANDB_HTTP_TIMEOUT=60
export WANDB_API_TIMEOUT=60
export WANDB_MAX_RETRIES=10

# Set offline mode — run `wandb sync <run_dir>` after training to upload
export WANDB_MODE=offline

# ===================== Section 4: Hardware ====================================
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# ===================== Section 5: Model & Data Paths ==========================
MODEL_PATH=${MODEL_PATH:-"/data-1/.cache/huggingface/QwenJoint-1.7B"}
TRAIN_FILE=${TRAIN_FILE:-"/data-1/dataset/gsm8k/train.parquet"}
TEST_FILE=${TEST_FILE:-"/data-1/dataset/gsm8k/test.parquet"}

# ===================== Section 6: Checkpoint Directory (on /data-2) ===========
BASE_CKPT_DIR=/data-2/checkpoints/JointTraining/GRPO
mkdir -p "$BASE_CKPT_DIR"

# Auto-resume: reuse existing checkpoint directory if one matches this prefix
LATEST_CKPT_DIR=$(find "$BASE_CKPT_DIR" -maxdepth 1 -type d -name "${RUN_PREFIX}_*" 2>/dev/null | sort | tail -1)

if [ -n "$LATEST_CKPT_DIR" ] && [ -d "$LATEST_CKPT_DIR" ]; then
    EXPERIMENT_NAME=$(basename "$LATEST_CKPT_DIR")
    echo "Resuming from existing checkpoint: $LATEST_CKPT_DIR"
    export WANDB_RUN_NAME="$EXPERIMENT_NAME"
    CKPTS_DIR="$LATEST_CKPT_DIR"
else
    echo "No matching checkpoint found. Starting new training..."
    CKPTS_DIR="$BASE_CKPT_DIR/${WANDB_RUN_NAME}"
    mkdir -p "$CKPTS_DIR"
fi

echo "Experiment Name : $WANDB_RUN_NAME"
echo "Checkpoint Dir  : $CKPTS_DIR"

# ===================== Section 7: Log File ====================================
LOG_DIR=/data-1/verl07/verl/recipe/joint_training
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${WANDB_RUN_NAME}.log"
echo "Log file: $LOG_FILE"

# ===================== Section 8: GRPO Algorithm Config =======================
adv_estimator=grpo
loss_agg_mode="token-mean"

# KL settings: no KL for pure GRPO
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

# Clipping
clip_ratio_low=0.2
clip_ratio_high=0.28

# ===================== Section 8b: Reward Manager Config =====================
# Use DAPO reward manager with overlong buffer penalty
reward_manager=dapo
enable_overlong_buffer=true
overlong_buffer_len=$((1024 * 1))   # 1024 tokens buffer zone
overlong_penalty_factor=0.5

# Custom reward function (LaTeX semantic verification with 3-tier fallback)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUSTOM_REWARD_FN_PATH="${SCRIPT_DIR}/custom_reward_function_latex_verify.py"
CUSTOM_REWARD_FN_NAME="compute_score_latex_verify"

# ===================== Section 9: Sequence Lengths ============================
max_prompt_length=512
max_response_length=1024

# ===================== Section 10: Batch Sizes (tuned for 1.7B model) =========
train_prompt_bsz=32
n_resp_per_prompt=4
train_prompt_mini_bsz=8

# ===================== Section 11: Sampling Parameters ========================
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.95

# ===================== Section 12: Performance & Memory =======================
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
offload=False
fsdp_size=-1

# HF rollout settings
micro_batch_size=4

# ===================== Section 13: Training Schedule ==========================
test_freq=5
save_freq=20
total_epochs=3
total_training_steps=100
val_before_train=True

# ==============================================================================
# Section 14: Launch Training
# ==============================================================================
python3 -m verl.trainer.main_ppo \
    \
    `# --- Algorithm ---` \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    \
    `# --- Actor (policy) ---` \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    \
    `# --- Reference model ---` \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    \
    `# --- Model ---` \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    +actor_rollout_ref.model.joint_training=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    `# --- Rollout (HF Transformer for joint training) ---` \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${micro_batch_size} \
    +actor_rollout_ref.rollout.micro_batch_size=${micro_batch_size} \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    \
    `# --- Data ---` \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    \
    `# --- Reward (DAPO with LaTeX semantic verification) ---` \
    reward_model.reward_manager=${reward_manager} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=false \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    custom_reward_function.path="${CUSTOM_REWARD_FN_PATH}" \
    custom_reward_function.name="${CUSTOM_REWARD_FN_NAME}" \
    \
    `# --- Trainer ---` \
    trainer.logger='["wandb"]' \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${WANDB_RUN_NAME}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    \
    "$@" 2>&1 | tee "$LOG_FILE"
