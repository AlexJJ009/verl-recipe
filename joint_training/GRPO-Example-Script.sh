#!/usr/bin/env bash
# GRPO Stage-3 Training with SFT Model (v2 - with System Prompt)
# Model: /data-1/huggingface_cache/hub/stage3_m3 (Qwen3-4B-Base SFT)
# Algorithm: GRPO with vanilla loss, token-mean aggregation
# Test sets: AIME-2024, AIME-2025, MATH-500
#
# Changes from v1:
# - Uses datasets with system prompt (_with_system_prompt.parquet)
# - Uses new reward function (custom_reward_function_latex_verify.py)

#SBATCH --job-name=grpo-stage1_m1-sft
#SBATCH --partition=main
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks-per-node=1      # One task per node
#SBATCH --cpus-per-task=128      # cpu-cores per task
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=500:00:00
#SBATCH --output=/rl/logs/Qwen3-4B/grpo-stage3/vllm_%x_%j.out
#SBATCH --error=/rl/logs/Qwen3-4B/grpo-stage3/vllm_%x_%j.err


set -xeuo pipefail

# activate the venv
echo "Activating verl environment..."
eval "$(conda shell.bash hook)"
conda deactivate
conda activate verl07

# 设置模型缓存路径
export HF_HOME=/data-1/huggingface_cache
export PYTHONUNBUFFERED=1

# 设置 Hugging Face 访问令牌
# 检查 .env 文件是否存在
if [ -f ".env" ]; then
    # 使用 source 命令 (或 . ) 加载 .env 文件
    # 这将执行 .env 里的 'export HUGGING_FACE_HUB_TOKEN=...' 命令
    # 使得 HUGGING_FACE_HUB_TOKEN 成为一个环境变量
    echo "正在从 .env 加载环境变量..."
    source .env
else
    echo "警告：.env 文件未找到。"
fi

# (可选) 检查 Token 是否真的加载成功
if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "错误：HUGGING_FACE_HUB_TOKEN 环境变量未设置！" >&2
    echo "请确保 .env 文件存在且包含 'export HUGGING_FACE_HUB_TOKEN=...'" >&2
    exit 1
fi


# 新增：为 Ray 指定一个大的临时目录
export RAY_TMPDIR=/data-1/ray_tmp

# --- 为 W&B 设置项目和实验名称 ---
RUN_PREFIX="Qwen3-4B-SFT-GRPO_stage1_m1"
export WANDB_PROJECT="LLMBOOST_v2"
export WANDB_RUN_NAME="${RUN_PREFIX}_$(date +%s)"

# 告诉 Python 和 W&B 使用您系统上的代理
export HTTP_PROXY="http://127.0.0.1:7891"
export HTTPS_PROXY="http://127.0.0.1:7891"

# 增加 W&B 的超时和重试设置，以防网络不稳定
export WANDB_HTTP_TIMEOUT=60
export WANDB_API_TIMEOUT=60
export WANDB_MAX_RETRIES=10

# # 设置离线模式,训练完手动上传
export WANDB_MODE=offline


export LD_LIBRARY_PATH=/data-1/miniconda/envs/verl07/lib:${LD_LIBRARY_PATH:-}

# can make training faster, depends on your infrastructure
export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_IB_HCA=mlx5
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1

# NCCL 超时设置 (保留以防止验证阶段超时)
export NCCL_TIMEOUT=3600  # 1 小时超时

# Set how many GPUs we actually have on this node.
export GPUS_PER_NODE=8

NNODES=${SLURM_JOB_NUM_NODES:-1} # 如果不在SLURM环境中，默认为1
export NNODES

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_LOGGING_LEVEL=WARNING
export HYDRA_FULL_ERROR=1

echo "Using $NNODES nodes for training..."

# ------------------------------------- Setup xp params ---------------------------------------

# GRPO 算法配置
adv_estimator=grpo
loss_mode=vanilla
loss_agg_mode="token-mean"

# 使用同事提供的 SFT 模型
MODEL_PATH=/data-1/huggingface_cache/hub/stage1_m1

offload=false # it's a small model, offloading will just slow-down training
rollout_engine=vllm
rollout_mode=sync # can be async to speedup large scale xps
gpu_memory_utilization=0.75
gen_tp=2
reward_manager=dapo
shuffle_dataset=true
first_time_dataset_prep=false # prepare dataset

test_freq=5
save_freq=20
total_epochs=5
total_training_steps=500
val_before_train=True

use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001
# GRPO需要使用KL Loss来稳定训练


clip_ratio_low=0.2
clip_ratio_high=0.28
train_batch_size=128
ppo_mini_batch_size=32
ppo_micro_batch_size_per_gpu=8 # setup depending on your GPU memory
n_resp_per_prompt=8

# 为4B模型适当调整batch size
train_batch_size=128
ppo_mini_batch_size=32
ppo_micro_batch_size_per_gpu=8 # setup depending on your GPU memory
n_resp_per_prompt=16

max_prompt_length=$((1024))       # 2048: 覆盖99%的DAPO prompts
max_response_length=$((1024 * 6))
# dapo reward manager params
enable_overlong_buffer=true # true
overlong_buffer_len=$((1024 * 1))     #
overlong_penalty_factor=0.5           # 降低惩罚系数

# Paths and namings
BASE_CKPT_DIR=/data-2/checkpoints/experimental/TSPO_Refined_Experiments/${loss_mode}
mkdir -p "$BASE_CKPT_DIR"

# 检查是否存在之前的 checkpoint，如果存在则使用相同的实验名称恢复训练
LATEST_CKPT_DIR=$(find "$BASE_CKPT_DIR" -maxdepth 1 -type d -name "${RUN_PREFIX}_*" 2>/dev/null | sort | tail -1)

if [ -n "$LATEST_CKPT_DIR" ] && [ -d "$LATEST_CKPT_DIR" ]; then
    # 从目录名提取实验名称
    EXPERIMENT_NAME=$(basename "$LATEST_CKPT_DIR")
    echo "Found existing checkpoint for this specific variant: $LATEST_CKPT_DIR"
    echo "Resuming training with experiment name: $EXPERIMENT_NAME"
    # 复用原来的 WANDB_RUN_NAME
    export WANDB_RUN_NAME="$EXPERIMENT_NAME"
    CKPTS_DIR="$LATEST_CKPT_DIR"
else
    echo "No matching checkpoint found for $RUN_PREFIX. Starting new training..."
    # 保持原有的新实验名称
    CKPTS_DIR="$BASE_CKPT_DIR/${WANDB_RUN_NAME}"
    # 创建新的 checkpoint 目录
    mkdir -p "$CKPTS_DIR"
fi

echo "Experiment Name: $WANDB_RUN_NAME"
echo "Checkpoint directory: $CKPTS_DIR"

# Sampling params at rollouts
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7
val_n_samples=8

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=true
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
offload=true

entropy_checkpointing=true # This enables entropy recomputation specifically for the entropy calculation, lowering memory usage during training.


# ------------------------------------- train/val data preparation ---------------------------------------
# 数据集路径 - 使用带 system prompt 的新版本
TRAIN_FILE=/data-1/dataset/DAPO-Math-17k-Processed/train_fixed_with_system_prompt.parquet

# 多个测试集 - 使用带 system prompt 的新版本
TEST_FILE_AIME24=/data-1/dataset/AIME-2024/aime-2024_with_system_prompt.parquet
TEST_FILE_AIME25=/data-1/dataset/AIME-2025/aime-2025_with_system_prompt.parquet
TEST_FILE_MATH500=/data-1/dataset/MATH-500/math500-test_with_system_prompt.parquet

# 设置训练和测试文件
train_files="['$TRAIN_FILE']"
# 多个测试集会自动合并，指标会按 data_source 字段分组输出
test_files="['$TEST_FILE_AIME24','$TEST_FILE_AIME25','$TEST_FILE_MATH500']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    \
    \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_checkpointing=${entropy_checkpointing} \
    \
    \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=${rollout_engine} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768  \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n_samples} \
    \
    \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    \
    \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.shuffle=$shuffle_dataset \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.filter_overlong_prompts=true \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    \
    \
    reward_model.reward_manager=${reward_manager} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=false \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    custom_reward_function.path='/data-1/verl07/verl/recipe/tspo/Qwen2.5-3B/custom_reward_function_latex_verify.py' \
    custom_reward_function.name='compute_score_latex_verify' \
    \
    \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${WANDB_RUN_NAME}" \
    trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=2 \
    \
    \
    $@ > grpo_Qwen3-4B-SFT-Stage1_m1.log 2>&1
