#!/usr/bin/env bash
# ==============================================================================
# Joint Training with MiniRL on MATH (vLLM rollout by default)
# Model: Qwen3-4B Joint Model (dual sub-models, ~8B total parameters)
# Algorithm: MiniRL loss + Dr.GRPO advantage + token-level IS correction
#
# Based on: run_joint_minirl_qwen3_1.7b_math.sh
# Key changes:
#   - Scaled from 1.7B to 4B (2× sub-models → ~8B total)
#   - Larger batch sizes (B=64, G=8) tuned for 4B on 8× H800
#   - Conservative vLLM memory (0.55) for dual-model KV cache
#   - Same MiniRL loss, Dr.GRPO advantage, token-level IS correction
#   - Same MATH dataset with MATH-500 + AIME-2025 validation
#
# Hardware target: 8× H800 (80 GB each, 640 GB total)
# Batch-size rationale:
#   - train_prompt_bsz 32→64: 2× per-prompt compute, offset by 4B efficiency
#   - actor_ppo_max_token_len 9192→18384: 2× GPU token budget
#   - ROLLOUT_GPU_MEMORY_UTILIZATION 0.60→0.55: dual-model KV cache needs headroom
#   - ROLLOUT_MAX_NUM_SEQS 256: kept conservative (dual KV ≈ 180 KB/tok)
#
# Usage:
#   # Both sub-models from same base (default):
#   bash recipe/joint_training/run_joint_minirl_qwen3_4b_math.sh
#
#   # Specify different model for sub_models.1:
#   MODEL2_PATH=/data-1/checkpoints/some_4b_ckpt/step_N \
#     bash recipe/joint_training/run_joint_minirl_qwen3_4b_math.sh
# ==============================================================================

set -xeuo pipefail

# ===================== Section 1: Environment Activation ======================
# Docker/uv: environment is already active in the container; skip conda.
echo "Environment ready (Docker/uv mode)."

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ===================== Section 2: Cache & Temp Directories ====================
export HF_HOME=/data-1/.cache/huggingface
export RAY_TMPDIR=/data-1/ray_tmp
export TMPDIR=${TMPDIR:-/data-1/tmp}
export VLLM_CONFIG_ROOT=${VLLM_CONFIG_ROOT:-/data-1/.config/vllm}
export VERL_ZMQ_IPC_DIR=${VERL_ZMQ_IPC_DIR:-$TMPDIR}
mkdir -p "$RAY_TMPDIR" "$TMPDIR" "$VLLM_CONFIG_ROOT" "$VERL_ZMQ_IPC_DIR"

# LD_LIBRARY_PATH: auto-detect from the active Python environment
export LD_LIBRARY_PATH="$(python3 -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"

# NCCL / networking settings
export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_TIMEOUT=3600
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASHINFER}
export VLLM_NO_USAGE_STATS=${VLLM_NO_USAGE_STATS:-1}
export VLLM_DO_NOT_TRACK=${VLLM_DO_NOT_TRACK:-1}
export RAY_LOGGING_LEVEL=WARNING
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ===================== Section 3: W&B Configuration ===========================
RUN_PREFIX=${RUN_PREFIX:-"Joint-MiniRL-Qwen3-4B-MATH-GC500"}
export WANDB_PROJECT=${WANDB_PROJECT:-"JointTraining"}
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_PREFIX}_$(date +%s)}"

# Increase W&B timeout/retries for unreliable networks
export WANDB_HTTP_TIMEOUT=60
export WANDB_API_TIMEOUT=60
export WANDB_MAX_RETRIES=10

# Set offline mode — run `wandb sync <run_dir>` after training to upload
export WANDB_MODE=${WANDB_MODE:-offline}

# ===================== Section 4: Hardware ====================================
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# ===================== Section 5: Model & Data Paths ==========================
BASE_MODEL_PATH=${BASE_MODEL_PATH:-"/data-1/.cache/huggingface/Qwen3-4B-Base"}
# MODEL2_PATH: optional second model for sub_models.1 (e.g. a merged checkpoint).
# If unset or empty, base model is cloned to both sub-models.
# e.g. MODEL2_PATH=/data-1/checkpoints/EXP-XX_.../step_N
MODEL2_PATH=${MODEL2_PATH:-""}
if [ -n "$MODEL2_PATH" ]; then
    MODEL2_CACHE_TAG=$(basename "$MODEL2_PATH")
    MODEL2_CACHE_TAG=${MODEL2_CACHE_TAG//[^[:alnum:]._-]/-}
    DEFAULT_MODEL_PATH="/data-1/.cache/huggingface/QwenJoint-4B-dual-${MODEL2_CACHE_TAG}"
else
    DEFAULT_MODEL_PATH="/data-1/.cache/huggingface/QwenJoint-4B"
fi
MODEL_PATH=${MODEL_PATH:-"$DEFAULT_MODEL_PATH"}
TRAIN_FILE=${TRAIN_FILE:-"/data-1/dataset/Maxwell-Jia-MATH-Processed-no-system-prompt/train_with_system_prompt.parquet"}
TEST_FILES=${TEST_FILES:-"['/data-1/dataset/MATH-500/math500-test.parquet','/data-1/dataset/AIME-2025/aime-2025.parquet']"}

# Auto-prepare joint model if it doesn't exist yet
if [ ! -d "$MODEL_PATH" ]; then
    echo "Joint model not found at $MODEL_PATH. Preparing from base model..."
    if [ ! -d "$BASE_MODEL_PATH" ]; then
        echo "ERROR: Base model not found at $BASE_MODEL_PATH"
        echo "Please download Qwen3-4B-Base first, e.g.:"
        echo "  python -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-4B-Base', local_dir='$BASE_MODEL_PATH', local_dir_use_symlinks=False)\""
        exit 1
    fi
    PREPARE_ARGS=(
        --base_model_path "$BASE_MODEL_PATH"
        --output_path "$MODEL_PATH"
        --fusion_lambda 0.50
    )
    if [ -n "$MODEL2_PATH" ]; then
        if [ ! -d "$MODEL2_PATH" ]; then
            echo "ERROR: Model2 not found at $MODEL2_PATH"
            exit 1
        fi
        PREPARE_ARGS+=(--model2_path "$MODEL2_PATH")
        echo "Using dual-model mode: model1=$BASE_MODEL_PATH, model2=$MODEL2_PATH"
    fi
    python3 -m verl.models.joint_model.prepare_joint_weights "${PREPARE_ARGS[@]}"
    echo "Joint model prepared at $MODEL_PATH"
fi

# ===================== Section 6: Checkpoint Directory =========================
MIN_FREE_GB_FOR_CKPT=${MIN_FREE_GB_FOR_CKPT:-30}
MIN_FREE_KB_FOR_CKPT=$((MIN_FREE_GB_FOR_CKPT * 1024 * 1024))
MAX_ACTOR_CKPTS_TO_KEEP=${MAX_ACTOR_CKPTS_TO_KEEP:-5}
MAX_CRITIC_CKPTS_TO_KEEP=${MAX_CRITIC_CKPTS_TO_KEEP:-2}

DEFAULT_CKPT_BASE_DIR="/data-1/checkpoints"

get_df_target() {
    local path="$1"
    while [ ! -e "$path" ] && [ "$path" != "/" ]; do
        path=$(dirname "$path")
    done
    printf '%s\n' "$path"
}

get_free_kb() {
    local target
    target=$(get_df_target "$1")
    df -Pk "$target" | awk 'NR==2 {print $4}'
}

get_mount_source() {
    local target
    target=$(get_df_target "$1")
    df -Pk "$target" | awk 'NR==2 {print $1}'
}

if [ -z "${BASE_CKPT_DIR+x}" ]; then
    BASE_CKPT_DIR="$DEFAULT_CKPT_BASE_DIR"
else
    BASE_CKPT_DIR="${BASE_CKPT_DIR}"
fi

ROOT_MOUNT_SOURCE=$(get_mount_source "/")
BASE_CKPT_MOUNT_SOURCE=$(get_mount_source "$BASE_CKPT_DIR")
if [ "$BASE_CKPT_MOUNT_SOURCE" = "$ROOT_MOUNT_SOURCE" ] && [[ "$BASE_CKPT_DIR" != /data-1/* ]]; then
    echo "[joint-minirl-4b] WARNING: ${BASE_CKPT_DIR} resolves to the root filesystem (${BASE_CKPT_MOUNT_SOURCE}), not the large /data-1 volume." >&2
    echo "[joint-minirl-4b] WARNING: On this host, prefer BASE_CKPT_DIR=/data-1/checkpoints." >&2
fi

BASE_CKPT_FREE_KB=$(get_free_kb "$BASE_CKPT_DIR")
if [ "$BASE_CKPT_FREE_KB" -lt "$MIN_FREE_KB_FOR_CKPT" ]; then
    echo "ERROR: ${BASE_CKPT_DIR} has only $((BASE_CKPT_FREE_KB / 1024 / 1024)) GiB free, below MIN_FREE_GB_FOR_CKPT=${MIN_FREE_GB_FOR_CKPT}." >&2
    echo "Set BASE_CKPT_DIR to a larger filesystem or lower MIN_FREE_GB_FOR_CKPT if you know the checkpoint budget is safe." >&2
    exit 1
fi

mkdir -p "$BASE_CKPT_DIR"

# Auto-resume: reuse existing checkpoint directory if one matches this prefix
LATEST_CKPT_DIR=$(find "$BASE_CKPT_DIR" -maxdepth 1 -type d -name "${RUN_PREFIX}_*" 2>/dev/null | sort | tail -1)

if [ -n "$LATEST_CKPT_DIR" ] && [ -d "$LATEST_CKPT_DIR" ]; then
    EXPERIMENT_NAME=$(basename "$LATEST_CKPT_DIR")
    echo "Resuming from existing checkpoint: $LATEST_CKPT_DIR"
    export WANDB_RUN_NAME="$EXPERIMENT_NAME"
    CKPTS_DIR="$LATEST_CKPT_DIR"
    IS_RESUME=true
else
    echo "No matching checkpoint found. Starting new training..."
    CKPTS_DIR="$BASE_CKPT_DIR/${WANDB_RUN_NAME}"
    mkdir -p "$CKPTS_DIR"
    IS_RESUME=false
fi

echo "Experiment Name : $WANDB_RUN_NAME"
echo "Checkpoint Dir  : $CKPTS_DIR"

# ===================== Section 7: Log File ====================================
LOG_DIR=${LOG_DIR:-/data-1/verl07/verl/recipe/joint_training}
mkdir -p "$LOG_DIR"
if [ "$IS_RESUME" = true ]; then
    LOG_FILE="${LOG_DIR}/${WANDB_RUN_NAME}_resumed_$(date +%s).log"
else
    LOG_FILE="${LOG_DIR}/${WANDB_RUN_NAME}.log"
fi
export VERL_FILE_LOGGER_ROOT=${VERL_FILE_LOGGER_ROOT:-"${LOG_DIR}/metrics"}
mkdir -p "$VERL_FILE_LOGGER_ROOT"
VAL_GENERATIONS_TO_LOG=${VAL_GENERATIONS_TO_LOG:-3}
VAL_GENERATIONS_TO_TRACKING=${VAL_GENERATIONS_TO_TRACKING:--1}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-"${LOG_DIR}/validation/${WANDB_RUN_NAME}"}
mkdir -p "$VALIDATION_DATA_DIR"
echo "Log file: $LOG_FILE"

# ===================== Section 8: MiniRL Algorithm Config =====================
adv_estimator=grpo
loss_agg_mode="seq-mean-token-sum"    # MiniRL: no per-token length normalization

# KL settings: no KL for MiniRL
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

# MiniRL clipping parameters (paper-recommended values)
clip_ratio_low=0.2
clip_ratio_high=0.27

# MiniRL loss mode
loss_mode="minirl"

# ===================== Section 8a: Rollout Correction (MiniRL) ================
# Token-level IS (MiniRL core: corrects vLLM/FSDP numerical differences)
rollout_is="token"
rollout_is_threshold=5.0              # MiniRL paper recommended value
rollout_is_batch_normalize="false"
rollout_rs="null"
rollout_rs_threshold="null"

# ===================== Section 8b: Reward Manager Config =====================
# Use DAPO reward manager with overlong buffer penalty
reward_manager=dapo
enable_overlong_buffer=false
overlong_buffer_len=$((1024 * 1))     # 1024 tokens buffer zone
overlong_penalty_factor=0.5

# Custom reward function (LaTeX semantic verification with 3-tier fallback)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUSTOM_REWARD_FN_PATH="${SCRIPT_DIR}/custom_reward_function_latex_verify.py"
CUSTOM_REWARD_FN_NAME="compute_score_latex_verify"

# ===================== Section 9: Sequence Lengths ============================
max_prompt_length=500                 # MATH problems are longer
max_response_length=4096              # MATH needs longer reasoning chains

# ===================== Section 10: Batch Sizes (tuned for 4B joint + 8× H800) =
# Joint 4B = ~8B total params. 8 GPUs, same per-GPU prompt load as 1.7B joint.
# 512 sequences/step (64 prompts × G=8), each up to 4596 tokens.
train_prompt_bsz=64                   # B=64 (2× from 1.7B joint)
n_resp_per_prompt=8                   # G=8
train_prompt_mini_bsz=16              # 2× larger with remove_padding (no padding overhead)

# ===================== Section 11: Sampling Parameters ========================
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.95

# ===================== Section 12: Performance & Memory =======================
# H800 80 GB per GPU; joint 4B = ~16 GB bf16 total weights.
# FSDP shards across 8 GPUs: ~2 GB weights + ~8 GB optimizer + ~2 GB grad = ~12 GB/GPU.
# Leaves ~68 GB per GPU for activations + rollout KV cache.
#
# vLLM KV cache per token (4B, GQA 8 heads, head_dim=80, 36 layers, bf16):
#   single sub-model: 2 × 8 × 80 × 2 bytes × 36 = 92 160 bytes ≈ 90 KB/token
#   joint (2 sub-models): ~180 KB/token
# At gpu_memory_utilization=0.55: 44 GB budget − 16 GB weights = 28 GB KV cache
#   → ~155 K tokens in cache pool; comfortable for 256 concurrent sequences.
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 8))   # 18 384 (2× from 1.7B joint)
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 6))   # 27 576
offload=False
fsdp_size=-1
USE_REMOVE_PADDING_WAS_SET=${USE_REMOVE_PADDING+x}
LOG_PROB_MICRO_BATCH_SIZE_WAS_SET=${LOG_PROB_MICRO_BATCH_SIZE+x}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}

# Rollout settings
GENERATION_MICRO_BATCH_SIZE=${GENERATION_MICRO_BATCH_SIZE:-16}    # 2× from 1.7B joint
LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-4}         # 2× from 1.7B joint
ROLLOUT_ENGINE=${ROLLOUT_ENGINE:-vllm}
ROLLOUT_MODE=${ROLLOUT_MODE:-async}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-true}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((max_prompt_length + max_response_length))}
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$((max_prompt_length + max_response_length))}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}   # 0.4: FlashInfer default (lower to avoid OOM with new attention backend)
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}                                     # 4B fits on 1 GPU
ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-4}
ROLLOUT_ENABLE_CHUNKED_PREFILL=${ROLLOUT_ENABLE_CHUNKED_PREFILL:-true}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((max_prompt_length + max_response_length))}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-256}

if [ "${ROLLOUT_ENGINE}" = "vllm" ]; then
    # vLLM CuMemAllocator.sleep() asserts against expandable_segments:True
    # (PYTORCH_CUDA_ALLOC_CONF), so free_cache_engine and sleep_mode must stay off.
    # Memory is managed by lowering gpu_memory_utilization instead.
    ROLLOUT_FREE_CACHE_ENGINE_DEFAULT=False
    ROLLOUT_ENABLE_SLEEP_MODE_DEFAULT=False
else
    ROLLOUT_FREE_CACHE_ENGINE_DEFAULT=True
    ROLLOUT_ENABLE_SLEEP_MODE_DEFAULT=True
fi
ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-${ROLLOUT_FREE_CACHE_ENGINE_DEFAULT}}
ROLLOUT_ENABLE_SLEEP_MODE=${ROLLOUT_ENABLE_SLEEP_MODE:-${ROLLOUT_ENABLE_SLEEP_MODE_DEFAULT}}

if [ "${USE_REMOVE_PADDING}" = "True" ]; then
    if ! python3 -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('flash_attn') else 1)" \
        >/dev/null 2>&1; then
        echo "[joint-minirl-4b] WARNING: flash_attn is not installed; disabling USE_REMOVE_PADDING to avoid a late actor crash." >&2
        USE_REMOVE_PADDING=False
        if [ -z "${LOG_PROB_MICRO_BATCH_SIZE_WAS_SET}" ]; then
            LOG_PROB_MICRO_BATCH_SIZE=1
            echo "[joint-minirl-4b] WARNING: LOG_PROB_MICRO_BATCH_SIZE was not set explicitly; reducing it to 1 for the dense-logits fallback path." >&2
        fi
    fi
fi

# ===================== Section 13: Training Schedule ==========================
test_freq=5
save_freq=20
total_epochs=3
total_training_steps=200
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
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize} \
    algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold} \
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
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.grad_clip=500.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    \
    `# --- Reference model ---` \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    \
    `# --- Model ---` \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING} \
    actor_rollout_ref.model.trust_remote_code=True \
    +actor_rollout_ref.model.joint_training=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
    \
    `# --- Rollout (vLLM with FlashInfer backend) ---` \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.name=${ROLLOUT_ENGINE} \
    actor_rollout_ref.rollout.mode=${ROLLOUT_MODE} \
    actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER} \
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE} \
    actor_rollout_ref.rollout.free_cache_engine=${ROLLOUT_FREE_CACHE_ENGINE} \
    +actor_rollout_ref.rollout.enable_sleep_mode=${ROLLOUT_ENABLE_SLEEP_MODE} \
    actor_rollout_ref.rollout.agent.num_workers=${ROLLOUT_AGENT_NUM_WORKERS} \
    actor_rollout_ref.rollout.enable_chunked_prefill=${ROLLOUT_ENABLE_CHUNKED_PREFILL} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE} \
    +actor_rollout_ref.rollout.micro_batch_size=${GENERATION_MICRO_BATCH_SIZE} \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    \
    `# --- Data ---` \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILES}" \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
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
    trainer.logger='["wandb","file"]' \
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
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPTS_TO_KEEP} \
    trainer.max_critic_ckpt_to_keep=${MAX_CRITIC_CKPTS_TO_KEEP} \
    trainer.log_val_generations=${VAL_GENERATIONS_TO_LOG} \
    +trainer.log_val_generations_to_tracking=${VAL_GENERATIONS_TO_TRACKING} \
    trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
    trainer.resume_mode=auto \
    \
    "$@" 2>&1 | tee "$LOG_FILE"
