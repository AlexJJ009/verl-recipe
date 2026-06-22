#!/usr/bin/env bash
# ==============================================================================
# Shared launcher for the "single-model" ablation on On-Policy WDL-SFT.
#
# This file is sourced by the per-experiment thin wrappers (run_2a_*.sh, …).
# It owns the env setup, checkpoint/resume logic, and the full Hydra launch
# command. The per-experiment wrappers only set a handful of knobs:
#
#   Required from caller (export before sourcing):
#     RUN_PREFIX            e.g. "WDL-SFT-Qwen3-4B-MATH-2A-BASE"
#     INIT_MODEL_PATH       absolute path to the HF model dir (Qwen3 arch)
#     LOSS_MODE             "wdl_sft_is" / "wdl_group_adv_is" (ablation) or
#                           "minirl" / "vanilla" (baselines)
#     LR                    learning rate (e.g. 5e-7, 1e-6)
#
#   Optional from caller:
#     WDL_SFT_BETA          reverse-SFT weight for wdl_sft_is only (default 0.0)
#     TRAIN_PROMPT_BSZ      default 64
#     TRAIN_PROMPT_MINI_BSZ default 8  (match 1A/1B/1C for compute-matched comparison)
#     TOTAL_TRAINING_STEPS  default 300
#
# Boundary w.r.t. 1A/1B/1C (joint training):
#   - joint_training flag: FALSE (default) — no fused-logit rollout
#   - no joint model prep step; INIT_MODEL_PATH points at a standard Qwen3 dir
#   - validation uses standard single-model metrics (no joint-eval aggregation)
#   - everything else (data, batch, rollout N, temp, eval cadence, IS config) is
#     identical to the joint runs so the delta is exactly "model count + loss".
# ==============================================================================

set -xeuo pipefail

: "${RUN_PREFIX:?RUN_PREFIX must be set by the caller}"
: "${INIT_MODEL_PATH:?INIT_MODEL_PATH must be set by the caller}"
: "${LOSS_MODE:?LOSS_MODE must be set by the caller}"
: "${LR:?LR must be set by the caller}"

WDL_SFT_BETA=${WDL_SFT_BETA:-0.0}

# ===================== Section 1: Environment Activation ======================
echo "Environment ready (Docker/uv mode)."

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ===================== Section 2: Cache & Temp Directories ====================
# All paths are overridable so the same scripts work on both the local dev box
# (defaults → /data-1/...) and Meituan MLP (overridden via meituan/env.sh).
export HF_HOME=${HF_HOME:-/data-1/.cache/huggingface}
export RAY_TMPDIR=${RAY_TMPDIR:-/data-1/ray_tmp}
export TMPDIR=${TMPDIR:-/data-1/tmp}
export VLLM_CONFIG_ROOT=${VLLM_CONFIG_ROOT:-/data-1/.config/vllm}
export VERL_ZMQ_IPC_DIR=${VERL_ZMQ_IPC_DIR:-$TMPDIR}
mkdir -p "$RAY_TMPDIR" "$TMPDIR" "$VLLM_CONFIG_ROOT" "$VERL_ZMQ_IPC_DIR"

export LD_LIBRARY_PATH="$(python3 -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"

export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_TIMEOUT=3600
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASHINFER}
export VLLM_NO_USAGE_STATS=${VLLM_NO_USAGE_STATS:-1}
export VLLM_DO_NOT_TRACK=${VLLM_DO_NOT_TRACK:-1}
export RAY_LOGGING_LEVEL=WARNING
export HYDRA_FULL_ERROR=1

# ===================== Section 3: W&B Configuration ===========================
export WANDB_PROJECT=${WANDB_PROJECT:-"OnPolicyWDLSFT"}
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_PREFIX}_$(date +%s)}"

export WANDB_HTTP_TIMEOUT=60
export WANDB_API_TIMEOUT=60
export WANDB_MAX_RETRIES=10
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-"/data-1/wandb_runs/${RUN_PREFIX}"}
mkdir -p "$WANDB_DIR"

# ===================== Section 4: Hardware ====================================
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# ===================== Section 5: Model & Data Paths ==========================
# SINGLE-MODEL ABLATION: no joint model prep — INIT_MODEL_PATH points at a
# standard Qwen3 HF dir (QwenJointForCausalLM path is NOT taken).
if [ ! -d "$INIT_MODEL_PATH" ]; then
    echo "ERROR: INIT_MODEL_PATH does not exist: $INIT_MODEL_PATH" >&2
    exit 1
fi
MODEL_PATH="$INIT_MODEL_PATH"

TRAIN_FILE=${TRAIN_FILE:-"/data-1/dataset/EnsembleLLM-data-processed/train_rl_format.parquet"}
TEST_FILES=${TEST_FILES:-"['/data-1/dataset/MATH-500/math500-test_with_system_prompt.parquet','/data-1/dataset/AIME-2025/aime-2025_with_system_prompt.parquet']"}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}

# ===================== Section 6: Checkpoint Directory ========================
MIN_FREE_GB_FOR_CKPT=${MIN_FREE_GB_FOR_CKPT:-30}
MIN_FREE_KB_FOR_CKPT=$((MIN_FREE_GB_FOR_CKPT * 1024 * 1024))
MAX_ACTOR_CKPTS_TO_KEEP=${MAX_ACTOR_CKPTS_TO_KEEP:-13}
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

if [ -z "${BASE_CKPT_DIR+x}" ]; then
    BASE_CKPT_DIR="$DEFAULT_CKPT_BASE_DIR"
else
    BASE_CKPT_DIR="${BASE_CKPT_DIR}"
fi

BASE_CKPT_FREE_KB=$(get_free_kb "$BASE_CKPT_DIR")
if [ "$BASE_CKPT_FREE_KB" -lt "$MIN_FREE_KB_FOR_CKPT" ]; then
    echo "ERROR: ${BASE_CKPT_DIR} has only $((BASE_CKPT_FREE_KB / 1024 / 1024)) GiB free, below MIN_FREE_GB_FOR_CKPT=${MIN_FREE_GB_FOR_CKPT}." >&2
    exit 1
fi

mkdir -p "$BASE_CKPT_DIR"

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
echo "Init model path : $MODEL_PATH"
echo "Loss mode       : $LOSS_MODE"
echo "Beta (wdl only) : $WDL_SFT_BETA"
echo "Learning rate   : $LR"

# ===================== Section 7: Log File ====================================
# WRAPPER_SCRIPT_DIR is set by the caller (the thin wrapper) before sourcing.
LOG_DIR=${LOG_DIR:-${WRAPPER_SCRIPT_DIR:-$(pwd)}}
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

# ===================== Section 8: Algorithm Config ============================
# Matched to 1A/1B/1C so the only delta is model count + loss choice.
adv_estimator=grpo
loss_agg_mode="seq-mean-token-sum"

use_kl_in_reward=${USE_KL_IN_REWARD:-False}
kl_coef=${KL_COEF:-0.0}
use_kl_loss=${USE_KL_LOSS:-False}
kl_loss_coef=${KL_LOSS_COEF:-0.0}

# GRPO advantage normalization. Default False (matches 1A/B/C and 2A/B/C/Z).
# Canonical DeepSeek GRPO uses True (divide advantage by group std).
# 2G-* (vanilla-loss GRPO baseline) overrides to True via env var.
NORM_ADV_BY_STD_IN_GRPO=${NORM_ADV_BY_STD_IN_GRPO:-False}

# Clip thresholds. Default asymmetric 0.2/0.27 (v2 binary-mask / MiniRL clip).
# 2G-* sets both to 0.2 for symmetric PPO-clip (canonical GRPO).
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.27}

# Rollout correction (π_fsdp/π_vllm log-prob mismatch) — identical to 1A/B/C
# for the v2 WDL-SFT-IS family. The newer wdl_group_adv_is loss owns its
# detached IS term internally and explicitly forbids external rollout_is weights.
rollout_is=${ROLLOUT_IS:-token}
rollout_is_threshold=${ROLLOUT_IS_THRESHOLD:-5.0}
rollout_is_batch_normalize=${ROLLOUT_IS_BATCH_NORMALIZE:-false}
rollout_rs=${ROLLOUT_RS:-null}
rollout_rs_threshold=${ROLLOUT_RS_THRESHOLD:-null}
if [ "$LOSS_MODE" = "wdl_group_adv_is" ]; then
    rollout_is=${ROLLOUT_IS:-null}
    rollout_rs=${ROLLOUT_RS:-null}
    NORM_ADV_BY_STD_IN_GRPO=${NORM_ADV_BY_STD_IN_GRPO:-false}
fi
ALL_CORRECT_SFT_FALLBACK=${ALL_CORRECT_SFT_FALLBACK:-true}
POS_SFT_FALLBACK_COEF=${POS_SFT_FALLBACK_COEF:-1.0}

# ===================== Section 8b: Reward Manager =============================
reward_manager=${REWARD_MANAGER:-dapo}
enable_overlong_buffer=${ENABLE_OVERLONG_BUFFER:-false}
overlong_buffer_len=${OVERLONG_BUFFER_LEN:-$((1024 * 1))}
overlong_penalty_factor=${OVERLONG_PENALTY_FACTOR:-0.5}

# Reuse the reward function from the parent recipe directory (one level up).
PARENT_RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUSTOM_REWARD_FN_PATH=${CUSTOM_REWARD_FN_PATH:-"${PARENT_RECIPE_DIR}/custom_reward_function_latex_verify.py"}
CUSTOM_REWARD_FN_NAME=${CUSTOM_REWARD_FN_NAME:-"compute_score_latex_verify"}

# ===================== Section 9: Sequence Lengths ============================
max_prompt_length=${MAX_PROMPT_LENGTH:-500}
max_response_length=${MAX_RESPONSE_LENGTH:-4096}

# ===================== Section 10: Batch Sizes ================================
# Compute-matched to 1A/1B/1C: 64 prompts/step × 8 rollouts = 512 responses/step.
# mini_bsz=8 matches the joint runs exactly (same # of updates per rollout batch).
# Note: on a single model one could safely go to mini_bsz=16 (MiniRL default);
# we keep 8 to isolate "model count" as the only variable.
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-64}
n_resp_per_prompt=${ROLLOUT_N:-8}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-8}

# ===================== Section 11: Sampling Parameters ========================
temperature=${TEMPERATURE:-1.0}
top_p=${TOP_P:-1.0}
top_k=${TOP_K:--1}
val_temperature=${VAL_TEMPERATURE:-$temperature}
val_top_p=${VAL_TOP_P:-0.95}
val_n=${VAL_N:-1}

# ===================== Section 12: Performance & Memory =======================
sp_size=1
use_dynamic_bsz=True
# Single model has 1× vocab (vs joint's 2×), so the token budget can be larger.
# Default kept at 9192 for strict comparability; override via env if you need speed.
actor_ppo_max_token_len=${ACTOR_PPO_MAX_TOKEN_LEN:-9192}
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 6))
offload=False
fsdp_size=-1
LOG_PROB_MICRO_BATCH_SIZE_WAS_SET=${LOG_PROB_MICRO_BATCH_SIZE+x}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}

GENERATION_MICRO_BATCH_SIZE=${GENERATION_MICRO_BATCH_SIZE:-16}
LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-4}
ROLLOUT_ENGINE=${ROLLOUT_ENGINE:-vllm}
ROLLOUT_MODE=${ROLLOUT_MODE:-async}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-true}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((max_prompt_length + max_response_length))}
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$((max_prompt_length + max_response_length))}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-4}
ROLLOUT_ENABLE_CHUNKED_PREFILL=${ROLLOUT_ENABLE_CHUNKED_PREFILL:-true}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((max_prompt_length + max_response_length))}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-256}
ROLLOUT_CALCULATE_LOG_PROBS=${ROLLOUT_CALCULATE_LOG_PROBS:-True}

if [ "${ROLLOUT_ENGINE}" = "vllm" ]; then
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
        echo "[ablation] WARNING: flash_attn is not installed; disabling USE_REMOVE_PADDING." >&2
        USE_REMOVE_PADDING=False
        if [ -z "${LOG_PROB_MICRO_BATCH_SIZE_WAS_SET}" ]; then
            LOG_PROB_MICRO_BATCH_SIZE=1
        fi
    fi
fi

# ===================== Section 13: Training Schedule ==========================
test_freq=${TEST_FREQ:-25}
save_freq=${SAVE_FREQ:-25}
total_epochs=${TOTAL_EPOCHS:-2}
total_training_steps=${TOTAL_TRAINING_STEPS:-300}
val_before_train=${VAL_BEFORE_TRAIN:-True}

# Meituan storage budget: keep latest full checkpoint plus best model-only checkpoint.
KEEP_BEST_CKPT=${KEEP_BEST_CKPT:-True}
BEST_CKPT_METRIC_KEY=${BEST_CKPT_METRIC_KEY:-"val-core/HuggingFaceH4/MATH-500/acc/mean@1"}
BEST_CKPT_METRIC_MODE=${BEST_CKPT_METRIC_MODE:-max}
BEST_CKPT_STRIP_OPTIMIZER=${BEST_CKPT_STRIP_OPTIMIZER:-True}
PROTECTED_CKPT_STEPS=${PROTECTED_CKPT_STEPS:-}
if [ -n "$PROTECTED_CKPT_STEPS" ] && [ "$PROTECTED_CKPT_STEPS" != "null" ] && [ "$PROTECTED_CKPT_STEPS" != "None" ]; then
    if [[ "$PROTECTED_CKPT_STEPS" != \[*\] ]]; then
        PROTECTED_CKPT_STEPS="[${PROTECTED_CKPT_STEPS}]"
    fi
fi

# ===================== Section 14: Loss-specific extra args ===================
# wdl_sft_is carries an additional +policy_loss.wdl_sft_beta override.
# minirl (baseline 2Z) does not consume that field, so we omit it there.
LOSS_EXTRA_ARGS=()
case "$LOSS_MODE" in
    wdl_sft_is|wdl_sft)
        LOSS_EXTRA_ARGS+=("+actor_rollout_ref.actor.policy_loss.wdl_sft_beta=${WDL_SFT_BETA}")
        ;;
    wdl_group_adv_is)
        LOSS_EXTRA_ARGS+=("+actor_rollout_ref.actor.policy_loss.all_correct_sft_fallback=${ALL_CORRECT_SFT_FALLBACK}")
        LOSS_EXTRA_ARGS+=("+actor_rollout_ref.actor.policy_loss.pos_sft_fallback_coef=${POS_SFT_FALLBACK_COEF}")
        ;;
    minirl|vanilla|gspo|sapo|gpg|clip_cov|kl_cov|geo_mean|cispo|bypass_mode)
        : # no extra args needed
        ;;
    *)
        echo "WARNING: unrecognized LOSS_MODE='$LOSS_MODE' — proceeding without extra args." >&2
        ;;
esac

# ==============================================================================
# Section 15: Launch Training
# ==============================================================================
python3 -m verl.trainer.main_ppo \
    \
    `# --- Algorithm ---` \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.norm_adv_by_std_in_grpo=${NORM_ADV_BY_STD_IN_GRPO} \
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
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy=True \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.grad_clip=500.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${LOSS_MODE} \
    "${LOSS_EXTRA_ARGS[@]}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    \
    `# --- Reference model (disabled) ---` \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    \
    `# --- Model (single-model path: joint_training stays at default=False) ---` \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.joint_training=${JOINT_TRAINING:-False} \
    actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING} \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
    \
    `# --- Rollout (vLLM, standard single-model) ---` \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.calculate_log_probs=${ROLLOUT_CALCULATE_LOG_PROBS} \
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
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n} \
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
    data.train_max_samples=${TRAIN_MAX_SAMPLES} \
    data.val_max_samples=${VAL_MAX_SAMPLES} \
    \
    `# --- Reward ---` \
    reward_model.reward_manager=${reward_manager} \
    +reward.timeout=${REWARD_TIMEOUT:-300} \
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
    +trainer.keep_best_ckpt=${KEEP_BEST_CKPT} \
    +trainer.best_ckpt_metric_key="${BEST_CKPT_METRIC_KEY}" \
    +trainer.best_ckpt_metric_mode=${BEST_CKPT_METRIC_MODE} \
    +trainer.best_ckpt_strip_optimizer=${BEST_CKPT_STRIP_OPTIMIZER} \
    +trainer.protected_ckpt_steps="${PROTECTED_CKPT_STEPS}" \
    +trainer.protected_ckpt_strip_optimizer=${PROTECTED_CKPT_STRIP_OPTIMIZER:-False} \
    trainer.log_val_generations=${VAL_GENERATIONS_TO_LOG} \
    +trainer.log_val_generations_to_tracking=${VAL_GENERATIONS_TO_TRACKING} \
    trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
    trainer.resume_mode=auto \
    \
    "$@" 2>&1 | tee "$LOG_FILE"
