#!/usr/bin/env bash
set -xeuo pipefail

echo "Environment ready (Docker/uv mode)."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${RECIPE_ROOT}/../.." && pwd)}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

export CACHE_ROOT=${CACHE_ROOT:-/data-1/.cache}
export DATA_ROOT=${DATA_ROOT:-/data-1/dataset}
if [ "${HF_HOME:-}" = "" ] || [ "${HF_HOME:-}" = "/root/.cache/huggingface" ]; then
    export HF_HOME="${CACHE_ROOT}/huggingface"
else
    export HF_HOME
fi
export RAY_TMPDIR=${RAY_TMPDIR:-/data-1/ray_tmp}
export TMPDIR=${TMPDIR:-/data-1/tmp}
export VLLM_CONFIG_ROOT=${VLLM_CONFIG_ROOT:-/data-1/.config/vllm}
export VERL_ZMQ_IPC_DIR=${VERL_ZMQ_IPC_DIR:-$TMPDIR}
mkdir -p "$HF_HOME" "$RAY_TMPDIR" "$TMPDIR" "$VLLM_CONFIG_ROOT" "$VERL_ZMQ_IPC_DIR"

export LD_LIBRARY_PATH="$(python3 -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"
export NCCL_IBEXT_DISABLE=${NCCL_IBEXT_DISABLE:-1}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-3600}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASHINFER}
export VLLM_NO_USAGE_STATS=${VLLM_NO_USAGE_STATS:-1}
export VLLM_DO_NOT_TRACK=${VLLM_DO_NOT_TRACK:-1}
export RAY_LOGGING_LEVEL=${RAY_LOGGING_LEVEL:-WARNING}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}

RUN_PREFIX=${RUN_PREFIX:-"WDL-SFT-Qwen3-4B-MATH-3A-DUAL-M2-BETA0"}
export WANDB_PROJECT=${WANDB_PROJECT:-"OnPolicyWDLSFT"}
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_PREFIX}_$(date +%s)}"
export WANDB_HTTP_TIMEOUT=${WANDB_HTTP_TIMEOUT:-60}
export WANDB_API_TIMEOUT=${WANDB_API_TIMEOUT:-60}
export WANDB_MAX_RETRIES=${WANDB_MAX_RETRIES:-10}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-"/data-1/wandb_runs/${RUN_PREFIX}"}
mkdir -p "$WANDB_DIR"

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

BASE_MODEL_PATH=${BASE_MODEL_PATH:-"${HF_HOME}/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539"}
MODEL2_PATH=${MODEL2_PATH:-"${CACHE_ROOT}/Qwen3-4B-Base-SFT-stage-1"}
FUSION_LAMBDA=${FUSION_LAMBDA:-0.50}
MODEL2_CACHE_TAG=$(basename "$MODEL2_PATH")
MODEL2_CACHE_TAG=${MODEL2_CACHE_TAG//[^[:alnum:]._-]/-}
MODEL_PATH=${MODEL_PATH:-"${HF_HOME}/QwenJoint-4B-WDL-SFT-${MODEL2_CACHE_TAG}"}

TRAIN_FILE=${TRAIN_FILE:-"${DATA_ROOT}/EnsembleLLM-data-processed/train_rl_format.parquet"}
TEST_FILES=${TEST_FILES:-"['${DATA_ROOT}/MATH-500/math500-test_with_system_prompt.parquet','${DATA_ROOT}/AIME-2025/aime-2025_with_system_prompt.parquet']"}

if [ ! -d "$MODEL_PATH" ]; then
    echo "Joint model not found at $MODEL_PATH. Preparing from base + strong models..."
    [ -d "$BASE_MODEL_PATH" ] || { echo "ERROR: Weak model not found at $BASE_MODEL_PATH" >&2; exit 1; }
    [ -d "$MODEL2_PATH" ] || { echo "ERROR: Strong model not found at $MODEL2_PATH" >&2; exit 1; }
    python3 -m verl.models.joint_model.prepare_joint_weights \
        --base_model_path "$BASE_MODEL_PATH" \
        --model2_path "$MODEL2_PATH" \
        --output_path "$MODEL_PATH" \
        --fusion_lambda "$FUSION_LAMBDA"
fi

MIN_FREE_GB_FOR_CKPT=${MIN_FREE_GB_FOR_CKPT:-160}
MIN_FREE_KB_FOR_CKPT=$((MIN_FREE_GB_FOR_CKPT * 1024 * 1024))
# Keep storage bounded by default: one latest checkpoint plus the best checkpoint
# tracked by trainer.keep_best_ckpt.
MAX_ACTOR_CKPTS_TO_KEEP=${MAX_ACTOR_CKPTS_TO_KEEP:-1}
MAX_CRITIC_CKPTS_TO_KEEP=${MAX_CRITIC_CKPTS_TO_KEEP:-1}
BASE_CKPT_DIR=${BASE_CKPT_DIR:-/data-1/checkpoints}

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

BASE_CKPT_FREE_KB=$(get_free_kb "$BASE_CKPT_DIR")
if [ "$BASE_CKPT_FREE_KB" -lt "$MIN_FREE_KB_FOR_CKPT" ]; then
    echo "ERROR: ${BASE_CKPT_DIR} has only $((BASE_CKPT_FREE_KB / 1024 / 1024)) GiB free." >&2
    exit 1
fi
mkdir -p "$BASE_CKPT_DIR"

LATEST_CKPT_DIR=$(find "$BASE_CKPT_DIR" -maxdepth 1 -type d -name "${RUN_PREFIX}_*" 2>/dev/null | sort | tail -1)
if [ -n "$LATEST_CKPT_DIR" ] && [ -d "$LATEST_CKPT_DIR" ]; then
    EXPERIMENT_NAME=$(basename "$LATEST_CKPT_DIR")
    export WANDB_RUN_NAME="$EXPERIMENT_NAME"
    CKPTS_DIR="$LATEST_CKPT_DIR"
    IS_RESUME=true
else
    CKPTS_DIR="$BASE_CKPT_DIR/${WANDB_RUN_NAME}"
    mkdir -p "$CKPTS_DIR"
    IS_RESUME=false
fi

LOG_DIR=${LOG_DIR:-${SCRIPT_DIR}}
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

adv_estimator=${ADV_ESTIMATOR:-grpo}
loss_agg_mode=${LOSS_AGG_MODE:-seq-mean-token-sum}
use_kl_in_reward=${USE_KL_IN_REWARD:-False}
kl_coef=${KL_COEF:-0.0}
use_kl_loss=${USE_KL_LOSS:-False}
kl_loss_coef=${KL_LOSS_COEF:-0.0}
loss_mode=${LOSS_MODE:-wdl_sft_is}
WDL_SFT_BETA=${WDL_SFT_BETA:-0.0}
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.27}
gamma_pos_sft=${GAMMA_POS_SFT:-1.0}
tis_threshold=${TIS_THRESHOLD:-5.0}
joint_rollout_sources=${JOINT_ROLLOUT_SOURCES:-"[sub_model_0,sub_model_1]"}
joint_rollout_select=${JOINT_ROLLOUT_SELECT:-sub_model_1}
joint_rollout_train_on_selected_only=${JOINT_ROLLOUT_TRAIN_ON_SELECTED_ONLY:-true}

# Dual rollout deliberately does not use rollout_is_weights as loss weights by default.
rollout_is=${ROLLOUT_IS:-null}
rollout_is_threshold=${ROLLOUT_IS_THRESHOLD:-5.0}
rollout_is_batch_normalize=${ROLLOUT_IS_BATCH_NORMALIZE:-false}
rollout_rs=${ROLLOUT_RS:-null}
rollout_rs_threshold=${ROLLOUT_RS_THRESHOLD:-null}

reward_manager=${REWARD_MANAGER:-dapo}
enable_overlong_buffer=${ENABLE_OVERLONG_BUFFER:-false}
overlong_buffer_len=${OVERLONG_BUFFER_LEN:-1024}
overlong_penalty_factor=${OVERLONG_PENALTY_FACTOR:-0.5}
CUSTOM_REWARD_FN_PATH=${CUSTOM_REWARD_FN_PATH:-"${RECIPE_ROOT}/custom_reward_function_latex_verify.py"}
CUSTOM_REWARD_FN_NAME=${CUSTOM_REWARD_FN_NAME:-compute_score_latex_verify}

max_prompt_length=${MAX_PROMPT_LENGTH:-500}
max_response_length=${MAX_RESPONSE_LENGTH:-4096}
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-64}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-8}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}

temperature=${TEMPERATURE:-1.0}
top_p=${TOP_P:-1.0}
top_k=${TOP_K:--1}
val_top_p=${VAL_TOP_P:-0.95}

sp_size=${SP_SIZE:-1}
use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}
actor_ppo_max_token_len=${ACTOR_PPO_MAX_TOKEN_LEN:-9192}
infer_ppo_max_token_len=${INFER_PPO_MAX_TOKEN_LEN:-$(((max_prompt_length + max_response_length) * 6))}
CALCULATE_ENTROPY=${CALCULATE_ENTROPY:-False}
offload=${OFFLOAD:-False}
fsdp_size=${FSDP_SIZE:--1}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}
LOG_PROB_MICRO_BATCH_SIZE_WAS_SET=${LOG_PROB_MICRO_BATCH_SIZE+x}

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
    if ! python3 -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('flash_attn') else 1)" >/dev/null 2>&1; then
        echo "[dual-rollout] WARNING: flash_attn is not installed; disabling USE_REMOVE_PADDING." >&2
        USE_REMOVE_PADDING=False
        if [ -z "${LOG_PROB_MICRO_BATCH_SIZE_WAS_SET}" ]; then
            LOG_PROB_MICRO_BATCH_SIZE=1
        fi
    fi
fi

test_freq=${TEST_FREQ:-25}
save_freq=${SAVE_FREQ:-25}
total_epochs=${TOTAL_EPOCHS:-2}
total_training_steps=${TOTAL_TRAINING_STEPS:-300}
val_before_train=${VAL_BEFORE_TRAIN:-True}
KEEP_BEST_CKPT=${KEEP_BEST_CKPT:-True}
BEST_CKPT_METRIC_KEY=${BEST_CKPT_METRIC_KEY:-"val-core/HuggingFaceH4/MATH-500/acc/mean@3"}
BEST_CKPT_METRIC_MODE=${BEST_CKPT_METRIC_MODE:-max}
BEST_CKPT_STRIP_OPTIMIZER=${BEST_CKPT_STRIP_OPTIMIZER:-True}
LR=${LR:-5e-7}

cd "$REPO_ROOT"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize} \
    algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold} \
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
    actor_rollout_ref.actor.calculate_entropy=${CALCULATE_ENTROPY} \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.grad_clip=500.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    +actor_rollout_ref.actor.policy_loss.wdl_sft_beta=${WDL_SFT_BETA} \
    +actor_rollout_ref.actor.policy_loss.gamma_pos_sft=${gamma_pos_sft} \
    +actor_rollout_ref.actor.policy_loss.tis_threshold=${tis_threshold} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING} \
    actor_rollout_ref.model.trust_remote_code=True \
    +actor_rollout_ref.model.joint_training=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
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
    actor_rollout_ref.rollout.val_kwargs.n=3 \
    +actor_rollout_ref.rollout.custom.joint_rollout_sources="${joint_rollout_sources}" \
    +actor_rollout_ref.rollout.custom.joint_rollout_select=${joint_rollout_select} \
    +actor_rollout_ref.rollout.custom.joint_rollout_train_on_selected_only=${joint_rollout_train_on_selected_only} \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILES}" \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    reward_model.reward_manager=${reward_manager} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=false \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    custom_reward_function.path="${CUSTOM_REWARD_FN_PATH}" \
    custom_reward_function.name="${CUSTOM_REWARD_FN_NAME}" \
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
    trainer.log_val_generations=${VAL_GENERATIONS_TO_LOG} \
    +trainer.log_val_generations_to_tracking=${VAL_GENERATIONS_TO_TRACKING} \
    trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
    trainer.resume_mode=auto \
    "$@" 2>&1 | tee "$LOG_FILE"
