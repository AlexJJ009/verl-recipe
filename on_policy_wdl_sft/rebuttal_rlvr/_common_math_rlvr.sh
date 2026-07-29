#!/usr/bin/env bash
# Shared standard-GRPO v2 launcher for the rebuttal MATH comparison.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
FROZEN_CONFIG_PATH="${SCRIPT_DIR}/frozen_grpo_v2.env"

: "${ARM:?ARM must be set by a thin wrapper}"
: "${INIT_CLASSIFIER:?INIT_CLASSIFIER must be set by a thin wrapper}"
: "${RUN_PREFIX:?RUN_PREFIX must be set by a thin wrapper}"
: "${INIT_MODEL_PATH:?INIT_MODEL_PATH must be set by a thin wrapper}"

export ROOT=${ROOT:-/data-1}
export DATASET_ROOT=${DATASET_ROOT:-"${ROOT}"}
export STATE_ROOT=${STATE_ROOT:-"${ROOT}"}

# shellcheck disable=SC1091
source "$FROZEN_CONFIG_PATH"

if [ "${REQUIRE_PLATFORM_RECEIPTS:-0}" = "1" ]; then
    : "${ALGORITHM_CONFIG_HASH:?platform run requires ALGORITHM_CONFIG_HASH}"
    if [ "$(sha256sum "$FROZEN_CONFIG_PATH" | awk '{print $1}')" != "$ALGORITHM_CONFIG_HASH" ]; then
        echo "ERROR: frozen GRPO config hash differs from the approved manifest" >&2
        exit 2
    fi
fi

RUN_MODE=${RUN_MODE:-formal}
INIT_PAIR=${INIT_PAIR:-I1}
RLVR_SEED=${RLVR_SEED:-20260727}
ATTEMPT_ID=${ATTEMPT_ID:-}
ALLOW_BASE_PLACEHOLDER=${ALLOW_BASE_PLACEHOLDER:-0}
DRY_RUN=${DRY_RUN:-0}
CONFIG_ONLY=${CONFIG_ONLY:-0}

case "$RUN_MODE" in
    formal|external_checkpoint_assumption|smoke) ;;
    *) echo "ERROR: RUN_MODE must be formal, external_checkpoint_assumption, or smoke" >&2; exit 2 ;;
esac

if [ "$ALLOW_BASE_PLACEHOLDER" = "1" ]; then
    if [ "$RUN_MODE" != "smoke" ]; then
        echo "ERROR: the Qwen3-4B-Base placeholder is SMOKE_ONLY" >&2
        exit 2
    fi
    : "${BASE_PLACEHOLDER_MODEL_PATH:?BASE_PLACEHOLDER_MODEL_PATH is required}"
    INIT_MODEL_PATH="$BASE_PLACEHOLDER_MODEL_PATH"
    INIT_CLASSIFIER="placeholder_base_smoke"
fi

if [ "$RUN_MODE" = "formal" ]; then
    : "${PAIRED_INIT_MANIFEST:?formal runs require PAIRED_INIT_MANIFEST}"
    : "${CHECKPOINT_RECEIPT:?formal runs require CHECKPOINT_RECEIPT}"
    : "${JOB_TAG:?formal runs require manifest-derived JOB_TAG}"
    : "${CELL_HASH:?formal runs require CELL_HASH}"
    : "${ATTEMPT_ID:?formal runs require ATTEMPT_ID}"
    if [ "$INIT_CLASSIFIER" = "placeholder_base_smoke" ]; then
        echo "ERROR: placeholder initialization cannot enter a formal cell" >&2
        exit 2
    fi
elif [ "$RUN_MODE" = "external_checkpoint_assumption" ]; then
    expected_assumption=user_approved_unrecoverable_public_checkpoint_metadata_20260728
    if [ "${EXTERNAL_PROVENANCE_ASSUMPTION:-}" != "$expected_assumption" ]; then
        echo "ERROR: external checkpoint mode requires the frozen user-approved provenance assumption" >&2
        exit 2
    fi
    JOB_TAG=${JOB_TAG:-"${ARM}-${INIT_PAIR}-r${RLVR_SEED}-external-assumed"}
    CELL_HASH=${CELL_HASH:-"EXTERNAL_PROVENANCE_ASSUMED"}
    ATTEMPT_ID=${ATTEMPT_ID:-"direct-$(date +%s)"}
else
    JOB_TAG=${JOB_TAG:-"SMOKE-${ARM}-${INIT_PAIR}-r${RLVR_SEED}"}
    CELL_HASH=${CELL_HASH:-"SMOKE_ONLY"}
    ATTEMPT_ID=${ATTEMPT_ID:-"manual-$(date +%s)"}
fi

if [ ! -d "$INIT_MODEL_PATH" ]; then
    echo "ERROR: INIT_MODEL_PATH does not exist: $INIT_MODEL_PATH" >&2
    echo "HINT: fill the authoritative path, or set RUN_MODE=smoke ALLOW_BASE_PLACEHOLDER=1 for plumbing only." >&2
    exit 2
fi

export TRAIN_FILE=${TRAIN_FILE:-"${DATASET_ROOT}/dataset/math/train_rl_format.parquet"}
export MATH7_AIME_FILE=${MATH7_AIME_FILE:-"${DATASET_ROOT}/dataset/AIME-2025/aime-2025_with_system_prompt.parquet"}
export MATH7_MATH500_FILE=${MATH7_MATH500_FILE:-"${DATASET_ROOT}/dataset/MATH-500/math500-test_with_system_prompt.parquet"}
export MATH7_AMC23_FILE=${MATH7_AMC23_FILE:-"${DATASET_ROOT}/dataset/AMC23/amc23-test_with_system_prompt.parquet"}
export MATH7_AQUA_FILE=${MATH7_AQUA_FILE:-"${DATASET_ROOT}/dataset/AQUA/aqua-test_with_system_prompt.parquet"}
export MATH7_GSM8K_FILE=${MATH7_GSM8K_FILE:-"${DATASET_ROOT}/dataset/gsm8k/gsm8k-test_with_system_prompt.parquet"}
export MATH7_MAWPS_FILE=${MATH7_MAWPS_FILE:-"${DATASET_ROOT}/dataset/MAWPS/mawps-test_with_system_prompt.parquet"}
export MATH7_SVAMP_FILE=${MATH7_SVAMP_FILE:-"${DATASET_ROOT}/dataset/SVAMP/svamp-test_with_system_prompt.parquet"}

for input_file in \
    "$TRAIN_FILE" \
    "$MATH7_AIME_FILE" \
    "$MATH7_MATH500_FILE" \
    "$MATH7_AMC23_FILE" \
    "$MATH7_AQUA_FILE" \
    "$MATH7_GSM8K_FILE" \
    "$MATH7_MAWPS_FILE" \
    "$MATH7_SVAMP_FILE"; do
    if [ ! -f "$input_file" ]; then
        echo "ERROR: required MATH input does not exist: $input_file" >&2
        exit 2
    fi
done

if [ "$RUN_MODE" = "formal" ]; then
    python3 "${SCRIPT_DIR}/validate_inputs.py" launch \
        --arm "$ARM" \
        --classifier "$INIT_CLASSIFIER" \
        --model "$INIT_MODEL_PATH" \
        --pair-manifest "$PAIRED_INIT_MANIFEST" \
        --checkpoint-receipt "$CHECKPOINT_RECEIPT"
fi

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASHINFER}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_NO_USAGE_STATS=${VLLM_NO_USAGE_STATS:-1}
export VLLM_DO_NOT_TRACK=${VLLM_DO_NOT_TRACK:-1}
export HYDRA_FULL_ERROR=1
export NCCL_IBEXT_DISABLE=${NCCL_IBEXT_DISABLE:-1}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-3600}

export OUTPUT_ROOT=${OUTPUT_ROOT:-"${STATE_ROOT}/verl-exp"}
export BASE_CKPT_DIR=${BASE_CKPT_DIR:-"${OUTPUT_ROOT}/checkpoints/rebuttal_rlvr"}
export EVAL_ROOT=${EVAL_ROOT:-"${OUTPUT_ROOT}/eval/rebuttal_rlvr"}
export LOG_DIR=${LOG_DIR:-"${OUTPUT_ROOT}/logs/rebuttal_rlvr"}
export WANDB_DIR=${WANDB_DIR:-"${OUTPUT_ROOT}/wandb_runs/rebuttal_rlvr"}
export RECEIPT_ROOT=${RECEIPT_ROOT:-"${OUTPUT_ROOT}/receipts/rebuttal_rlvr"}
export HF_HOME=${HF_HOME:-"${STATE_ROOT}/.cache/huggingface"}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-"${HF_HOME}/hub"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"${HF_HOME}/datasets"}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-"${STATE_ROOT}/.cache"}
export RAY_TMPDIR=${RAY_TMPDIR:-"${STATE_ROOT}/ray_tmp"}
export TMPDIR=${TMPDIR:-"${STATE_ROOT}/tmp"}
export VLLM_CONFIG_ROOT=${VLLM_CONFIG_ROOT:-"${XDG_CACHE_HOME}/vllm"}
export VERL_ZMQ_IPC_DIR=${VERL_ZMQ_IPC_DIR:-"${TMPDIR}"}

ATTEMPT_ROOT="${RECEIPT_ROOT}/${JOB_TAG}/attempts/${ATTEMPT_ID}"
CKPTS_DIR="${BASE_CKPT_DIR}/${JOB_TAG}/attempts/${ATTEMPT_ID}"
RUN_LOG_DIR="${LOG_DIR}/${JOB_TAG}/attempts/${ATTEMPT_ID}"
RUN_WANDB_DIR="${WANDB_DIR}/${JOB_TAG}/attempts/${ATTEMPT_ID}"
mkdir -p \
    "$ATTEMPT_ROOT" "$CKPTS_DIR" "$EVAL_ROOT" "$RUN_LOG_DIR" "$RUN_WANDB_DIR" \
    "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" \
    "$XDG_CACHE_HOME" "$RAY_TMPDIR" "$TMPDIR" "$VLLM_CONFIG_ROOT" "$VERL_ZMQ_IPC_DIR"

export WANDB_PROJECT=${WANDB_PROJECT:-Rebuttal-RLVR-MATH}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"${RUN_PREFIX}-${JOB_TAG}-${ATTEMPT_ID}"}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR="$RUN_WANDB_DIR"
if [ "${REQUIRE_PLATFORM_RECEIPTS:-0}" = "1" ]; then
    export REGISTRY_ROOT="${STATE_ROOT}/experiment_registry"
    export EXPERIMENT_REGISTRY_DB="${REGISTRY_ROOT}/experiment_registry.sqlite"
    export TRAINING_RELEASE_GATE_STATE="${REGISTRY_ROOT}/training_release_gate.jsonl"
    export RELEASE_LOG_FILE="${ATTEMPT_ROOT}/release.log"
    export RELEASE_STATUS_FILE="${ATTEMPT_ROOT}/release_status.env"
    export VERL_FILE_LOGGER_ROOT="${RUN_LOG_DIR}/metrics"
else
    export VERL_FILE_LOGGER_ROOT=${VERL_FILE_LOGGER_ROOT:-"${RUN_LOG_DIR}/metrics"}
fi

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-true}
ACTOR_PPO_MAX_TOKEN_LEN=${ACTOR_PPO_MAX_TOKEN_LEN:-9192}
GENERATION_MICRO_BATCH_SIZE=${GENERATION_MICRO_BATCH_SIZE:-16}
LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-4}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.70}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-4}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-256}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-true}
ROLLOUT_ENABLE_CHUNKED_PREFILL=${ROLLOUT_ENABLE_CHUNKED_PREFILL:-true}
ROLLOUT_MODE=${ROLLOUT_MODE:-async}
ROLLOUT_ENGINE=${ROLLOUT_ENGINE:-vllm}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-false}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-false}

if [ "$RUN_MODE" = "smoke" ]; then
    EFFECTIVE_TOTAL_STEPS=${SMOKE_TRAINING_STEPS:-5}
    EFFECTIVE_SAVE_FREQ=-1
    EFFECTIVE_LOGGER='["file"]'
else
    EFFECTIVE_TOTAL_STEPS=$TOTAL_TRAINING_STEPS
    EFFECTIVE_SAVE_FREQ=$SAVE_FREQ
    EFFECTIVE_LOGGER='["wandb","file"]'
fi

TEST_FILES="['${MATH7_AIME_FILE}','${MATH7_MATH500_FILE}','${MATH7_AMC23_FILE}','${MATH7_AQUA_FILE}','${MATH7_GSM8K_FILE}','${MATH7_MAWPS_FILE}','${MATH7_SVAMP_FILE}']"
if [ "${REQUIRE_PLATFORM_RECEIPTS:-0}" = "1" ]; then
    # The platform worker validates this exact file/function against the bound
    # grader receipt before it reaches this launcher. Never preserve an image-
    # or client-inherited override in a formal AFO job.
    CUSTOM_REWARD_FN_PATH="${REPO_ROOT}/recipe/joint_training/custom_reward_function_latex_verify.py"
    CUSTOM_REWARD_FN_NAME="compute_score_latex_verify"
else
    CUSTOM_REWARD_FN_PATH=${CUSTOM_REWARD_FN_PATH:-"${REPO_ROOT}/recipe/joint_training/custom_reward_function_latex_verify.py"}
    CUSTOM_REWARD_FN_NAME=${CUSTOM_REWARD_FN_NAME:-compute_score_latex_verify}
fi
RESOLVED_CONFIG_PATH="${ATTEMPT_ROOT}/resolved_config.yaml"
LOG_FILE="${RUN_LOG_DIR}/${WANDB_RUN_NAME}.log"

HYDRA_ARGS=(
    "algorithm.adv_estimator=${ADV_ESTIMATOR}"
    "algorithm.norm_adv_by_std_in_grpo=${NORM_ADV_BY_STD_IN_GRPO}"
    "algorithm.use_kl_in_reward=${USE_KL_IN_REWARD}"
    "algorithm.kl_ctrl.kl_coef=${KL_COEF}"
    "algorithm.rollout_correction.rollout_is=${ROLLOUT_IS}"
    "algorithm.rollout_correction.rollout_is_threshold=${ROLLOUT_IS_THRESHOLD}"
    "algorithm.rollout_correction.rollout_is_batch_normalize=${ROLLOUT_IS_BATCH_NORMALIZE}"
    "algorithm.rollout_correction.rollout_rs=${ROLLOUT_RS}"
    "algorithm.rollout_correction.rollout_rs_threshold=${ROLLOUT_RS_THRESHOLD}"
    "algorithm.rollout_correction.bypass_mode=${ROLLOUT_CORRECTION_BYPASS_MODE}"
    "algorithm.rollout_correction.loss_type=${ROLLOUT_CORRECTION_LOSS_TYPE}"
    "actor_rollout_ref.actor.policy_loss.loss_mode=${POLICY_LOSS_MODE}"
    "+actor_rollout_ref.actor.policy_loss.all_correct_sft_fallback=${ALL_CORRECT_SFT_FALLBACK}"
    "actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO}"
    "actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}"
    "actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}"
    "actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C}"
    "actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE}"
    "actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_PROMPT_MINI_BSZ}"
    "actor_rollout_ref.actor.optim.optimizer=${OPTIMIZER_NAME}"
    "actor_rollout_ref.actor.optim.optimizer_impl=${OPTIMIZER_IMPL}"
    "actor_rollout_ref.actor.optim.lr=${LR}"
    "actor_rollout_ref.actor.optim.weight_decay=${WEIGHT_DECAY}"
    "actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS}"
    "actor_rollout_ref.actor.optim.lr_scheduler_type=${LR_SCHEDULER_TYPE}"
    "actor_rollout_ref.actor.optim.betas=${OPTIMIZER_BETAS}"
    "actor_rollout_ref.actor.optim.zero_indexed_step=${ZERO_INDEXED_STEP}"
    "actor_rollout_ref.actor.optim.override_optimizer_config={eps:${OPTIMIZER_EPS}}"
    "actor_rollout_ref.actor.grad_clip=${GRAD_CLIP}"
    "actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}"
    "actor_rollout_ref.actor.calculate_entropy=${CALCULATE_ENTROPY}"
    "actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}"
    "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}"
    "actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE}"
    "actor_rollout_ref.actor.track_joint_submodel_losses=${TRACK_JOINT_SUBMODEL_LOSSES}"
    "actor_rollout_ref.actor.submodel_kl.enabled=${SUBMODEL_KL_ENABLED}"
    "actor_rollout_ref.actor.shuffle=${ACTOR_SHUFFLE}"
    "actor_rollout_ref.actor.data_loader_seed=${RLVR_SEED}"
    "actor_rollout_ref.actor.fsdp_config.seed=${RLVR_SEED}"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD}"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD}"
    "actor_rollout_ref.actor.use_dynamic_bsz=true"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN}"
    "actor_rollout_ref.actor.use_torch_compile=false"
    "actor_rollout_ref.ref.fsdp_config.seed=${RLVR_SEED}"
    "actor_rollout_ref.ref.fsdp_config.param_offload=false"
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true"
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE}"
    "actor_rollout_ref.model.path=${INIT_MODEL_PATH}"
    "+actor_rollout_ref.model.joint_training=${JOINT_TRAINING}"
    "actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING}"
    "actor_rollout_ref.model.trust_remote_code=true"
    "actor_rollout_ref.model.enable_gradient_checkpointing=true"
    "+actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2"
    "actor_rollout_ref.rollout.name=${ROLLOUT_ENGINE}"
    "actor_rollout_ref.rollout.mode=${ROLLOUT_MODE}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "+actor_rollout_ref.rollout.seed=${RLVR_SEED}"
    "actor_rollout_ref.rollout.temperature=${TEMPERATURE}"
    "actor_rollout_ref.rollout.top_p=${TOP_P}"
    "actor_rollout_ref.rollout.top_k=${TOP_K}"
    "actor_rollout_ref.rollout.do_sample=${ROLLOUT_DO_SAMPLE}"
    "actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}"
    "actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
    "actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER}"
    "actor_rollout_ref.rollout.enable_chunked_prefill=${ROLLOUT_ENABLE_CHUNKED_PREFILL}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_MODEL_LEN}"
    "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
    "actor_rollout_ref.rollout.agent.num_workers=${ROLLOUT_AGENT_NUM_WORKERS}"
    "actor_rollout_ref.rollout.calculate_log_probs=true"
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE}"
    "+actor_rollout_ref.rollout.micro_batch_size=${GENERATION_MICRO_BATCH_SIZE}"
    "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}"
    "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}"
    "actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K}"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE}"
    "actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}"
    "data.train_files=${TRAIN_FILE}"
    "data.val_files=${TEST_FILES}"
    "data.prompt_key=prompt"
    "data.filter_overlong_prompts=true"
    "data.truncation=left"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.train_batch_size=${TRAIN_PROMPT_BSZ}"
    "data.shuffle=${DATA_SHUFFLE}"
    "data.seed=${RLVR_SEED}"
    "reward_model.reward_manager=${REWARD_MANAGER}"
    "+reward_model.reward_kwargs.overlong_buffer_cfg.enable=false"
    "+reward_model.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}"
    "custom_reward_function.path=${CUSTOM_REWARD_FN_PATH}"
    "custom_reward_function.name=${CUSTOM_REWARD_FN_NAME}"
    "trainer.logger=${EFFECTIVE_LOGGER}"
    "trainer.project_name=${WANDB_PROJECT}"
    "trainer.experiment_name=${WANDB_RUN_NAME}"
    "trainer.n_gpus_per_node=${NGPUS_PER_NODE}"
    "trainer.nnodes=${NNODES}"
    "trainer.val_before_train=${VAL_BEFORE_TRAIN}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.save_freq=${EFFECTIVE_SAVE_FREQ}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.total_training_steps=${EFFECTIVE_TOTAL_STEPS}"
    "trainer.default_local_dir=${CKPTS_DIR}"
    "trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPTS_TO_KEEP}"
    "+trainer.keep_best_ckpt=${KEEP_BEST_CKPT}"
    "+trainer.best_ckpt_metric_key=${BEST_CKPT_METRIC_KEY}"
    "+trainer.best_ckpt_metric_mode=${BEST_CKPT_METRIC_MODE}"
    "+trainer.best_ckpt_strip_optimizer=${BEST_CKPT_STRIP_OPTIMIZER}"
    "trainer.resume_mode=disable"
)

printf '%s\n' \
    "config_version=${REBUTTAL_GRPO_CONFIG_VERSION}" \
    "arm=${ARM}" \
    "classifier=${INIT_CLASSIFIER}" \
    "init_pair=${INIT_PAIR}" \
    "rlvr_seed=${RLVR_SEED}" \
    "job_tag=${JOB_TAG}" \
    "cell_hash=${CELL_HASH}" \
    "attempt_id=${ATTEMPT_ID}" \
    "init_model=${INIT_MODEL_PATH}" \
    "run_mode=${RUN_MODE}" \
    >"${ATTEMPT_ROOT}/launch_identity.env"
if [ "$RUN_MODE" = "external_checkpoint_assumption" ]; then
    printf '%s\n' \
        'policy=conditional_checkpoint_comparison' \
        'decision_date=2026-07-28' \
        'initialization_dataset_assumption=AM-1.4M' \
        'unavailable_fields=initialization_seed,optimizer,training_steps,data_receipt,checkpoint_selection' \
        'claim_boundary=results_apply_only_to_the_two_supplied_checkpoints' \
        "model_path=${INIT_MODEL_PATH}" \
        >"${ATTEMPT_ROOT}/external_provenance_assumption.env"
fi
sha256sum "$FROZEN_CONFIG_PATH" >"${ATTEMPT_ROOT}/frozen_grpo_v2.env.sha256"

if [ "$DRY_RUN" = "1" ]; then
    printf 'python3 -m verl.trainer.main_ppo'
    printf ' %q' "${HYDRA_ARGS[@]}"
    printf '\n'
    exit 0
fi

cd "$REPO_ROOT"
python3 -m verl.trainer.main_ppo --cfg job --resolve "${HYDRA_ARGS[@]}" >"$RESOLVED_CONFIG_PATH"
python3 "${SCRIPT_DIR}/validate_inputs.py" resolved-config \
    --config "$RESOLVED_CONFIG_PATH" \
    --seed "$RLVR_SEED" \
    --model "$INIT_MODEL_PATH" \
    --total-training-steps "$EFFECTIVE_TOTAL_STEPS" \
    --save-freq "$EFFECTIVE_SAVE_FREQ"

if [ "$CONFIG_ONLY" = "1" ]; then
    echo "Resolved configuration validated: $RESOLVED_CONFIG_PATH"
    exit 0
fi

python3 -m verl.trainer.main_ppo "${HYDRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

if [ "$RUN_MODE" = "formal" ] || [ "$RUN_MODE" = "external_checkpoint_assumption" ]; then
    export CKPTS_DIR RUN_LOG_DIR RUN_WANDB_DIR ATTEMPT_ROOT WANDB_RUN_NAME
    if bash "${SCRIPT_DIR}/release_after_success.sh"; then
        echo "Automatic release completed: ${ATTEMPT_ROOT}/release_status.env"
    else
        release_rc=$?
        printf '%s\n' \
            'release_status=failed' \
            "release_rc=${release_rc}" \
            "training_status=success_complete" \
            >"${ATTEMPT_ROOT}/release_status.env"
        echo "WARNING: training completed, but automatic release failed; see ${ATTEMPT_ROOT}/release.log" >&2
    fi
fi
