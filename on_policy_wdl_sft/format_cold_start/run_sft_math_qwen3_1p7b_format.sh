#!/usr/bin/env bash
# Format cold-start SFT on MATH with Qwen3-1.7B.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export RUN_PREFIX=${RUN_PREFIX:-SFT-FORMAT-COLDSTART-Qwen3-1P7B-MATH-V1}
export RUN_NAME=${RUN_NAME:-${RUN_PREFIX}_$(date +%s)}
export MODEL_PATH=${MODEL_PATH:-/data-1/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
export TRAIN_FILE=${TRAIN_FILE:-/data-1/dataset/math/format_cold_start/math_sft_messages.parquet}
export CKPT_ROOT=${CKPT_ROOT:-/data-1/checkpoints/format_cold_start}
export WANDB_PROJECT=${WANDB_PROJECT:-FormatColdStartSFT}
export WANDB_MODE=${WANDB_MODE:-offline}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-80}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
export MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
export MAX_LENGTH=${MAX_LENGTH:-4096}
export MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-4096}
export LR=${LR:-2e-5}
export SAVE_FREQ=${SAVE_FREQ:-20}
export TEST_FREQ=${TEST_FREQ:--1}
export MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP:-2}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
export VAL_FILES=${VAL_FILES:-null}
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}
export DATA_NUM_WORKERS=${DATA_NUM_WORKERS:-8}
export DATA_SHUFFLE=${DATA_SHUFFLE:-False}
export LOSS_MASK_PREFLIGHT_SAMPLES=${LOSS_MASK_PREFLIGHT_SAMPLES:--1}
export LOSS_MASK_PREFLIGHT_DIR=${LOSS_MASK_PREFLIGHT_DIR:-/data-1/tmp/verl_agent_scratch/sft_loss_mask_preflight}
export TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-False}
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "[format-sft-math] ERROR: MODEL_PATH not found: $MODEL_PATH" >&2
    exit 1
fi
if [ ! -f "$TRAIN_FILE" ]; then
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[format-sft-math] WARNING: TRAIN_FILE not found during DRY_RUN: $TRAIN_FILE" >&2
    else
        echo "[format-sft-math] ERROR: TRAIN_FILE not found: $TRAIN_FILE" >&2
        exit 1
    fi
fi
if [ "${DRY_RUN:-0}" != "1" ] && [ "${ALLOW_FORMAT_COLD_START_SFT:-0}" != "1" ]; then
    echo "[format-sft-math] ERROR: non-dry-run requires ALLOW_FORMAT_COLD_START_SFT=1" >&2
    exit 1
fi

CMD=(
    python3 -m verl.trainer.sft_trainer_ray
    "model.path=${MODEL_PATH}"
    "model.tokenizer_path=${MODEL_PATH}"
    "model.trust_remote_code=${TRUST_REMOTE_CODE}"
    "model.enable_gradient_checkpointing=True"
    "model.enable_activation_offload=False"
    "engine.strategy=fsdp"
    "engine.dtype=bfloat16"
    "engine.model_dtype=fp32"
    "engine.fsdp_size=-1"
    "engine.param_offload=false"
    "engine.optimizer_offload=false"
    "optim.lr=${LR}"
    "optim.lr_warmup_steps=5"
    "optim.weight_decay=0.1"
    "optim.clip_grad=1.0"
    "data.train_files=${TRAIN_FILE}"
    "data.val_files=${VAL_FILES}"
    "data.messages_key=messages"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU}"
    "data.max_length=${MAX_LENGTH}"
    "data.max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}"
    "data.train_max_samples=${TRAIN_MAX_SAMPLES}"
    "data.val_max_samples=${VAL_MAX_SAMPLES}"
    "data.truncation=error"
    "data.num_workers=${DATA_NUM_WORKERS}"
    "+data.shuffle=${DATA_SHUFFLE}"
    "data.tokenize_whole_message=True"
    "data.ignore_input_ids_mismatch=False"
    "trainer.project_name=${WANDB_PROJECT}"
    "trainer.experiment_name=${RUN_NAME}"
    "trainer.default_local_dir=${CKPT_ROOT}/${RUN_NAME}"
    "trainer.logger=['console','wandb']"
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.max_ckpt_to_keep=${MAX_CKPT_TO_KEEP}"
    "trainer.n_gpus_per_node=${NGPUS_PER_NODE}"
    "trainer.nnodes=1"
    "trainer.resume_mode=auto"
)

printf '[format-sft-math] command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[format-sft-math] DRY_RUN=1; exiting before training"
    exit 0
fi

cd "$REPO_ROOT"
mkdir -p "$LOSS_MASK_PREFLIGHT_DIR"
python3 scripts/validate_sft_loss_mask.py \
    --model "$MODEL_PATH" \
    --dataset "$TRAIN_FILE" \
    --output "$LOSS_MASK_PREFLIGHT_DIR/${RUN_NAME}.json" \
    --samples "$LOSS_MASK_PREFLIGHT_SAMPLES" \
    --max-length "$MAX_LENGTH"
exec "${CMD[@]}" "$@"
