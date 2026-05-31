#!/usr/bin/env bash
# Stage 2 fast validation: Model2-only rollout + fused joint WDL-SFT loss.

set -xeuo pipefail

: "${RUN_PREFIX:?RUN_PREFIX must be set by the caller}"
: "${WDL_SFT_BETA:?WDL_SFT_BETA must be set by the caller}"
: "${STAGE1_CKPT_DIR:?STAGE1_CKPT_DIR must pin the Stage 1 checkpoint dir}"
: "${STAGE1_STEP:?STAGE1_STEP must pin the Stage 1 best step}"

WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WRAPPER_SCRIPT_DIR

export LOSS_MODE=${LOSS_MODE:-wdl_sft}
export LR=${LR:-5e-7}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-75}
export WANDB_PROJECT=${WANDB_PROJECT:-"OnPolicySFT-Then-WDLSFT-StagedV1"}
export WANDB_MODE=${WANDB_MODE:-offline}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export TEST_FREQ=${TEST_FREQ:-5}
export SAVE_FREQ=${SAVE_FREQ:-5}
export VAL_N=${VAL_N:-3}
export DATA_SEED=${DATA_SEED:-20260528}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
export TRAIN_FILE=${TRAIN_FILE:-"/data-1/dataset/EnsembleLLM-data-processed/staged_v1/stage2_after_s1_150steps_seed20260528_75steps.parquet"}
export ROLLOUT_IS=${ROLLOUT_IS:-null}
export ROLLOUT_RS=${ROLLOUT_RS:-null}
export ROLLOUT_IS_THRESHOLD=${ROLLOUT_IS_THRESHOLD:-5.0}
export ROLLOUT_IS_BATCH_NORMALIZE=${ROLLOUT_IS_BATCH_NORMALIZE:-false}
export MAX_ACTOR_CKPTS_TO_KEEP=${MAX_ACTOR_CKPTS_TO_KEEP:-1}
export MAX_CRITIC_CKPTS_TO_KEEP=${MAX_CRITIC_CKPTS_TO_KEEP:-1}
export KEEP_BEST_CKPT=${KEEP_BEST_CKPT:-True}
export BEST_CKPT_METRIC_KEY=${BEST_CKPT_METRIC_KEY:-"val-core/HuggingFaceH4/MATH-500/acc/mean@3"}
export BEST_CKPT_METRIC_MODE=${BEST_CKPT_METRIC_MODE:-max}
export BEST_CKPT_STRIP_OPTIMIZER=${BEST_CKPT_STRIP_OPTIMIZER:-True}
export JOINT_TRAINING_ROLLOUT_SOURCE=${JOINT_TRAINING_ROLLOUT_SOURCE:-model2}
export ROLLOUT_CALCULATE_LOG_PROBS=${ROLLOUT_CALCULATE_LOG_PROBS:-True}
export CALCULATE_ENTROPY=${CALCULATE_ENTROPY:-False}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}
export TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-64}
export ROLLOUT_N=${ROLLOUT_N:-8}
export TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ:-$((TRAIN_PROMPT_BSZ * ROLLOUT_N))}
export ACTOR_PPO_EPOCHS=${ACTOR_PPO_EPOCHS:-1}
export ACTOR_SHUFFLE=${ACTOR_SHUFFLE:-false}

if [ ! -d "$STAGE1_CKPT_DIR" ]; then
    echo "ERROR: STAGE1_CKPT_DIR not found: $STAGE1_CKPT_DIR" >&2
    exit 1
fi

FSDP_ACTOR_DIR="${STAGE1_CKPT_DIR}/global_step_${STAGE1_STEP}/actor"
if [ ! -d "$FSDP_ACTOR_DIR" ]; then
    echo "ERROR: pinned Stage 1 actor checkpoint not found: $FSDP_ACTOR_DIR" >&2
    exit 1
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: Stage 2 non-overlap train shard not found: $TRAIN_FILE" >&2
    exit 1
fi

export STAGE1_MERGED_MODEL_ROOT=${STAGE1_MERGED_MODEL_ROOT:-/data-1/model_weights/staged_v1}
export MERGED_MODEL2_DIR=${MERGED_MODEL2_DIR:-"${STAGE1_MERGED_MODEL_ROOT}/$(basename "$STAGE1_CKPT_DIR")/step_${STAGE1_STEP}"}
MODEL2_PATH_FOR_CONFIG=${MODEL2_PATH:-"$MERGED_MODEL2_DIR"}

MODEL2_CACHE_TAG=$(basename "$MODEL2_PATH_FOR_CONFIG")
MODEL2_CACHE_TAG=${MODEL2_CACHE_TAG//[^[:alnum:]._-]/-}
if [ -z "${HF_HOME+x}" ] || [ "$HF_HOME" = "/root/.cache/huggingface" ]; then
    STAGE2_HF_HOME="/data-1/.cache/huggingface"
else
    STAGE2_HF_HOME="$HF_HOME"
fi
export MODEL_PATH=${MODEL_PATH:-"${STAGE2_HF_HOME}/QwenJoint-4B-Stage2-${MODEL2_CACHE_TAG}"}
export DATA_SHUFFLE=${DATA_SHUFFLE:-False}

print_stage2_config() {
    cat <<EOF
[STAGE2 CONFIG]
RUN_PREFIX=$RUN_PREFIX
TRAIN_FILE=$TRAIN_FILE
BASE_MODEL_PATH=${BASE_MODEL_PATH:-/data-1/.cache/huggingface/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539}
STAGE1_CKPT_DIR=$STAGE1_CKPT_DIR
STAGE1_STEP=$STAGE1_STEP
FSDP_ACTOR_DIR=$FSDP_ACTOR_DIR
MODEL2_PATH=${MODEL2_PATH:-$MERGED_MODEL2_DIR}
MODEL_PATH=$MODEL_PATH
ROLLOUT_ENGINE_ARCHITECTURE_PATH=$MODEL_PATH
ROLLOUT_WEIGHT_SOURCE_MODEL2_PATH=${MODEL2_PATH:-$MERGED_MODEL2_DIR}
LOSS_MODE=$LOSS_MODE
WDL_SFT_BETA=$WDL_SFT_BETA
JOINT_TRAINING=True
JOINT_TRAINING_ROLLOUT_SOURCE=$JOINT_TRAINING_ROLLOUT_SOURCE
ROLLOUT_SOURCE=model2-only
ACTOR_TRAINING_MODEL=joint
CALCULATE_ENTROPY=$CALCULATE_ENTROPY
ROLLOUT_GPU_MEMORY_UTILIZATION=$ROLLOUT_GPU_MEMORY_UTILIZATION
ROLLOUT_IS=$ROLLOUT_IS
ROLLOUT_RS=$ROLLOUT_RS
TRAIN_PROMPT_BSZ=$TRAIN_PROMPT_BSZ
ROLLOUT_N=$ROLLOUT_N
TRAIN_PROMPT_MINI_BSZ=$TRAIN_PROMPT_MINI_BSZ
ACTOR_PPO_EPOCHS=$ACTOR_PPO_EPOCHS
ACTOR_SHUFFLE=$ACTOR_SHUFFLE
TOTAL_TRAINING_STEPS=$TOTAL_TRAINING_STEPS
VAL_BEFORE_TRAIN=$VAL_BEFORE_TRAIN
TEST_FREQ=$TEST_FREQ
SAVE_FREQ=$SAVE_FREQ
VAL_N=$VAL_N
TRAIN_MAX_SAMPLES=$TRAIN_MAX_SAMPLES
WANDB_PROJECT=$WANDB_PROJECT
WANDB_MODE=$WANDB_MODE
[STAGE2 HYDRA OVERRIDES]
data.train_batch_size=$TRAIN_PROMPT_BSZ
actor_rollout_ref.rollout.n=$ROLLOUT_N
actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_PROMPT_MINI_BSZ
actor_rollout_ref.actor.ppo_epochs=$ACTOR_PPO_EPOCHS
actor_rollout_ref.actor.shuffle=$ACTOR_SHUFFLE
actor_rollout_ref.model.joint_training_rollout_source=$JOINT_TRAINING_ROLLOUT_SOURCE
actor_rollout_ref.rollout.calculate_log_probs=$ROLLOUT_CALCULATE_LOG_PROBS
EOF
}

load_check_model2() {
    python3 - "$MODEL2_PATH" <<'PY'
import json
import os
import sys
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer

path = Path(sys.argv[1])
cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
summary = {
    "path": str(path),
    "model_type": getattr(cfg, "model_type", None),
    "architectures": getattr(cfg, "architectures", None),
    "hidden_size": getattr(cfg, "hidden_size", None),
    "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
    "vocab_size": getattr(cfg, "vocab_size", None),
    "tokenizer_class": tok.__class__.__name__,
    "has_model_safetensors": (path / "model.safetensors").exists(),
    "has_model_index": (path / "model.safetensors.index.json").exists(),
}
if summary["model_type"] != "qwen3":
    raise SystemExit(f"unexpected model_type for Model2: {summary}")
if not (summary["has_model_safetensors"] or summary["has_model_index"]):
    raise SystemExit(f"missing safetensors weights for Model2: {summary}")
print("[STAGE2 MODEL2 LOAD CHECK] " + json.dumps(summary, sort_keys=True))
PY
}

print_stage2_config

if [ "${STAGE2_DRY_RUN:-0}" = "1" ]; then
    echo "[STAGE2 DRY RUN] exiting before joint preparation/training"
    exit 0
fi

# shellcheck disable=SC1091
source "${WRAPPER_SCRIPT_DIR}/_resolve_stage1_model2.sh"

load_check_model2

if [ "${STAGE2_MERGE_ONLY:-0}" = "1" ]; then
    echo "[STAGE2 MERGE ONLY] Model2 merge/load check complete"
    exit 0
fi

# shellcheck disable=SC1091
source "${WRAPPER_SCRIPT_DIR}/../_common_wdl_sft_is_joint.sh" \
    data.seed=${DATA_SEED} \
    data.train_max_samples=${TRAIN_MAX_SAMPLES} \
    data.shuffle=${DATA_SHUFFLE} \
    +actor_rollout_ref.model.joint_training_rollout_source=${JOINT_TRAINING_ROLLOUT_SOURCE} \
    actor_rollout_ref.actor.ppo_epochs=${ACTOR_PPO_EPOCHS} \
    actor_rollout_ref.actor.shuffle=${ACTOR_SHUFFLE} \
    actor_rollout_ref.rollout.calculate_log_probs=${ROLLOUT_CALCULATE_LOG_PROBS} \
    "$@"
