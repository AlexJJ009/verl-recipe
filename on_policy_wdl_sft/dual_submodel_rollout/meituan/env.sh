#!/usr/bin/env bash
# ==============================================================================
# Meituan MLP path overrides for dual_submodel_rollout runs.
#
# Source this before invoking any run_*.sh in dual_submodel_rollout on AFO
# worker pods. The run scripts stay default-local and all persistent paths are
# redirected to dolphinfs here.
# ==============================================================================

LGX=${LGX:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yangfengkai02/lgx}

# High-churn temp dirs stay container-local.
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_tmp}
export TMPDIR=${TMPDIR:-/tmp/verl_tmp}
export VLLM_CONFIG_ROOT=${VLLM_CONFIG_ROOT:-/tmp/vllm_config}
export VERL_ZMQ_IPC_DIR=${VERL_ZMQ_IPC_DIR:-$TMPDIR}

# Persistent cache/state on dolphinfs.
export CACHE_ROOT=${CACHE_ROOT:-$LGX/verl-exp/cache}
export DATA_ROOT=${DATA_ROOT:-$LGX/verl-exp/data}
export HF_HOME=${HF_HOME:-$LGX/verl-exp/hf_cache}
export BASE_CKPT_DIR=${BASE_CKPT_DIR:-$LGX/verl-exp/checkpoints}
export WANDB_DIR=${WANDB_DIR:-$LGX/verl-exp/wandb_runs}
export LOG_DIR=${LOG_DIR:-$LGX/verl-exp/logs/dual_submodel_rollout}
export WANDB_MODE=${WANDB_MODE:-offline}
export MIN_FREE_GB_FOR_CKPT=${MIN_FREE_GB_FOR_CKPT:-60}
export MAX_ACTOR_CKPTS_TO_KEEP=${MAX_ACTOR_CKPTS_TO_KEEP:-1}
export MAX_CRITIC_CKPTS_TO_KEEP=${MAX_CRITIC_CKPTS_TO_KEEP:-1}
export KEEP_BEST_CKPT=${KEEP_BEST_CKPT:-True}
export BEST_CKPT_METRIC_KEY=${BEST_CKPT_METRIC_KEY:-val-core/HuggingFaceH4/MATH-500/acc/mean@3}
export BEST_CKPT_METRIC_MODE=${BEST_CKPT_METRIC_MODE:-max}
export BEST_CKPT_STRIP_OPTIMIZER=${BEST_CKPT_STRIP_OPTIMIZER:-True}

# Init-model paths: flat dirs, no HF cache symlink layout on dolphinfs.
export MEITUAN_BASE_MODEL_PATH=${MEITUAN_BASE_MODEL_PATH:-$LGX/huggingface.co/Qwen/Qwen3-4B-Base}
export MEITUAN_SFT_MODEL_PATH=${MEITUAN_SFT_MODEL_PATH:-$LGX/huggingface.co/Qwen/Qwen3-4B-Base-SFT-stage-1/flat}
export BASE_MODEL_PATH=${BASE_MODEL_PATH:-$MEITUAN_BASE_MODEL_PATH}
export MODEL2_PATH=${MODEL2_PATH:-$MEITUAN_SFT_MODEL_PATH}
export MODEL_PATH=${MODEL_PATH:-$LGX/verl-exp/models/QwenJoint-4B-WDL-SFT-Qwen3-4B-Base-SFT-stage-1}

# MATH train and validation data used by the 4A production run.
export TRAIN_FILE=${TRAIN_FILE:-$LGX/verl-exp/data/math/train_rl_format.parquet}
export TEST_FILES=${TEST_FILES:-"['$LGX/verl-exp/data/MATH-500/math500-test_with_system_prompt.parquet','$LGX/verl-exp/data/AIME-2025/aime-2025_with_system_prompt.parquet']"}

mkdir -p \
    "$CACHE_ROOT" \
    "$HF_HOME" \
    "$BASE_CKPT_DIR" \
    "$WANDB_DIR" \
    "$LOG_DIR" \
    "$RAY_TMPDIR" \
    "$TMPDIR" \
    "$VLLM_CONFIG_ROOT" \
    "$VERL_ZMQ_IPC_DIR" \
    "$(dirname "$MODEL_PATH")"

echo "[dual/meituan/env.sh] LGX              = $LGX"
echo "[dual/meituan/env.sh] CACHE_ROOT       = $CACHE_ROOT"
echo "[dual/meituan/env.sh] HF_HOME          = $HF_HOME"
echo "[dual/meituan/env.sh] BASE_CKPT_DIR    = $BASE_CKPT_DIR"
echo "[dual/meituan/env.sh] WANDB_DIR        = $WANDB_DIR"
echo "[dual/meituan/env.sh] LOG_DIR          = $LOG_DIR"
echo "[dual/meituan/env.sh] BASE_MODEL_PATH  = $BASE_MODEL_PATH"
echo "[dual/meituan/env.sh] MODEL2_PATH      = $MODEL2_PATH"
echo "[dual/meituan/env.sh] MODEL_PATH       = $MODEL_PATH"
echo "[dual/meituan/env.sh] TRAIN_FILE       = $TRAIN_FILE"
