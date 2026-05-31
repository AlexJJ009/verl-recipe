#!/usr/bin/env bash
# Meituan AFO path overrides for staged v1 On-Policy SFT -> WDL-SFT.

LGX=${LGX:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yangfengkai02/lgx}

export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_tmp}
export TMPDIR=${TMPDIR:-/tmp/verl_tmp}
export VLLM_CONFIG_ROOT=${VLLM_CONFIG_ROOT:-/tmp/vllm_config}
export VERL_ZMQ_IPC_DIR=${VERL_ZMQ_IPC_DIR:-$TMPDIR}

export HF_HOME=${HF_HOME:-$LGX/verl-exp/hf_cache}
export BASE_CKPT_DIR=${BASE_CKPT_DIR:-$LGX/verl-exp/checkpoints}
export WANDB_DIR=${WANDB_DIR:-$LGX/verl-exp/wandb_runs}
export LOG_DIR=${LOG_DIR:-$LGX/verl-exp/logs/staged_v1}
export WANDB_PROJECT=${WANDB_PROJECT:-OnPolicySFT-Then-WDLSFT-StagedV1}
export WANDB_MODE=${WANDB_MODE:-offline}

export MIN_FREE_GB_FOR_CKPT=${MIN_FREE_GB_FOR_CKPT:-100}
export MAX_ACTOR_CKPTS_TO_KEEP=${MAX_ACTOR_CKPTS_TO_KEEP:-1}
export MAX_CRITIC_CKPTS_TO_KEEP=${MAX_CRITIC_CKPTS_TO_KEEP:-1}
export KEEP_BEST_CKPT=${KEEP_BEST_CKPT:-True}
export BEST_CKPT_METRIC_KEY=${BEST_CKPT_METRIC_KEY:-val-core/HuggingFaceH4/MATH-500/acc/mean@3}
export BEST_CKPT_METRIC_MODE=${BEST_CKPT_METRIC_MODE:-max}
export BEST_CKPT_STRIP_OPTIMIZER=${BEST_CKPT_STRIP_OPTIMIZER:-True}

export TEST_FREQ=${TEST_FREQ:-5}
export SAVE_FREQ=${SAVE_FREQ:-5}
export VAL_N=${VAL_N:-3}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export DATA_SEED=${DATA_SEED:-20260528}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-150}

export TRAIN_FILE=${TRAIN_FILE:-$LGX/verl-exp/data/EnsembleLLM-data-processed/staged_v1/train_rl_format_boxed_prompt.parquet}
export TEST_FILES=${TEST_FILES:-"['$LGX/verl-exp/data/MATH-500/math500-test_with_system_prompt.parquet','$LGX/verl-exp/data/AIME-2025/aime-2025_with_system_prompt.parquet']"}

export MEITUAN_BASE_MODEL_PATH=${MEITUAN_BASE_MODEL_PATH:-$LGX/huggingface.co/Qwen/Qwen3-4B-Base}
export BASE_MODEL_PATH=${BASE_MODEL_PATH:-$MEITUAN_BASE_MODEL_PATH}

export STAGE1_RUN_PREFIX=${STAGE1_RUN_PREFIX:-ONPOLICY-SFT-Qwen3-4B-MATH-S1-BOXED-BETA0-V1}
export STAGE1_MERGED_MODEL_ROOT=${STAGE1_MERGED_MODEL_ROOT:-$LGX/verl-exp/model_weights/staged_v1}

mkdir -p \
    "$HF_HOME" \
    "$BASE_CKPT_DIR" \
    "$WANDB_DIR" \
    "$LOG_DIR" \
    "$RAY_TMPDIR" \
    "$TMPDIR" \
    "$VLLM_CONFIG_ROOT" \
    "$VERL_ZMQ_IPC_DIR" \
    "$STAGE1_MERGED_MODEL_ROOT"

echo "[staged_v1/meituan/env.sh] LGX                       = $LGX"
echo "[staged_v1/meituan/env.sh] BASE_CKPT_DIR             = $BASE_CKPT_DIR"
echo "[staged_v1/meituan/env.sh] LOG_DIR                   = $LOG_DIR"
echo "[staged_v1/meituan/env.sh] TRAIN_FILE                = $TRAIN_FILE"
echo "[staged_v1/meituan/env.sh] BASE_MODEL_PATH           = $BASE_MODEL_PATH"
echo "[staged_v1/meituan/env.sh] STAGE1_RUN_PREFIX         = $STAGE1_RUN_PREFIX"
echo "[staged_v1/meituan/env.sh] STAGE1_MERGED_MODEL_ROOT  = $STAGE1_MERGED_MODEL_ROOT"
