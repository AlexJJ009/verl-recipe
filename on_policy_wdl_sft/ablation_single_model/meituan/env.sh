#!/usr/bin/env bash
# ==============================================================================
# Meituan MLP path overrides for ablation_single_model.
#
# Source this BEFORE invoking any run_2X_*.sh on AFO worker pods. Sets all
# overridable paths in _common_ablation.sh to their dolphinfs equivalents,
# so the SAME run_2X_*.sh scripts work on both the local dev box and Meituan.
#
# This file is idempotent and safe to source multiple times.
# ==============================================================================

# Dolphinfs anchor (persistent, visible to BOTH Codelab and AFO workers).
# Override via LGX env if the account/path changes.
LGX=${LGX:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yangfengkai02/lgx}

# --- High-churn temp dirs: keep on container-local disk, NOT dolphinfs --------
# Ray tmp generates tens of thousands of small files per step; dolphinfs
# network latency would murder throughput. /tmp is ephemeral but that is fine —
# ray tmp is supposed to be ephemeral. Dolphinfs is only for *persistent* state
# (checkpoints, logs, wandb).
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_tmp}
export TMPDIR=${TMPDIR:-/tmp/verl_tmp}
export VLLM_CONFIG_ROOT=${VLLM_CONFIG_ROOT:-/tmp/vllm_config}
export VERL_ZMQ_IPC_DIR=${VERL_ZMQ_IPC_DIR:-$TMPDIR}

# --- HF_HOME: tokenizer/config auto-resolution cache --------------------------
# Model weights are passed explicitly via INIT_MODEL_PATH, so HF_HOME is only
# used as a working cache — put on dolphinfs to persist across pods.
export HF_HOME=${HF_HOME:-$LGX/verl-exp/hf_cache}

# --- Persistent state: checkpoints, wandb, logs → dolphinfs -------------------
export BASE_CKPT_DIR=${BASE_CKPT_DIR:-$LGX/verl-exp/checkpoints}
export WANDB_DIR=${WANDB_DIR:-$LGX/verl-exp/wandb_runs}
export LOG_DIR=${LOG_DIR:-$LGX/verl-exp/logs/ablation_single_model}
# wandb defaults to offline on MLP (no outbound internet in worker pods).
export WANDB_MODE=${WANDB_MODE:-offline}
# Keep latest full checkpoint plus best model-only checkpoint. This keeps the
# nine-run Meituan batch below the 1T storage budget.
export MAX_ACTOR_CKPTS_TO_KEEP=${MAX_ACTOR_CKPTS_TO_KEEP:-1}
export KEEP_BEST_CKPT=${KEEP_BEST_CKPT:-True}
export BEST_CKPT_METRIC_KEY=${BEST_CKPT_METRIC_KEY:-val-core/HuggingFaceH4/MATH-500/acc/mean@1}
export BEST_CKPT_METRIC_MODE=${BEST_CKPT_METRIC_MODE:-max}
export BEST_CKPT_STRIP_OPTIMIZER=${BEST_CKPT_STRIP_OPTIMIZER:-True}
export MIN_FREE_GB_FOR_CKPT=${MIN_FREE_GB_FOR_CKPT:-60}

# --- Datasets (user must upload these to the below paths) ---------------------
# Uploaded as of 2026-04-22 to $LGX/verl-exp/data/ (EnsembleLLM-data-processed,
# MATH-500, AIME-2025, AMC23, AQUA, SVAMP). The run script will fail fast with
# a clear ENOENT if these are missing. Upload command template:
#   rsync -avP /data-1/dataset/EnsembleLLM-data-processed/ \
#     $LGX/verl-exp/data/EnsembleLLM-data-processed/
export TRAIN_FILE=${TRAIN_FILE:-$LGX/verl-exp/data/EnsembleLLM-data-processed/train_rl_format.parquet}
export MATH_TRAIN_FILE=${MATH_TRAIN_FILE:-$LGX/verl-exp/data/math/train_rl_format.parquet}
export TEST_FILES=${TEST_FILES:-"['$LGX/verl-exp/data/MATH-500/math500-test_with_system_prompt.parquet','$LGX/verl-exp/data/AIME-2025/aime-2025_with_system_prompt.parquet']"}

# --- Init-model paths — consumed by jupyter.sh based on EXPERIMENT suffix -----
# Both models use a flat directory layout (real files, no HF cache symlinks).
# Rationale: dolphinfs disallows hardlinks AND symlinks, so the HF cache layout
# (models--<org>--<name>/snapshots/<hash>/<file> -> ../../blobs/<sha>) cannot be
# materialized. The SFT model's blobs were `mv`'d into a flat dir on 2026-04-23
# after a Mac-zip upload lost all symlinks; verified via sha256 against the
# original LFS oids. Future uploads should go straight to flat layout — see
# docs/joint_training/guides/meituan_platform.md.
export MEITUAN_BASE_MODEL_PATH=${MEITUAN_BASE_MODEL_PATH:-$LGX/huggingface.co/Qwen/Qwen3-4B-Base}
export MEITUAN_SFT_MODEL_PATH=${MEITUAN_SFT_MODEL_PATH:-$LGX/huggingface.co/Qwen/Qwen3-4B-Base-SFT-stage-1/flat}

# --- Pre-create dolphinfs dirs so training script doesn't fail on first write -
mkdir -p "$HF_HOME" "$BASE_CKPT_DIR" "$WANDB_DIR" "$LOG_DIR"

echo "[meituan/env.sh] LGX                     = $LGX"
echo "[meituan/env.sh] HF_HOME                 = $HF_HOME"
echo "[meituan/env.sh] RAY_TMPDIR (local)      = $RAY_TMPDIR"
echo "[meituan/env.sh] BASE_CKPT_DIR           = $BASE_CKPT_DIR"
echo "[meituan/env.sh] WANDB_DIR               = $WANDB_DIR"
echo "[meituan/env.sh] LOG_DIR                 = $LOG_DIR"
echo "[meituan/env.sh] TRAIN_FILE              = $TRAIN_FILE"
echo "[meituan/env.sh] MATH_TRAIN_FILE         = $MATH_TRAIN_FILE"
echo "[meituan/env.sh] MEITUAN_BASE_MODEL_PATH = $MEITUAN_BASE_MODEL_PATH"
echo "[meituan/env.sh] MEITUAN_SFT_MODEL_PATH  = $MEITUAN_SFT_MODEL_PATH"
