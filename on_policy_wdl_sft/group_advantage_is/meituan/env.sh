#!/usr/bin/env bash
# Meituan AFO path overrides for WDL group-advantage IS.

LGX=${LGX:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yangfengkai02/lgx}

export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_tmp}
export TMPDIR=${TMPDIR:-/tmp/verl_tmp}
export VLLM_CONFIG_ROOT=${VLLM_CONFIG_ROOT:-/tmp/vllm_config}
export VERL_ZMQ_IPC_DIR=${VERL_ZMQ_IPC_DIR:-$TMPDIR}

export DATA_ROOT=${DATA_ROOT:-$LGX/verl-exp}
export HF_HOME=${HF_HOME:-$LGX/verl-exp/hf_cache}
export BASE_CKPT_DIR=${BASE_CKPT_DIR:-$LGX/verl-exp/checkpoints}
export WANDB_DIR=${WANDB_DIR:-$LGX/verl-exp/wandb_runs}
export LOG_DIR=${LOG_DIR:-$LGX/verl-exp/logs/group_advantage_is}
export VALIDATION_OUTPUT_DIR=${VALIDATION_OUTPUT_DIR:-$LOG_DIR/validation}
export WANDB_MODE=${WANDB_MODE:-offline}
export MAX_ACTOR_CKPTS_TO_KEEP=${MAX_ACTOR_CKPTS_TO_KEEP:-1}
export KEEP_BEST_CKPT=${KEEP_BEST_CKPT:-True}
export BEST_CKPT_METRIC_KEY=${BEST_CKPT_METRIC_KEY:-val-core/HuggingFaceH4/MATH-500/acc/mean@3}
export BEST_CKPT_METRIC_MODE=${BEST_CKPT_METRIC_MODE:-max}
export BEST_CKPT_STRIP_OPTIMIZER=${BEST_CKPT_STRIP_OPTIMIZER:-True}
export MIN_FREE_GB_FOR_CKPT=${MIN_FREE_GB_FOR_CKPT:-60}

export TRAIN_FILE=${TRAIN_FILE:-$LGX/verl-exp/data/EnsembleLLM-data-processed/train_rl_format.parquet}
export TEST_FILES=${TEST_FILES:-"['$LGX/verl-exp/data/MATH-500/math500-test_with_system_prompt.parquet','$LGX/verl-exp/data/AIME-2025/aime-2025_with_system_prompt.parquet']"}
export CUSTOM_REWARD_FN_PATH=${CUSTOM_REWARD_FN_PATH:-$REPO_ROOT/recipe/on_policy_wdl_sft/custom_reward_function_latex_verify.py}

export BASE_MODEL_PATH=${BASE_MODEL_PATH:-$LGX/huggingface.co/Qwen/Qwen3-4B-Base}
export MODEL2_PATH=${MODEL2_PATH:-$LGX/huggingface.co/Qwen/Qwen3-4B-Base-SFT-stage-1/flat}
export MODEL_PATH=${MODEL_PATH:-$LGX/huggingface.co/Qwen/QwenJoint-4B-WDL-SFT-stage-1}

mkdir -p "$HF_HOME" "$BASE_CKPT_DIR" "$WANDB_DIR" "$LOG_DIR" "$VALIDATION_OUTPUT_DIR"

echo "[group_adv_is/env.sh] LGX                   = $LGX"
echo "[group_adv_is/env.sh] HF_HOME               = $HF_HOME"
echo "[group_adv_is/env.sh] BASE_CKPT_DIR         = $BASE_CKPT_DIR"
echo "[group_adv_is/env.sh] LOG_DIR               = $LOG_DIR"
echo "[group_adv_is/env.sh] TRAIN_FILE            = $TRAIN_FILE"
echo "[group_adv_is/env.sh] BASE_MODEL_PATH       = $BASE_MODEL_PATH"
echo "[group_adv_is/env.sh] MODEL2_PATH           = $MODEL2_PATH"
echo "[group_adv_is/env.sh] MODEL_PATH            = $MODEL_PATH"
