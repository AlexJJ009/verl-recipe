#!/usr/bin/env bash
# Shared beta=0 Code A/C/D0 P60 contract. Paths remain host-overridable.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/qwen3_1p7b_stage123_resource_profile.sh"

export LOSS_MODE=wdl_sft
export WDL_SFT_BETA=0.0
export LR=1e-6
export LR_WARMUP_STEPS=0
export TOTAL_TRAINING_STEPS=60
export DATA_SHUFFLE=False
export TRAIN_PROMPT_BSZ=64
export ROLLOUT_N=8
export TRAIN_PROMPT_MINI_BSZ=512
export TEST_FREQ=5
export SAVE_FREQ=5
export VAL_N=3
export PROTECTED_CKPT_STEPS='[20,40,45,50,60]'
export PROTECTED_CKPT_STRIP_OPTIMIZER=False
export MAX_ACTOR_CKPTS_TO_KEEP=${MAX_ACTOR_CKPTS_TO_KEEP:-2}
export MAX_CRITIC_CKPTS_TO_KEEP=${MAX_CRITIC_CKPTS_TO_KEEP:-2}
export KEEP_BEST_CKPT=True
export BEST_CKPT_STRIP_OPTIMIZER=False
export CODE_ONLINE_HUMANEVAL_PLUS_VAL_FILE=${CODE_ONLINE_HUMANEVAL_PLUS_VAL_FILE:-/data-1/dataset/code/verl_rl/online_full_humaneval_plus/official_humaneval_plus_val.parquet}
export CODE_ONLINE_MBPP_PLUS_VAL_FILE=${CODE_ONLINE_MBPP_PLUS_VAL_FILE:-/data-1/dataset/code/verl_rl/online_full_mbpp_plus/official_mbpp_plus_val.parquet}
export CODE_ONLINE_LCB_V5_SUBSET_VAL_FILE=${CODE_ONLINE_LCB_V5_SUBSET_VAL_FILE:-/data-1/dataset/code/verl_rl/online_full_livecodebench_v5/official_livecodebench_val.parquet}
export CODE_VAL_FILES=${CODE_VAL_FILES:-"['$CODE_ONLINE_HUMANEVAL_PLUS_VAL_FILE','$CODE_ONLINE_MBPP_PLUS_VAL_FILE','$CODE_ONLINE_LCB_V5_SUBSET_VAL_FILE']"}
export TEST_FILES="$CODE_VAL_FILES"
export CUSTOM_REWARD_FN_PATH=${CUSTOM_REWARD_FN_PATH:-"${SCRIPT_DIR}/official_aligned_reward.py"}
export CUSTOM_REWARD_FN_NAME=compute_score_code_official_aligned
export CODE_EVAL_OFFICIAL_SITE=${CODE_EVAL_OFFICIAL_SITE:-/data-1/code_eval_envs/official_site}
export LCB_REPO_DIR=${LCB_REPO_DIR:-/data-1/code_eval_envs/LiveCodeBench}
export REPO_PYTHONPATH_ROOT=${REPO_PYTHONPATH_ROOT:-/workspace/verl}
export PYTHONPATH="${REPO_PYTHONPATH_ROOT}:${CODE_EVAL_OFFICIAL_SITE}:${LCB_REPO_DIR}:${PYTHONPATH:-}"
export WANDB_PROJECT=${WANDB_PROJECT:-OnPolicyWDLSFT-CodeTask}
export WANDB_MODE=${WANDB_MODE:-offline}
export BEST_CKPT_METRIC_KEY=${BEST_CKPT_METRIC_KEY:-val-core/model2/code3_macro/acc/mean@3}
export MIN_FREE_GB_FOR_CKPT=${MIN_FREE_GB_FOR_CKPT:-60}

if [ "${CODE_WDL_ACD0_GPU_PROBE_ADMITTED:-0}" = "1" ]; then
  if [ -z "${TMUX:-}" ]; then
    echo "ERROR: admitted A/D0/C GPU probe must run inside tmux" >&2
    exit 1
  fi
  export TOTAL_TRAINING_STEPS=1
  export VAL_BEFORE_TRAIN=False
  export TEST_FREQ=-1
  export SAVE_FREQ=-1
  export KEEP_BEST_CKPT=False
  export MAX_ACTOR_CKPTS_TO_KEEP=0
  export MAX_CRITIC_CKPTS_TO_KEEP=0
fi

: "${TRAIN_FILE:?TRAIN_FILE required}"
: "${BASE_CKPT_DIR:?BASE_CKPT_DIR required}"
