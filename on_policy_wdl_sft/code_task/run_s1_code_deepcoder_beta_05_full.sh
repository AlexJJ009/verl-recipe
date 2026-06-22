#!/usr/bin/env bash
# DeepCoder Stage1 code-task On-Policy WDL-SFT, beta=0.5, formal full 150-step run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_PREFIX=${RUN_PREFIX:-ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA05-V1-FULL}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.5}
export CODE_TRAIN_FILE=${CODE_TRAIN_FILE:-/data-1/dataset/code/verl_rl/deepcoder_preview_train_prompt1024_rl_format.parquet}
export TRAIN_FILE=${TRAIN_FILE:-$CODE_TRAIN_FILE}
# This is a formal curve run, but the current local disk cannot safely duplicate
# the dense handoff retention used by beta=0.0/0.1. Keep latest plus best only.
export MAX_ACTOR_CKPTS_TO_KEEP=${MAX_ACTOR_CKPTS_TO_KEEP:-1}
export MAX_CRITIC_CKPTS_TO_KEEP=${MAX_CRITIC_CKPTS_TO_KEEP:-1}
export KEEP_BEST_CKPT=${KEEP_BEST_CKPT:-True}
export BEST_CKPT_STRIP_OPTIMIZER=${BEST_CKPT_STRIP_OPTIMIZER:-True}
export PROTECTED_CKPT_STEPS=${PROTECTED_CKPT_STEPS:-[]}
export PROTECTED_CKPT_STRIP_OPTIMIZER=${PROTECTED_CKPT_STRIP_OPTIMIZER:-True}
export MIN_FREE_GB_FOR_CKPT=${MIN_FREE_GB_FOR_CKPT:-220}
export CODE_REWARD_TIMEOUT=${CODE_REWARD_TIMEOUT:-30}

exec bash "${SCRIPT_DIR}/run_s1_code_onpolicy_sft_beta_0.sh" "$@"
