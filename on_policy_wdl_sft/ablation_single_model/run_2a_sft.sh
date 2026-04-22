#!/usr/bin/env bash
# ==============================================================================
# Ablation 2A-SFT: single model, Qwen3-4B-Base-SFT-stage-1 init, wdl_sft_is, β=0.
# Compared with 1A (joint model2 uses the SAME SFT init) and with 2A-BASE.
#
# Question: with SFT init already in place, does wdl_sft_is alone match joint's
# +2.4pp lift? Also: disentangles "loss vs fusion" from "init quality".
# Plan reference: docs/joint_training/plans/active/ablation_single_model.md
# ==============================================================================
set -xeuo pipefail

export RUN_PREFIX=${RUN_PREFIX:-"WDL-SFT-Qwen3-4B-MATH-2A-SFT"}
export INIT_MODEL_PATH=${INIT_MODEL_PATH:-"/data-1/.cache/Qwen3-4B-Base-SFT-stage-1"}
export LOSS_MODE=${LOSS_MODE:-"wdl_sft_is"}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.0}
export LR=${LR:-5e-7}

WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WRAPPER_SCRIPT_DIR
# shellcheck disable=SC1091
source "${WRAPPER_SCRIPT_DIR}/_common_ablation.sh" "$@"
