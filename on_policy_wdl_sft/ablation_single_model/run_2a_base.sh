#!/usr/bin/env bash
# ==============================================================================
# Ablation 2A-BASE: single model, Qwen3-4B-Base init, wdl_sft_is loss, β=0.
# Compared with 1A (joint model, SFT init for model2, same β/lr).
#
# Question: does the wdl_sft_is loss alone (no logit fusion) lift a Base model?
# Plan reference: docs/joint_training/plans/active/ablation_single_model.md
# ==============================================================================
set -xeuo pipefail

export RUN_PREFIX=${RUN_PREFIX:-"WDL-SFT-Qwen3-4B-MATH-2A-BASE-LABELFIX"}
export INIT_MODEL_PATH=${INIT_MODEL_PATH:-"/data-1/.cache/huggingface/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539"}
export LOSS_MODE=${LOSS_MODE:-"wdl_sft_is"}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.0}
export LR=${LR:-5e-7}

WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WRAPPER_SCRIPT_DIR
# shellcheck disable=SC1091
source "${WRAPPER_SCRIPT_DIR}/_common_ablation.sh" "$@"
