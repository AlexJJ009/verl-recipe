#!/usr/bin/env bash
# ==============================================================================
# Ablation 2Z-BASE: single model, Qwen3-4B-Base init, LOSS=minirl (plain
# clipped-PG with IS correction). The "pure RL baseline" — gives us a ceiling/
# floor reference so we can decompose the total effect into
#   (loss effect)   = wdl_sft_is − minirl on the same single-model setup
#   (fusion effect) = joint wdl_sft_is − single wdl_sft_is
#
# We already have historical MiniRL reference data; this script is kept for
# reproducibility on the Meituan platform (all hyperparams aligned with 2A).
# ==============================================================================
set -xeuo pipefail

export RUN_PREFIX=${RUN_PREFIX:-"MINIRL-Qwen3-4B-MATH-2Z-BASE"}
export INIT_MODEL_PATH=${INIT_MODEL_PATH:-"/data-1/.cache/huggingface/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539"}
export LOSS_MODE=${LOSS_MODE:-"minirl"}
export LR=${LR:-5e-7}
# WDL_SFT_BETA is ignored by minirl loss — omit.

WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WRAPPER_SCRIPT_DIR
# shellcheck disable=SC1091
source "${WRAPPER_SCRIPT_DIR}/_common_ablation.sh" "$@"
