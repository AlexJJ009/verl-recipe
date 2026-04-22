#!/usr/bin/env bash
# ==============================================================================
# Ablation 2C-BASE: single model, Qwen3-4B-Base init, wdl_sft_is, β=0, lr=1e-6.
# Compared with 1C (same lr, joint). Tests the higher-lr regime on a single
# model — is lr=1e-6 inherently unstable, or only unstable when combined with
# fusion dynamics?
# ==============================================================================
set -xeuo pipefail

export RUN_PREFIX=${RUN_PREFIX:-"WDL-SFT-Qwen3-4B-MATH-2C-BASE"}
export INIT_MODEL_PATH=${INIT_MODEL_PATH:-"/data-1/.cache/huggingface/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539"}
export LOSS_MODE=${LOSS_MODE:-"wdl_sft_is"}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.0}
export LR=${LR:-1e-6}

WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WRAPPER_SCRIPT_DIR
# shellcheck disable=SC1091
source "${WRAPPER_SCRIPT_DIR}/_common_ablation.sh" "$@"
