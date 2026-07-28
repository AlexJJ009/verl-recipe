#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${ROOT:-/data-1}
MODEL_ROOT=${MODEL_ROOT:-"${ROOT}/model_weights/rebuttal_rlvr/init"}
HF_MODEL_CACHE_ROOT=${HF_MODEL_CACHE_ROOT:-"${ROOT}/.cache/huggingface"}

export ARM="sft"
export INIT_CLASSIFIER="ordinary_sft"
export RUN_PREFIX=${RUN_PREFIX:-"REBUTTAL-RLVR-MATH-ORDINARY-SFT"}
export INIT_MODEL_PATH=${INIT_MODEL_PATH:-"${MODEL_ROOT}/REPLACE_WITH_ORDINARY_SFT_MATH"}
export BASE_PLACEHOLDER_MODEL_PATH=${BASE_PLACEHOLDER_MODEL_PATH:-"${HF_MODEL_CACHE_ROOT}/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539"}

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_math_rlvr.sh" "$@"
