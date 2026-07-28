#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/model_paths.env"

export ARM="wdl"
export INIT_CLASSIFIER="offline_wdl_sft"
export RUN_PREFIX=${RUN_PREFIX:-"REBUTTAL-RLVR-MATH-OFFLINE-WDL-SFT"}
export INIT_MODEL_PATH=${INIT_MODEL_PATH:-"$WDL_4B_MODEL_PATH"}
export BASE_PLACEHOLDER_MODEL_PATH=${BASE_PLACEHOLDER_MODEL_PATH:-"${HF_MODEL_CACHE_ROOT}/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539"}

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_math_rlvr.sh" "$@"
