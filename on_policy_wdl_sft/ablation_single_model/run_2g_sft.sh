#!/usr/bin/env bash
# ==============================================================================
# Ablation 2G-SFT: single model, Qwen3-4B-Base-SFT-stage-1 init, LOSS=vanilla
# (standard PPO-clip). Canonical GRPO baseline: grpo adv + std-normalized +
# symmetric clip — matches DeepSeek/TRL GRPO literature.
#
# Paired with 2Z-SFT (MiniRL baseline, same init): isolates loss family
# (PPO-clip vs MiniRL IS + binary-mask) with everything else held fixed.
# ==============================================================================
set -xeuo pipefail

export RUN_PREFIX=${RUN_PREFIX:-"GRPO-Qwen3-4B-MATH-2G-SFT"}
export INIT_MODEL_PATH=${INIT_MODEL_PATH:-"/data-1/.cache/Qwen3-4B-Base-SFT-stage-1"}
export LOSS_MODE=${LOSS_MODE:-"vanilla"}
export LR=${LR:-5e-7}

# Canonical GRPO knobs — differ from 2Z-SFT's MiniRL defaults:
export NORM_ADV_BY_STD_IN_GRPO=${NORM_ADV_BY_STD_IN_GRPO:-True}
export CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
export CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.2}

WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WRAPPER_SCRIPT_DIR
# shellcheck disable=SC1091
source "${WRAPPER_SCRIPT_DIR}/_common_ablation.sh" "$@"
