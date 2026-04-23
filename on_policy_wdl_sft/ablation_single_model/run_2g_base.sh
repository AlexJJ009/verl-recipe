#!/usr/bin/env bash
# ==============================================================================
# Ablation 2G-BASE: single model, Qwen3-4B-Base init, LOSS=vanilla
# (standard PPO-clip). Canonical GRPO baseline from the un-SFT'd base — the
# pure-RL cold-start reference.
#
# Paired with 2Z-BASE (MiniRL baseline, same init): isolates loss family
# (PPO-clip vs MiniRL IS + binary-mask) from the Base-init cold start.
# Expect a slow first ~50 steps: early rollouts will be mostly incorrect
# so the advantage signal will be sparse.
# ==============================================================================
set -xeuo pipefail

export RUN_PREFIX=${RUN_PREFIX:-"GRPO-Qwen3-4B-MATH-2G-BASE"}
export INIT_MODEL_PATH=${INIT_MODEL_PATH:-"/data-1/.cache/huggingface/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539"}
export LOSS_MODE=${LOSS_MODE:-"vanilla"}
export LR=${LR:-5e-7}

# Canonical GRPO knobs — differ from 2Z-BASE's MiniRL defaults:
export NORM_ADV_BY_STD_IN_GRPO=${NORM_ADV_BY_STD_IN_GRPO:-True}
export CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
export CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.2}

WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WRAPPER_SCRIPT_DIR
# shellcheck disable=SC1091
source "${WRAPPER_SCRIPT_DIR}/_common_ablation.sh" "$@"
