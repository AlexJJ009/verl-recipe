#!/usr/bin/env bash
# ==============================================================================
# Meituan AFO worker entry point for single-model ablation runs.
#
# Required env (set via run.hope `afo.app.env.X=Y`):
#   EXPERIMENT   one of: 2a-base | 2a-sft | 2b-base | 2b-sft |
#                        2c-base | 2c-sft | 2z-base | 2z-sft |
#                        2g-base | 2g-sft
#
# Optional env (all overridable via run.hope):
#   LGX                      dolphinfs anchor (default: hadoop-ai-search/yangfengkai02/lgx)
#   MEITUAN_BASE_MODEL_PATH  override init path for *-base runs
#   MEITUAN_SFT_MODEL_PATH   override init path for *-sft runs
#   TRAIN_FILE, TEST_FILES   dataset parquet paths
#   BASE_CKPT_DIR, WANDB_DIR persistent output roots
#
# Flow:
#   1. Source meituan/env.sh (path overrides)
#   2. Pick INIT_MODEL_PATH based on EXPERIMENT's *-base or *-sft suffix
#   3. Validate prerequisites (model + dataset exist)
#   4. Invoke the matching run_2X_*.sh from ablation_single_model/
# ==============================================================================

set -xeuo pipefail

: "${EXPERIMENT:?EXPERIMENT must be set (one of 2a-base|2a-sft|2b-base|2b-sft|2c-base|2c-sft|2z-base|2z-sft|2g-base|2g-sft)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# SCRIPT_DIR = <repo>/recipe/on_policy_wdl_sft/ablation_single_model/meituan
# → repo root is 4 levels up
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

echo "[meituan/jupyter.sh] SCRIPT_DIR    = $SCRIPT_DIR"
echo "[meituan/jupyter.sh] ABLATION_DIR  = $ABLATION_DIR"
echo "[meituan/jupyter.sh] REPO_ROOT     = $REPO_ROOT"
echo "[meituan/jupyter.sh] EXPERIMENT    = $EXPERIMENT"

# ---- 1. Source path overrides ------------------------------------------------
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

# ---- 2. Pick init model based on EXPERIMENT suffix ---------------------------
case "$EXPERIMENT" in
    *-base) export INIT_MODEL_PATH="$MEITUAN_BASE_MODEL_PATH" ;;
    *-sft)  export INIT_MODEL_PATH="$MEITUAN_SFT_MODEL_PATH"  ;;
    *)
        echo "[meituan/jupyter.sh] ERROR: EXPERIMENT='$EXPERIMENT' must end in -base or -sft" >&2
        exit 1
        ;;
esac

# ---- 3. Validate prerequisites -----------------------------------------------
if [ ! -d "$INIT_MODEL_PATH" ]; then
    echo "[meituan/jupyter.sh] ERROR: init model not found at $INIT_MODEL_PATH" >&2
    if [[ "$EXPERIMENT" == *-sft ]]; then
        echo "[meituan/jupyter.sh] HINT: -sft experiments need Qwen3-4B-Base-SFT-stage-1 at $MEITUAN_SFT_MODEL_PATH (flat dir)" >&2
        echo "[meituan/jupyter.sh] HINT: see docs/joint_training/guides/meituan_platform.md for upload workflow" >&2
    fi
    exit 1
fi
if [ ! -f "$TRAIN_FILE" ]; then
    echo "[meituan/jupyter.sh] ERROR: TRAIN_FILE not found at $TRAIN_FILE" >&2
    echo "[meituan/jupyter.sh] HINT: upload EnsembleLLM train_rl_format.parquet to \$LGX/verl-exp/data/EnsembleLLM-data-processed/" >&2
    exit 1
fi

# ---- 4. Resolve run script ---------------------------------------------------
# EXPERIMENT "2z-base" → run_2z_base.sh
RUN_SCRIPT="${ABLATION_DIR}/run_${EXPERIMENT//-/_}.sh"
if [ ! -f "$RUN_SCRIPT" ]; then
    echo "[meituan/jupyter.sh] ERROR: run script not found: $RUN_SCRIPT" >&2
    exit 1
fi
echo "[meituan/jupyter.sh] resolved run script: $RUN_SCRIPT"

# ---- 5. Launch ---------------------------------------------------------------
# Enter repo root — some verl modules auto-detect paths relative to cwd.
cd "$REPO_ROOT"
echo "[meituan/jupyter.sh] cwd = $(pwd); launching training…"
exec bash "$RUN_SCRIPT"
