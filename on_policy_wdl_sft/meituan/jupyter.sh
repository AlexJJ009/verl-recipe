#!/usr/bin/env bash
# ==============================================================================
# Meituan AFO worker entry point for joint On-Policy WDL-SFT-IS 1X runs.
#
# Required env (set via run.hope `afo.app.env.X=Y`):
#   EXPERIMENT   one of: 1a | 1b | 1c | 1a-joint | 1b-joint | 1c-joint
#
# Optional env:
#   LGX, BASE_MODEL_PATH, MODEL2_PATH, MODEL_PATH, TRAIN_FILE, TEST_FILES,
#   BASE_CKPT_DIR, WANDB_DIR, LOG_DIR, VAL_N, and all training knobs accepted by
#   run_on_policy_wdl_sft_qwen3_4b_math_1{a,b,c}.sh.
#
# Flow:
#   1. Source meituan/env.sh for default-local-overridable path mapping.
#   2. Normalize EXPERIMENT aliases.
#   3. Validate model/data prerequisites.
#   4. Invoke the matching portable local run script.
# ==============================================================================

set -xeuo pipefail

: "${EXPERIMENT:?EXPERIMENT must be set (one of 1a|1b|1c|1a-joint|1b-joint|1c-joint)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# SCRIPT_DIR = <repo>/recipe/on_policy_wdl_sft/meituan
# repo root is 3 levels up.
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "[joint/meituan/jupyter.sh] SCRIPT_DIR = $SCRIPT_DIR"
echo "[joint/meituan/jupyter.sh] RECIPE_DIR = $RECIPE_DIR"
echo "[joint/meituan/jupyter.sh] REPO_ROOT  = $REPO_ROOT"
echo "[joint/meituan/jupyter.sh] EXPERIMENT = $EXPERIMENT"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

EXPERIMENT_LC="${EXPERIMENT,,}"
case "$EXPERIMENT_LC" in
    1a|1a-joint) RUN_SCRIPT="${RECIPE_DIR}/run_on_policy_wdl_sft_qwen3_4b_math_1a.sh" ;;
    1b|1b-joint) RUN_SCRIPT="${RECIPE_DIR}/run_on_policy_wdl_sft_qwen3_4b_math_1b.sh" ;;
    1c|1c-joint) RUN_SCRIPT="${RECIPE_DIR}/run_on_policy_wdl_sft_qwen3_4b_math_1c.sh" ;;
    *)
        echo "[joint/meituan/jupyter.sh] ERROR: unsupported EXPERIMENT='$EXPERIMENT'" >&2
        exit 1
        ;;
esac

if [ ! -f "$RUN_SCRIPT" ]; then
    echo "[joint/meituan/jupyter.sh] ERROR: run script not found: $RUN_SCRIPT" >&2
    exit 1
fi
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "[joint/meituan/jupyter.sh] ERROR: BASE_MODEL_PATH not found: $BASE_MODEL_PATH" >&2
    exit 1
fi
if [ ! -d "$MODEL2_PATH" ]; then
    echo "[joint/meituan/jupyter.sh] ERROR: MODEL2_PATH not found: $MODEL2_PATH" >&2
    exit 1
fi
if [ ! -f "$TRAIN_FILE" ]; then
    echo "[joint/meituan/jupyter.sh] ERROR: TRAIN_FILE not found: $TRAIN_FILE" >&2
    exit 1
fi

echo "[joint/meituan/jupyter.sh] resolved run script: $RUN_SCRIPT"
cd "$REPO_ROOT"
echo "[joint/meituan/jupyter.sh] cwd = $(pwd); launching training..."
exec bash "$RUN_SCRIPT"
