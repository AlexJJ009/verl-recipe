#!/usr/bin/env bash
# ==============================================================================
# Meituan AFO worker entry point for joint On-Policy WDL-SFT 1X reruns.
#
# Required env:
#   EXPERIMENT   1a | 1b | 1c  (also accepts 1a-joint / 1b-joint / 1c-joint)
#
# This adapter only resolves paths and chooses the matching run script. All
# hyperparameters stay in the existing run_on_policy_wdl_sft_qwen3_4b_math_1*.sh
# scripts.
# ==============================================================================

set -xeuo pipefail

: "${EXPERIMENT:?EXPERIMENT must be set (1a|1b|1c)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EXPERIMENT_LC="${EXPERIMENT,,}"

echo "[joint/meituan/jupyter.sh] SCRIPT_DIR = $SCRIPT_DIR"
echo "[joint/meituan/jupyter.sh] RECIPE_DIR = $RECIPE_DIR"
echo "[joint/meituan/jupyter.sh] REPO_ROOT  = $REPO_ROOT"
echo "[joint/meituan/jupyter.sh] EXPERIMENT = $EXPERIMENT_LC"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

case "$EXPERIMENT_LC" in
    1a|1a-joint) RUN_SCRIPT="${RECIPE_DIR}/run_on_policy_wdl_sft_qwen3_4b_math_1a.sh" ;;
    1b|1b-joint) RUN_SCRIPT="${RECIPE_DIR}/run_on_policy_wdl_sft_qwen3_4b_math_1b.sh" ;;
    1c|1c-joint) RUN_SCRIPT="${RECIPE_DIR}/run_on_policy_wdl_sft_qwen3_4b_math_1c.sh" ;;
    *)
        echo "[joint/meituan/jupyter.sh] ERROR: unsupported EXPERIMENT='$EXPERIMENT'. Use 1a, 1b, or 1c." >&2
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

cd "$REPO_ROOT"
echo "[joint/meituan/jupyter.sh] cwd = $(pwd); launching $RUN_SCRIPT"
exec bash "$RUN_SCRIPT"
