#!/usr/bin/env bash
# ==============================================================================
# Meituan AFO worker entry point for dual_submodel_rollout runs.
#
# Required env:
#   EXPERIMENT   4a | 4a-dual | 4a-model2-group-adv-is
#
# This adapter resolves paths and chooses the matching run script. Algorithm
# defaults stay in the run_*.sh wrapper and _common_dual_rollout.sh.
# ==============================================================================

set -xeuo pipefail

: "${EXPERIMENT:?EXPERIMENT must be set (4a|4a-dual|4a-model2-group-adv-is)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
EXPERIMENT_LC="${EXPERIMENT,,}"

echo "[dual/meituan/jupyter.sh] SCRIPT_DIR = $SCRIPT_DIR"
echo "[dual/meituan/jupyter.sh] DUAL_DIR   = $DUAL_DIR"
echo "[dual/meituan/jupyter.sh] REPO_ROOT  = $REPO_ROOT"
echo "[dual/meituan/jupyter.sh] EXPERIMENT = $EXPERIMENT_LC"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

case "$EXPERIMENT_LC" in
    4a|4a-dual|4a-model2-group-adv-is)
        RUN_SCRIPT="${DUAL_DIR}/run_4a_model2_group_adv_is.sh"
        ;;
    *)
        echo "[dual/meituan/jupyter.sh] ERROR: unsupported EXPERIMENT='$EXPERIMENT'. Use 4a or 4a-dual." >&2
        exit 1
        ;;
esac

if [ ! -f "$RUN_SCRIPT" ]; then
    echo "[dual/meituan/jupyter.sh] ERROR: run script not found: $RUN_SCRIPT" >&2
    exit 1
fi
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "[dual/meituan/jupyter.sh] ERROR: BASE_MODEL_PATH not found: $BASE_MODEL_PATH" >&2
    exit 1
fi
if [ ! -d "$MODEL2_PATH" ]; then
    echo "[dual/meituan/jupyter.sh] ERROR: MODEL2_PATH not found: $MODEL2_PATH" >&2
    exit 1
fi
if [ ! -f "$TRAIN_FILE" ]; then
    echo "[dual/meituan/jupyter.sh] ERROR: TRAIN_FILE not found: $TRAIN_FILE" >&2
    echo "[dual/meituan/jupyter.sh] HINT: upload MATH train_rl_format.parquet to \$LGX/verl-exp/data/math/" >&2
    exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
    echo "[dual/meituan/jupyter.sh] WARN: MODEL_PATH not found; _common_dual_rollout.sh will prepare it at $MODEL_PATH" >&2
fi

cd "$REPO_ROOT"
echo "[dual/meituan/jupyter.sh] cwd = $(pwd); launching $RUN_SCRIPT"
exec bash "$RUN_SCRIPT"
