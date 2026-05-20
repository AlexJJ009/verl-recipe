#!/usr/bin/env bash
# Meituan AFO worker entry point for WDL group-advantage IS.
set -xeuo pipefail

: "${EXPERIMENT:?EXPERIMENT must be set, for example 1a-group-adv-is}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAMILY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
export REPO_ROOT

echo "[group_adv_is/meituan] SCRIPT_DIR = $SCRIPT_DIR"
echo "[group_adv_is/meituan] FAMILY_DIR = $FAMILY_DIR"
echo "[group_adv_is/meituan] REPO_ROOT  = $REPO_ROOT"
echo "[group_adv_is/meituan] EXPERIMENT = $EXPERIMENT"
echo "[group_adv_is/meituan] SMOKE      = ${SMOKE:-0}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

if [ "${SMOKE:-0}" = "1" ]; then
    export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-10}
    export TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-2}
    export TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ:-1}
    export TEST_FREQ=${TEST_FREQ:--1}
    export SAVE_FREQ=${SAVE_FREQ:-5}
    export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
    export ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-1}
    echo "[group_adv_is/meituan] SMOKE mode propagated"
fi

[ -d "$BASE_MODEL_PATH" ] || { echo "ERROR: BASE_MODEL_PATH not found: $BASE_MODEL_PATH" >&2; exit 1; }
[ -d "$MODEL2_PATH" ] || { echo "ERROR: MODEL2_PATH not found: $MODEL2_PATH" >&2; exit 1; }
[ -f "$TRAIN_FILE" ] || { echo "ERROR: TRAIN_FILE not found: $TRAIN_FILE" >&2; exit 1; }

RUN_SCRIPT="${FAMILY_DIR}/run_${EXPERIMENT//-/_}.sh"
if [ ! -f "$RUN_SCRIPT" ]; then
    echo "ERROR: run script not found: $RUN_SCRIPT" >&2
    exit 1
fi

cd "$REPO_ROOT"
exec bash "$RUN_SCRIPT"
