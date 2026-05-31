#!/usr/bin/env bash
# Meituan AFO worker entry point for staged v1 runs.

set -xeuo pipefail

: "${EXPERIMENT:?EXPERIMENT must be set (s1-beta-* or s1-base-sft)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGED_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

echo "[staged_v1/meituan/jupyter.sh] SCRIPT_DIR = $SCRIPT_DIR"
echo "[staged_v1/meituan/jupyter.sh] STAGED_DIR = $STAGED_DIR"
echo "[staged_v1/meituan/jupyter.sh] REPO_ROOT  = $REPO_ROOT"
echo "[staged_v1/meituan/jupyter.sh] EXPERIMENT = $EXPERIMENT"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

EXPERIMENT_LC="${EXPERIMENT,,}"
case "$EXPERIMENT_LC" in
    s1-base-sft|stage1-base-sft)
        export INIT_MODEL_PATH="$MEITUAN_BASE_MODEL_PATH"
        RUN_SCRIPT="${STAGED_DIR}/run_s1_base_sft.sh"
        ;;
    s1-beta-0)   export INIT_MODEL_PATH="$MEITUAN_BASE_MODEL_PATH"; RUN_SCRIPT="${STAGED_DIR}/run_s1_beta_0.sh" ;;
    s1-beta-01)  export INIT_MODEL_PATH="$MEITUAN_BASE_MODEL_PATH"; RUN_SCRIPT="${STAGED_DIR}/run_s1_beta_01.sh" ;;
    s1-beta-02)  export INIT_MODEL_PATH="$MEITUAN_BASE_MODEL_PATH"; RUN_SCRIPT="${STAGED_DIR}/run_s1_beta_02.sh" ;;
    s1-beta-03)  export INIT_MODEL_PATH="$MEITUAN_BASE_MODEL_PATH"; RUN_SCRIPT="${STAGED_DIR}/run_s1_beta_03.sh" ;;
    s1-beta-04)  export INIT_MODEL_PATH="$MEITUAN_BASE_MODEL_PATH"; RUN_SCRIPT="${STAGED_DIR}/run_s1_beta_04.sh" ;;
    s1-beta-05)  export INIT_MODEL_PATH="$MEITUAN_BASE_MODEL_PATH"; RUN_SCRIPT="${STAGED_DIR}/run_s1_beta_05.sh" ;;
    s1-beta-06)  export INIT_MODEL_PATH="$MEITUAN_BASE_MODEL_PATH"; RUN_SCRIPT="${STAGED_DIR}/run_s1_beta_06.sh" ;;
    s1-beta-07)  export INIT_MODEL_PATH="$MEITUAN_BASE_MODEL_PATH"; RUN_SCRIPT="${STAGED_DIR}/run_s1_beta_07.sh" ;;
    s1-beta-08)  export INIT_MODEL_PATH="$MEITUAN_BASE_MODEL_PATH"; RUN_SCRIPT="${STAGED_DIR}/run_s1_beta_08.sh" ;;
    s1-beta-09)  export INIT_MODEL_PATH="$MEITUAN_BASE_MODEL_PATH"; RUN_SCRIPT="${STAGED_DIR}/run_s1_beta_09.sh" ;;
    s1-beta-10)  export INIT_MODEL_PATH="$MEITUAN_BASE_MODEL_PATH"; RUN_SCRIPT="${STAGED_DIR}/run_s1_beta_10.sh" ;;
    s2-beta-0)   RUN_SCRIPT="${STAGED_DIR}/run_s2_beta_0.sh" ;;
    s2-beta-01)  RUN_SCRIPT="${STAGED_DIR}/run_s2_beta_01.sh" ;;
    s2-beta-02)  RUN_SCRIPT="${STAGED_DIR}/run_s2_beta_02.sh" ;;
    s2-beta-03)  RUN_SCRIPT="${STAGED_DIR}/run_s2_beta_03.sh" ;;
    s2-beta-04)  RUN_SCRIPT="${STAGED_DIR}/run_s2_beta_04.sh" ;;
    s2-beta-05)  RUN_SCRIPT="${STAGED_DIR}/run_s2_beta_05.sh" ;;
    *)
        echo "[staged_v1/meituan/jupyter.sh] ERROR: unsupported EXPERIMENT='$EXPERIMENT'" >&2
        exit 1
        ;;
esac

if [ ! -f "$RUN_SCRIPT" ]; then
    echo "[staged_v1/meituan/jupyter.sh] ERROR: run script not found: $RUN_SCRIPT" >&2
    exit 1
fi
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "[staged_v1/meituan/jupyter.sh] ERROR: BASE_MODEL_PATH not found: $BASE_MODEL_PATH" >&2
    exit 1
fi
if [ ! -f "$TRAIN_FILE" ]; then
    echo "[staged_v1/meituan/jupyter.sh] ERROR: TRAIN_FILE not found: $TRAIN_FILE" >&2
    exit 1
fi

echo "[staged_v1/meituan/jupyter.sh] resolved run script: $RUN_SCRIPT"
cd "$REPO_ROOT"
echo "[staged_v1/meituan/jupyter.sh] cwd = $(pwd); launching training..."
exec bash "$RUN_SCRIPT"
