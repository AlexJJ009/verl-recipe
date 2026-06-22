#!/usr/bin/env bash
# Meituan AFO worker entry point for code-task runs.
set -euo pipefail

: "${EXPERIMENT:?EXPERIMENT must be set}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export REPO_PYTHONPATH_ROOT=${REPO_PYTHONPATH_ROOT:-$REPO_ROOT}

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

case "${EXPERIMENT,,}" in
    s1-code-smoke-beta-0) RUN_SCRIPT="${CODE_DIR}/run_s1_code_smoke_beta_0.sh" ;;
    s1-code-pilot-beta-0) RUN_SCRIPT="${CODE_DIR}/run_s1_code_pilot_beta_0.sh" ;;
    s1-code-onpolicy-sft-beta-0) RUN_SCRIPT="${CODE_DIR}/run_s1_code_onpolicy_sft_beta_0.sh" ;;
    s1-code-onpolicy-sft-beta-01) RUN_SCRIPT="${CODE_DIR}/run_s1_code_onpolicy_sft_beta_01.sh" ;;
    s1-code-onpolicy-sft-beta-0-retention) RUN_SCRIPT="${CODE_DIR}/run_s1_code_onpolicy_sft_beta_0_retention.sh" ;;
    s1-code-onpolicy-sft-beta-01-retention) RUN_SCRIPT="${CODE_DIR}/run_s1_code_onpolicy_sft_beta_01_retention.sh" ;;
    s1-code-deepcoder-beta-0-retention)
        export CODE_TRAIN_FILE=${DEEPCODER_TRAIN_FILE}
        export TRAIN_FILE=${DEEPCODER_TRAIN_FILE}
        RUN_SCRIPT="${CODE_DIR}/run_s1_code_deepcoder_beta_0_retention.sh"
        ;;
    s1-code-deepcoder-beta-01-retention)
        export CODE_TRAIN_FILE=${DEEPCODER_TRAIN_FILE}
        export TRAIN_FILE=${DEEPCODER_TRAIN_FILE}
        RUN_SCRIPT="${CODE_DIR}/run_s1_code_deepcoder_beta_01_retention.sh"
        ;;
    s2-code-smoke-beta0-beta0)
        export TRAIN_FILE=${CODE_STAGE2_TRAIN_FILE}
        RUN_SCRIPT="${CODE_DIR}/run_s2_code_smoke_beta0_beta0.sh"
        ;;
    s2-code-pilot-beta0-beta0)
        export TRAIN_FILE=${CODE_STAGE2_TRAIN_FILE}
        RUN_SCRIPT="${CODE_DIR}/run_s2_code_pilot_beta0_beta0.sh"
        ;;
    s2-code-retention-beta0-beta0)
        export TRAIN_FILE=${CODE_STAGE2_RETENTION_BETA0_TRAIN_FILE}
        RUN_SCRIPT="${CODE_DIR}/run_s2_code_retention_beta0_beta0.sh"
        ;;
    s2-code-retention-beta01-beta01)
        export TRAIN_FILE=${CODE_STAGE2_RETENTION_BETA01_TRAIN_FILE}
        RUN_SCRIPT="${CODE_DIR}/run_s2_code_retention_beta01_beta01.sh"
        ;;
    *)
        echo "[code_task/meituan/jupyter.sh] ERROR: unsupported EXPERIMENT=${EXPERIMENT}" >&2
        exit 1
        ;;
esac

if [ ! -f "$RUN_SCRIPT" ]; then
    echo "[code_task/meituan/jupyter.sh] ERROR: run script not found: $RUN_SCRIPT" >&2
    exit 1
fi

if [ "${DRY_RUN:-0}" != "1" ]; then
    [ -d "$BASE_MODEL_PATH" ] || { echo "ERROR: BASE_MODEL_PATH not found: $BASE_MODEL_PATH" >&2; exit 1; }
    [ -f "$TRAIN_FILE" ] || { echo "ERROR: TRAIN_FILE not found: $TRAIN_FILE" >&2; exit 1; }
    case "${EXPERIMENT,,}" in
        *pilot*|s1-code-onpolicy-sft-beta-0|s1-code-onpolicy-sft-beta-01|s1-code-onpolicy-sft-beta-0-retention|s1-code-onpolicy-sft-beta-01-retention|s1-code-deepcoder-beta-0-retention|s1-code-deepcoder-beta-01-retention|s2-code-retention-beta0-beta0|s2-code-retention-beta01-beta01) [ -n "$SANDBOX_FUSION_URL" ] || { echo "ERROR: SANDBOX_FUSION_URL required for non-smoke AFO runs" >&2; exit 1; } ;;
    esac
fi

export CUSTOM_REWARD_FN_PATH=${CUSTOM_REWARD_FN_PATH:-"${CODE_DIR}/official_aligned_reward.py"}
echo "[code_task/meituan/jupyter.sh] EXPERIMENT=$EXPERIMENT"
echo "[code_task/meituan/jupyter.sh] RUN_SCRIPT=$RUN_SCRIPT"
echo "[code_task/meituan/jupyter.sh] DRY_RUN=${DRY_RUN:-0}"
cd "$REPO_ROOT"
exec bash "$RUN_SCRIPT"
