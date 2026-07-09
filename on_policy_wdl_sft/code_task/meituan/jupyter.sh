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
    s1-code-deepcoder-instruct2507-r8k-beta-0)
        export INIT_MODEL_PATH=${INSTRUCT2507_MODEL_PATH}
        export CODE_TRAIN_FILE=${DEEPCODER_TRAIN_FILE}
        export TRAIN_FILE=${DEEPCODER_TRAIN_FILE}
        RUN_SCRIPT="${CODE_DIR}/run_s1_code_deepcoder_instruct2507_r8k_beta_0.sh"
        ;;
    s1-code-deepcoder-instruct2507-r8k-beta-01)
        export INIT_MODEL_PATH=${INSTRUCT2507_MODEL_PATH}
        export CODE_TRAIN_FILE=${DEEPCODER_TRAIN_FILE}
        export TRAIN_FILE=${DEEPCODER_TRAIN_FILE}
        RUN_SCRIPT="${CODE_DIR}/run_s1_code_deepcoder_instruct2507_r8k_beta_01.sh"
        ;;
    s1-code-kodcode-qwen3-1p7b-instruct-ctx8k-beta-0)
        export INIT_MODEL_PATH=${QWEN3_1P7B_MODEL_PATH}
        RUN_SCRIPT="${CODE_DIR}/run_s1_code_kodcode_qwen3_1p7b_instruct_ctx8k_beta_0.sh"
        ;;
    s1-code-kodcode-qwen3-1p7b-instruct-ctx8k-beta-01)
        export INIT_MODEL_PATH=${QWEN3_1P7B_MODEL_PATH}
        RUN_SCRIPT="${CODE_DIR}/run_s1_code_kodcode_qwen3_1p7b_instruct_ctx8k_beta_01.sh"
        ;;
    s1-code-kodcode-qwen3-1p7b-coldstart-ctx8k-beta-0)
        export INIT_MODEL_PATH=${QWEN3_1P7B_CODE_COLDSTART_MODEL_PATH}
        RUN_SCRIPT="${CODE_DIR}/run_s1_code_kodcode_qwen3_1p7b_instruct_ctx8k_coldstart_beta_0.sh"
        ;;
    s1-code-kodcode-qwen3-1p7b-coldstart-ctx8k-beta-01)
        export INIT_MODEL_PATH=${QWEN3_1P7B_CODE_COLDSTART_MODEL_PATH}
        RUN_SCRIPT="${CODE_DIR}/run_s1_code_kodcode_qwen3_1p7b_instruct_ctx8k_coldstart_beta_01.sh"
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
    s2-code-kodcode-instruct2507-ctx8k-p60-beta0-beta0)
        export BASE_MODEL_PATH=${INSTRUCT2507_MODEL_PATH}
        export TRAIN_FILE=${CODE_STAGE2_KODCODE_I2507_CTX8K_P60_BETA0_TRAIN_FILE}
        export STAGE1_CKPT_DIR=${STAGE1_CKPT_DIR:-$CODE_STAGE2_KODCODE_I2507_CTX8K_P60_BETA0_STAGE1_CKPT_DIR}
        export STAGE1_MERGED_MODEL_ROOT=${STAGE1_MERGED_MODEL_ROOT:-$CODE_STAGE2_KODCODE_I2507_CTX8K_P60_BETA0_MERGED_MODEL_ROOT}
        export STAGE2_HANDOFF_STEP=${STAGE2_HANDOFF_STEP:-60}
        if [ "${MIN_FREE_GB_FOR_CKPT:-30}" = "30" ]; then
            export MIN_FREE_GB_FOR_CKPT=300
        fi
        RUN_SCRIPT="${CODE_DIR}/run_s2_code_kodcode_instruct2507_ctx8k_p60_beta0_beta0.sh"
        ;;
    s2-code-kodcode-instruct2507-ctx8k-p60-beta01-beta01)
        export BASE_MODEL_PATH=${INSTRUCT2507_MODEL_PATH}
        export TRAIN_FILE=${CODE_STAGE2_KODCODE_I2507_CTX8K_P60_BETA01_TRAIN_FILE}
        export STAGE1_CKPT_DIR=${STAGE1_CKPT_DIR:-$CODE_STAGE2_KODCODE_I2507_CTX8K_P60_BETA01_STAGE1_CKPT_DIR}
        export STAGE1_MERGED_MODEL_ROOT=${STAGE1_MERGED_MODEL_ROOT:-$CODE_STAGE2_KODCODE_I2507_CTX8K_P60_BETA01_MERGED_MODEL_ROOT}
        export STAGE2_HANDOFF_STEP=${STAGE2_HANDOFF_STEP:-60}
        if [ "${MIN_FREE_GB_FOR_CKPT:-30}" = "30" ]; then
            export MIN_FREE_GB_FOR_CKPT=300
        fi
        RUN_SCRIPT="${CODE_DIR}/run_s2_code_kodcode_instruct2507_ctx8k_p60_beta01_beta01.sh"
        ;;
    s2-code-kodcode-instruct2507-ctx8k-p40-beta01-subkl-smoke)
        export BASE_MODEL_PATH=${INSTRUCT2507_MODEL_PATH}
        export TRAIN_FILE=${CODE_STAGE2_KODCODE_I2507_CTX8K_P40_BETA01_TRAIN_FILE:-/data-1/dataset/code/verl_rl/kodcode_stage2_after_s1_seed20260604_beta01_p40_handoff.parquet}
        export STAGE1_CKPT_DIR=${STAGE1_CKPT_DIR:-${CODE_STAGE2_KODCODE_I2507_CTX8K_P40_BETA01_STAGE1_CKPT_DIR:-/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA01-V1_1782398871}}
        export STAGE1_MERGED_MODEL_ROOT=${CODE_STAGE2_KODCODE_I2507_CTX8K_P40_BETA01_MERGED_MODEL_ROOT:-/data-1/model_weights/code_task/kodcode_instruct2507_ctx8k_stage2_p40/beta01}
        export STAGE2_HANDOFF_STEP=${STAGE2_HANDOFF_STEP:-40}
        if [ "${MIN_FREE_GB_FOR_CKPT:-30}" = "30" ]; then
            export MIN_FREE_GB_FOR_CKPT=60
        fi
        RUN_SCRIPT="${CODE_DIR}/run_s2_code_kodcode_instruct2507_ctx8k_p40_beta01_subkl_smoke.sh"
        ;;
    s2-code-kodcode-qwen3-1p7b-instruct-ctx8k-p40-common)
        export BASE_MODEL_PATH=${QWEN3_1P7B_MODEL_PATH}
        export TRAIN_FILE=${TRAIN_FILE:-${CODE_TRAIN_FILE:-$CODE_STAGE2_QWEN3_1P7B_P40_BETA0_TRAIN_FILE}}
        export CODE_TRAIN_FILE=${CODE_TRAIN_FILE:-$TRAIN_FILE}
        export STAGE1_CKPT_DIR=${STAGE1_CKPT_DIR:-$CODE_STAGE2_QWEN3_1P7B_P40_BETA0_STAGE1_CKPT_DIR}
        export STAGE1_MERGED_MODEL_ROOT=${STAGE1_MERGED_MODEL_ROOT:-$CODE_STAGE2_QWEN3_1P7B_P40_MERGED_MODEL_ROOT}
        export STAGE2_HANDOFF_STEP=${STAGE2_HANDOFF_STEP:-40}
        if [ "${MIN_FREE_GB_FOR_CKPT:-30}" = "30" ]; then
            export MIN_FREE_GB_FOR_CKPT=300
        fi
        RUN_SCRIPT="${CODE_DIR}/run_s2_code_kodcode_qwen3_1p7b_instruct_ctx8k_p40_common.sh"
        ;;
    s2-code-kodcode-qwen3-1p7b-coldstart-ctx8k-p40-common)
        export BASE_MODEL_PATH=${QWEN3_1P7B_MODEL_PATH}
        export TRAIN_FILE=${TRAIN_FILE:-${CODE_TRAIN_FILE:-$CODE_STAGE2_QWEN3_1P7B_COLDSTART_P40_BETA0_TRAIN_FILE}}
        export CODE_TRAIN_FILE=${CODE_TRAIN_FILE:-$TRAIN_FILE}
        coldstart_stage2_beta_tag=beta0
        coldstart_stage1_default_prefix=ONPOLICY-SFT-Qwen3-1P7B-COLDSTART-CODE-KODCODE-CTX8K-S1-BETA0-V1
        if [ "${WDL_SFT_BETA:-0.0}" = "0.1" ]; then
            coldstart_stage2_beta_tag=beta01
            coldstart_stage1_default_prefix=ONPOLICY-SFT-Qwen3-1P7B-COLDSTART-CODE-KODCODE-CTX8K-S1-BETA01-V1
        fi
        export STAGE1_RUN_PREFIX=${STAGE1_RUN_PREFIX:-$coldstart_stage1_default_prefix}
        export STAGE1_CKPT_DIR=${STAGE1_CKPT_DIR:-$CODE_STAGE2_QWEN3_1P7B_COLDSTART_P40_BETA0_STAGE1_CKPT_DIR}
        export STAGE1_MERGED_MODEL_ROOT=${STAGE1_MERGED_MODEL_ROOT:-$CODE_STAGE2_QWEN3_1P7B_COLDSTART_P40_MERGED_MODEL_ROOT}
        export STAGE2_HANDOFF_STEP=${STAGE2_HANDOFF_STEP:-40}
        export MERGED_MODEL2_DIR=${MERGED_MODEL2_DIR:-$CODE_STAGE2_QWEN3_1P7B_COLDSTART_P40_MERGED_MODEL_ROOT/${coldstart_stage2_beta_tag}/${STAGE1_RUN_PREFIX}/step_${STAGE2_HANDOFF_STEP}}
        unset MERGED_MODEL2_PROVENANCE_FILE
        if [ "${MIN_FREE_GB_FOR_CKPT:-30}" = "30" ]; then
            export MIN_FREE_GB_FOR_CKPT=300
        fi
        RUN_SCRIPT="${CODE_DIR}/run_s2_code_kodcode_qwen3_1p7b_instruct_ctx8k_p40_common.sh"
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
    [ -d "${INIT_MODEL_PATH:-$BASE_MODEL_PATH}" ] || { echo "ERROR: INIT_MODEL_PATH/BASE_MODEL_PATH not found: ${INIT_MODEL_PATH:-$BASE_MODEL_PATH}" >&2; exit 1; }
    [ -f "$TRAIN_FILE" ] || { echo "ERROR: TRAIN_FILE not found: $TRAIN_FILE" >&2; exit 1; }
    case "${EXPERIMENT,,}" in
        s2-code-kodcode-instruct2507-ctx8k-p60-beta0-beta0|s2-code-kodcode-instruct2507-ctx8k-p60-beta01-beta01|s2-code-kodcode-instruct2507-ctx8k-p40-beta01-subkl-smoke|s2-code-kodcode-qwen3-1p7b-instruct-ctx8k-p40-common|s2-code-kodcode-qwen3-1p7b-coldstart-ctx8k-p40-common)
            [ -d "${STAGE1_CKPT_DIR}/global_step_${STAGE2_HANDOFF_STEP}/actor" ] || { echo "ERROR: Stage2 Stage1 actor not found: ${STAGE1_CKPT_DIR}/global_step_${STAGE2_HANDOFF_STEP}/actor" >&2; exit 1; }
            ;;
    esac
    case "${EXPERIMENT,,}" in
        *pilot*|s1-code-onpolicy-sft-beta-0|s1-code-onpolicy-sft-beta-01|s1-code-onpolicy-sft-beta-0-retention|s1-code-onpolicy-sft-beta-01-retention|s1-code-deepcoder-beta-0-retention|s1-code-deepcoder-beta-01-retention|s1-code-deepcoder-instruct2507-r8k-beta-0|s1-code-deepcoder-instruct2507-r8k-beta-01|s1-code-kodcode-qwen3-1p7b-instruct-ctx8k-beta-0|s1-code-kodcode-qwen3-1p7b-instruct-ctx8k-beta-01|s1-code-kodcode-qwen3-1p7b-coldstart-ctx8k-beta-0|s1-code-kodcode-qwen3-1p7b-coldstart-ctx8k-beta-01|s2-code-retention-beta0-beta0|s2-code-retention-beta01-beta01|s2-code-kodcode-instruct2507-ctx8k-p60-beta0-beta0|s2-code-kodcode-instruct2507-ctx8k-p60-beta01-beta01|s2-code-kodcode-qwen3-1p7b-instruct-ctx8k-p40-common|s2-code-kodcode-qwen3-1p7b-coldstart-ctx8k-p40-common) [ -n "$SANDBOX_FUSION_URL" ] || { echo "ERROR: SANDBOX_FUSION_URL required for non-smoke AFO runs" >&2; exit 1; } ;;
    esac
fi

export CUSTOM_REWARD_FN_PATH=${CUSTOM_REWARD_FN_PATH:-"${CODE_DIR}/official_aligned_reward.py"}
echo "[code_task/meituan/jupyter.sh] EXPERIMENT=$EXPERIMENT"
echo "[code_task/meituan/jupyter.sh] RUN_SCRIPT=$RUN_SCRIPT"
echo "[code_task/meituan/jupyter.sh] TRAIN_FILE=$TRAIN_FILE"
echo "[code_task/meituan/jupyter.sh] BASE_MODEL_PATH=$BASE_MODEL_PATH"
echo "[code_task/meituan/jupyter.sh] STAGE1_CKPT_DIR=${STAGE1_CKPT_DIR:-}"
echo "[code_task/meituan/jupyter.sh] STAGE2_HANDOFF_STEP=${STAGE2_HANDOFF_STEP:-}"
echo "[code_task/meituan/jupyter.sh] MIN_FREE_GB_FOR_CKPT=$MIN_FREE_GB_FOR_CKPT"
echo "[code_task/meituan/jupyter.sh] DRY_RUN=${DRY_RUN:-0}"
cd "$REPO_ROOT"
exec bash "$RUN_SCRIPT"
