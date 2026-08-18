#!/usr/bin/env bash
set -euo pipefail
: "${EXPERIMENT:?EXPERIMENT required: arm-c or arm-d0; arm-d is optional}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
source "${SCRIPT_DIR}/env.sh"
case "${EXPERIMENT,,}" in
    arm-c) RUN_SCRIPT="${MATH_DIR}/run_math_qwen3_1p7b_wdl_causal_arm_c.sh" ;;
    fixed-m1-stage1) RUN_SCRIPT="${MATH_DIR}/run_math_qwen3_1p7b_wdl_fixed_m1_stage1.sh" ;;
    fixed-m1-cold-start)
        RUN_SCRIPT="${MATH_DIR}/run_math_qwen3_1p7b_wdl_fixed_m1_cold_start.sh"
        export MODEL2_PATH="$BASE_MODEL_PATH"
        export STAGE1_MODEL2_PROVENANCE_FILE="$BASE_MODEL_PATH/format_cold_start_source.json"
        export EXPECTED_MODEL2_PROVENANCE_SHA256="$EXPECTED_MODEL1_SOURCE_SHA256"
        export EXPECTED_MODEL2_CONFIG_SHA256="$EXPECTED_MODEL1_CONFIG_SHA256"
        export EXPECTED_MODEL2_WEIGHTS_SHA256="$EXPECTED_MODEL1_WEIGHTS_SHA256"
        ;;
    arm-d)
        if [ "${RUN_OPTIONAL_D:-0}" != "1" ]; then
            echo "ERROR: arm-d is optional after equivalence; set RUN_OPTIONAL_D=1 to run it explicitly" >&2
            exit 1
        fi
        RUN_SCRIPT="${MATH_DIR}/run_math_qwen3_1p7b_wdl_causal_arm_d.sh"
        ;;
    arm-d0) RUN_SCRIPT="${MATH_DIR}/run_math_qwen3_1p7b_wdl_causal_arm_d0.sh" ;;
    *) echo "ERROR: unsupported EXPERIMENT=$EXPERIMENT" >&2; exit 1 ;;
esac
if [[ "${EXPERIMENT,,}" == fixed-m1-* ]]; then
    export WANDB_PROJECT=OnPolicyWDLSFT-Math-1P7B-Fixed-M1-P60
    export CAUSAL_ARTIFACT_ROOT="$DATA_ROOT/model_weights/math_task/qwen3_1p7b_wdl_fixed_m1_p60"
    export WANDB_DIR="$DATA_ROOT/wandb_runs/math_wdl_fixed_m1_p60"
    export LOG_DIR="$CAUSAL_ARTIFACT_ROOT/logs"
    export VERL_FILE_LOGGER_ROOT="$LOG_DIR/metrics"
    export VALIDATION_DATA_DIR="$LOG_DIR/validation"
    export WDL_MANIPULATION_RECEIPT="$DATA_ROOT/model_weights/math_task/qwen3_1p7b_wdl_causal_p60/admission/manipulation_receipt.json"
    mkdir -p "$CAUSAL_ARTIFACT_ROOT" "$WANDB_DIR" "$LOG_DIR" "$VERL_FILE_LOGGER_ROOT" "$VALIDATION_DATA_DIR"
fi
if [ "${DRY_RUN:-0}" = "1" ]; then
    printf '[MEITUAN DRY RUN] experiment=%s script=%s base_model=%s model2=%s train=%s math7=%s\n' \
        "$EXPERIMENT" "$RUN_SCRIPT" "$BASE_MODEL_PATH" "$MODEL2_PATH" "$TRAIN_FILE" "$MATH7_VALIDATION_ROOT"
    exit 0
fi
if [ "${DRY_RUN:-0}" != "1" ]; then
    for required in "$BASE_MODEL_PATH" "$MODEL2_PATH" "$STAGE1_MODEL2_PROVENANCE_FILE" "$TRAIN_FILE"; do
        [ -e "$required" ] || { echo "ERROR: required input missing: $required" >&2; exit 1; }
    done
    hash_check() {
        local path="$1" expected="$2" label="$3" actual
        actual=$(sha256sum "$path" | awk '{print $1}')
        [ "$actual" = "$expected" ] || { echo "ERROR: $label hash mismatch: $actual != $expected" >&2; exit 1; }
    }
    hash_check "$BASE_MODEL_PATH/format_cold_start_source.json" "$EXPECTED_MODEL1_SOURCE_SHA256" model1-source
    hash_check "$BASE_MODEL_PATH/config.json" "$EXPECTED_MODEL1_CONFIG_SHA256" model1-config
    hash_check "$BASE_MODEL_PATH/model.safetensors" "$EXPECTED_MODEL1_WEIGHTS_SHA256" model1-weights
    hash_check "$STAGE1_MODEL2_PROVENANCE_FILE" "$EXPECTED_MODEL2_PROVENANCE_SHA256" model2-provenance
    hash_check "$MODEL2_PATH/config.json" "$EXPECTED_MODEL2_CONFIG_SHA256" model2-config
    hash_check "$MODEL2_PATH/model.safetensors" "$EXPECTED_MODEL2_WEIGHTS_SHA256" model2-weights
    hash_check "$TRAIN_FILE" "$EXPECTED_TRAIN_SHA256" train-shard
    required_math7_files=(
        aime-2025_with_system_prompt_schema_aligned.parquet
        math500-test_with_system_prompt_schema_aligned.parquet
        amc23-test_with_system_prompt_schema_aligned.parquet
        aqua-test_with_system_prompt_schema_aligned.parquet
        gsm8k-test_with_system_prompt_schema_aligned.parquet
        mawps-test_with_system_prompt_schema_aligned.parquet
        svamp-test_with_system_prompt_schema_aligned.parquet
    )
    for val_name in "${required_math7_files[@]}"; do
        val_file="${MATH7_VALIDATION_ROOT}/${val_name}"
        [ -f "$val_file" ] || { echo "ERROR: Math-7 validation file missing: $val_file" >&2; exit 1; }
    done
fi
cd "$REPO_ROOT"
exec bash "$RUN_SCRIPT"
