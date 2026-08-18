#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_code_qwen3_1p7b_wdl_acd0_common.sh"
: "${RUN_PREFIX:?RUN_PREFIX required}"
: "${BASE_MODEL_PATH:?BASE_MODEL_PATH required}"
: "${MODEL2_PATH:?MODEL2_PATH required}"
: "${FUSION_LAMBDA:?FUSION_LAMBDA required}"
: "${FUSION_MODE:?FUSION_MODE required}"
export JOINT_MODEL_CACHE_ROOT=${JOINT_MODEL_CACHE_ROOT:-/data-1/.cache/huggingface}
if [ -z "${MODEL_PATH:-}" ]; then
  case "$RUN_PREFIX" in
    *ARM-D0*) arm_tag=d0 ;;
    *ARM-C*) arm_tag=c ;;
    *)
      echo "ERROR: cannot derive ACD0 joint arm tag from RUN_PREFIX=$RUN_PREFIX" >&2
      exit 1
      ;;
  esac
  run_timestamp=${WANDB_RUN_NAME:-}
  run_timestamp=${run_timestamp##*_}
  if [ -z "${WANDB_RUN_NAME:-}" ] || [ "$run_timestamp" = "$WANDB_RUN_NAME" ]; then
    run_timestamp=$(date +%s)
  fi
  export MODEL_PATH="${JOINT_MODEL_CACHE_ROOT}/code-acd0-${arm_tag}-${run_timestamp}"
fi
export STAGE2_HANDOFF_STEP=40
export ALLOW_EXTERNAL_MODEL2=1
export STAGE1_RUN_PREFIX=CODE-B0_STAGE1-QWEN3-1P7B-COTMASK-V3-AUTHOR-SIGNATURE-V2-STEP20
export EXPECTED_STAGE1_RUN_PREFIX=CODE-B0_STAGE1-QWEN3-1P7B-COTMASK-V3-AUTHOR-SIGNATURE-V2-STEP20
export EXPECTED_STAGE1_BETA=0.0
export MERGED_MODEL2_PROVENANCE_FILE=${MERGED_MODEL2_PROVENANCE_FILE:-${STAGE1_MODEL2_PROVENANCE_FILE:?STAGE1_MODEL2_PROVENANCE_FILE required}}
export EXPECTED_MODEL1_PATH="$BASE_MODEL_PATH"
export EXPECTED_MODEL1_CONFIG_SHA256=d0cd49de1bdd99bc3420e9f605db1bfdd54be14eb244e585263729d351570126
export EXPECTED_MODEL1_TOKENIZER_CONFIG_SHA256=443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0
export EXPECTED_MODEL1_CHAT_TEMPLATE_SHA256=a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8
export EXPECTED_MODEL1_PROVENANCE_PATH="$BASE_MODEL_PATH/format_cold_start_source.json"
export EXPECTED_MODEL1_PROVENANCE_SHA256=3dc86e2398a07e05b622e67e8d8f0bbfa231e2d89ba638738938b249e6b7d4f1
export SUBMODEL_KL_ENABLED=false
export SUBMODEL_KL_MODEL1_ENABLED=false
export SUBMODEL_KL_MODEL2_ENABLED=false
export TRACK_JOINT_SUBMODEL_LOSSES=true
export JOINT_VALIDATION_VIEWS='[model1,model2]'
exec bash "${SCRIPT_DIR}/run_s2_code_model2_rollout_common.sh" \
  +trainer.validation_macro_average_sources='[HumanEval+,MBPP+,LiveCodeBench]' \
  +trainer.validation_macro_average_name=code3_macro \
  +trainer.validation_macro_average_metric=acc/mean@3 "$@"
