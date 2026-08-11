#!/usr/bin/env bash
set -euo pipefail

: "${EXPERIMENT:?Set EXPERIMENT to math-stage1-grpo, math-cold-start-grpo, code-stage1-grpo, or code-cold-start-grpo}"
: "${GRPO_INPUT_MANIFEST:?Set GRPO_INPUT_MANIFEST to the staged immutable input manifest}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

case "${EXPERIMENT}" in
  math-stage1-grpo)
    export DATASET_ROOT=${MATH_DATASET_ROOT}
    export STAGE1_MODEL_PATH=${MATH_STAGE1_MODEL_PATH}
    export TEST_FILES=${MATH7_TEST_FILES}
    RUN_SCRIPT=${GRPO_DIR}/run_math_stage1_grpo.sh
    INIT_PATH=${STAGE1_MODEL_PATH}
    TRAIN_PATH=${DATASET_ROOT}/stage1_control_stage2_then_stage3.parquet
    ;;
  math-cold-start-grpo)
    export DATASET_ROOT=${MATH_DATASET_ROOT}
    export COLD_START_MODEL_PATH=${MATH_COLD_START_MODEL_PATH}
    export TEST_FILES=${MATH7_TEST_FILES}
    RUN_SCRIPT=${GRPO_DIR}/run_math_cold_start_grpo.sh
    INIT_PATH=${COLD_START_MODEL_PATH}
    TRAIN_PATH=${DATASET_ROOT}/cold_start_grpo_stage1_stage2_stage3.parquet
    ;;
  code-stage1-grpo)
    export DATASET_ROOT=${CODE_DATASET_ROOT}
    export STAGE1_MODEL_PATH=${CODE_STAGE1_MODEL_PATH}
    RUN_SCRIPT=${GRPO_DIR}/run_code_stage1_grpo.sh
    INIT_PATH=${STAGE1_MODEL_PATH}
    TRAIN_PATH=${DATASET_ROOT}/stage1_control_stage2_then_stage3.parquet
    ;;
  code-cold-start-grpo)
    export DATASET_ROOT=${CODE_DATASET_ROOT}
    export COLD_START_MODEL_PATH=${CODE_COLD_START_MODEL_PATH}
    RUN_SCRIPT=${GRPO_DIR}/run_code_cold_start_grpo.sh
    INIT_PATH=${COLD_START_MODEL_PATH}
    TRAIN_PATH=${DATASET_ROOT}/cold_start_grpo_stage1_stage2_stage3.parquet
    ;;
  *) echo "ERROR: unsupported EXPERIMENT=${EXPERIMENT}" >&2; exit 2 ;;
esac

for path in "${GRPO_INPUT_MANIFEST}" "${INIT_PATH}" "${TRAIN_PATH}" "${RUN_SCRIPT}"; do
  if [ ! -e "${path}" ]; then
    echo "ERROR: required staged input does not exist: ${path}" >&2
    exit 2
  fi
done

cd "${REPO_ROOT}"
exec bash "${RUN_SCRIPT}" "$@"
