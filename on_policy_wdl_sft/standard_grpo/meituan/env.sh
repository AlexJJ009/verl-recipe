#!/usr/bin/env bash
set -euo pipefail

# Host-specific endpoints stay outside Git. The AFO submission must supply LGX;
# all remaining defaults are derived from that private persistent root.
: "${LGX:?Set LGX to the private dolphinfs root before launching on AFO}"

export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_tmp}
export TMPDIR=${TMPDIR:-/tmp/verl_tmp}
export VLLM_CONFIG_ROOT=${VLLM_CONFIG_ROOT:-/tmp/vllm_config}
export VERL_ZMQ_IPC_DIR=${VERL_ZMQ_IPC_DIR:-${TMPDIR}}

GRPO_PERSIST_ROOT=${GRPO_PERSIST_ROOT:-${LGX}/verl-exp/standard_grpo_qwen3_1p7b}
export HF_HOME=${HF_HOME:-${GRPO_PERSIST_ROOT}/hf_cache}
export BASE_CKPT_DIR=${BASE_CKPT_DIR:-${GRPO_PERSIST_ROOT}/checkpoints}
export LOG_DIR=${LOG_DIR:-${GRPO_PERSIST_ROOT}/logs}
export WANDB_DIR=${WANDB_DIR:-${GRPO_PERSIST_ROOT}/wandb_runs}
export WANDB_MODE=offline
export GRPO_LAUNCH_ALLOWED=1
export GRPO_SCHEDULER_MANAGED=1

export MATH_DATASET_ROOT=${MATH_DATASET_ROOT:-${GRPO_PERSIST_ROOT}/data/math_stage123}
export CODE_DATASET_ROOT=${CODE_DATASET_ROOT:-${GRPO_PERSIST_ROOT}/data/code_stage123}
export MATH_COLD_START_MODEL_PATH=${MATH_COLD_START_MODEL_PATH:-${GRPO_PERSIST_ROOT}/models/math_cold_start}
export MATH_STAGE1_MODEL_PATH=${MATH_STAGE1_MODEL_PATH:-${GRPO_PERSIST_ROOT}/models/math_stage1_model2}
export CODE_COLD_START_MODEL_PATH=${CODE_COLD_START_MODEL_PATH:-${GRPO_PERSIST_ROOT}/models/code_cold_start}
export CODE_STAGE1_MODEL_PATH=${CODE_STAGE1_MODEL_PATH:-${GRPO_PERSIST_ROOT}/models/code_stage1_model2}

export MATH_VALIDATION_ROOT=${MATH_VALIDATION_ROOT:-${GRPO_PERSIST_ROOT}/data/math7_validation}
export MATH7_TEST_FILES=${MATH7_TEST_FILES:-"['${MATH_VALIDATION_ROOT}/aime-2025_with_system_prompt_schema_aligned.parquet','${MATH_VALIDATION_ROOT}/math500-test_with_system_prompt_schema_aligned.parquet','${MATH_VALIDATION_ROOT}/amc23-test_with_system_prompt_schema_aligned.parquet','${MATH_VALIDATION_ROOT}/aqua-test_with_system_prompt_schema_aligned.parquet','${MATH_VALIDATION_ROOT}/gsm8k-test_with_system_prompt_schema_aligned.parquet','${MATH_VALIDATION_ROOT}/mawps-test_with_system_prompt_schema_aligned.parquet','${MATH_VALIDATION_ROOT}/svamp-test_with_system_prompt_schema_aligned.parquet']"}
export CODE_ONLINE_HUMANEVAL_PLUS_VAL_FILE=${CODE_ONLINE_HUMANEVAL_PLUS_VAL_FILE:-${GRPO_PERSIST_ROOT}/data/code_validation/official_humaneval_plus_val.parquet}
export CODE_ONLINE_MBPP_PLUS_VAL_FILE=${CODE_ONLINE_MBPP_PLUS_VAL_FILE:-${GRPO_PERSIST_ROOT}/data/code_validation/official_mbpp_plus_val.parquet}
export CODE_ONLINE_LCB_V5_SUBSET_VAL_FILE=${CODE_ONLINE_LCB_V5_SUBSET_VAL_FILE:-${GRPO_PERSIST_ROOT}/data/code_validation/official_livecodebench_val.parquet}
export CODE_EVAL_OFFICIAL_SITE=${CODE_EVAL_OFFICIAL_SITE:-${GRPO_PERSIST_ROOT}/code_eval_envs/official_site}
export LCB_REPO_DIR=${LCB_REPO_DIR:-${GRPO_PERSIST_ROOT}/code_eval_envs/LiveCodeBench}

mkdir -p "${RAY_TMPDIR}" "${TMPDIR}" "${VLLM_CONFIG_ROOT}" \
  "${HF_HOME}" "${BASE_CKPT_DIR}" "${LOG_DIR}" "${WANDB_DIR}"
