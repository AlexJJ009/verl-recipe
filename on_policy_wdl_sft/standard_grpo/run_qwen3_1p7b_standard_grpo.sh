#!/usr/bin/env bash
set -euo pipefail

# Single implementation for the four friendly task/pipeline entry points.
# Formal launches are gated; GRPO_CONFIG_ONLY=1 prints the frozen contract.

: "${TASK:?TASK must be math or code}"
: "${PIPELINE:?PIPELINE must be stage1_grpo or cold_start_grpo}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMMON_LAUNCHER="${RECIPE_ROOT}/on_policy_wdl_sft/ablation_single_model/_common_ablation.sh"
CALLER_WANDB_PROJECT=${WANDB_PROJECT-}
CALLER_TEST_FILES=${TEST_FILES-}

export LOSS_MODE=vanilla
export LR=${LR:-5e-7}
export LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-0}
export TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-64}
export ROLLOUT_N=${ROLLOUT_N:-8}
STANDARD_GRPO_TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ:-${TRAIN_PROMPT_BSZ}}
# Stage123 resource profiles still validate their historical response-count
# convention while being sourced. Restore the GRPO prompt-group value below.
unset TRAIN_PROMPT_MINI_BSZ
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-seq-mean-token-mean}
export ACTOR_GRAD_CLIP=${ACTOR_GRAD_CLIP:-1.0}
export NORM_ADV_BY_STD_IN_GRPO=${NORM_ADV_BY_STD_IN_GRPO:-True}
export CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
export CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.2}
export USE_KL_IN_REWARD=${USE_KL_IN_REWARD:-False}
export KL_COEF=${KL_COEF:-0.0}
export USE_KL_LOSS=${USE_KL_LOSS:-True}
export KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
export KL_LOSS_TYPE=${KL_LOSS_TYPE:-low_var_kl}
export ROLLOUT_IS=${ROLLOUT_IS:-null}
export DATA_SHUFFLE=${DATA_SHUFFLE:-False}
export ENABLE_THINKING=${ENABLE_THINKING:-True}
export TEMPERATURE=${TEMPERATURE:-1.0}
export TOP_P=${TOP_P:-1.0}
export TOP_K=${TOP_K:--1}
export TEST_FREQ=${TEST_FREQ:-5}
export SAVE_FREQ=${SAVE_FREQ:-5}
# Keep only scientific phase boundaries plus best/latest. Older retained
# checkpoints are model-only; latest keeps optimizer state for continuation.
export PROTECTED_CKPT_STRIP_OPTIMIZER=${PROTECTED_CKPT_STRIP_OPTIMIZER:-True}
export VAL_N=${VAL_N:-3}
export VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.2}
export VAL_TOP_P=${VAL_TOP_P:-0.95}
export VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
export WANDB_MODE=${WANDB_MODE:-offline}
export REF_FSDP_OFFLOAD=${REF_FSDP_OFFLOAD:-True}
export ACTOR_CALCULATE_ENTROPY=${ACTOR_CALCULATE_ENTROPY:-False}
export CALCULATE_ENTROPY=${CALCULATE_ENTROPY:-False}
# The historical Code Stage123 profile defaulted to n=1; this baseline owns n=3.
export STAGE123_EXPECTED_VAL_N=${STAGE123_EXPECTED_VAL_N:-3}

case "${TASK}" in
  math)
    # shellcheck disable=SC1091
    source "${RECIPE_ROOT}/on_policy_wdl_sft/math_task/qwen3_1p7b_math_stage123_resource_profile.sh"
    DATASET_ROOT=${DATASET_ROOT:-/data-2/dataset/math/qwen3_1p7b_stage123_seed20260719}
    COLD_START_MODEL_PATH=${COLD_START_MODEL_PATH:-/data-2/model_weights/math_task/qwen3_1p7b_cold_start_cotmask_v3/candidates/step_20}
    STAGE1_MODEL_PATH=${STAGE1_MODEL_PATH:-}
    MATH_STAGE1_MODEL_PROVENANCE_PATH=${MATH_STAGE1_MODEL_PROVENANCE_PATH:-${STAGE1_MODEL_PATH}/model_input_provenance.json}
    MATH_STAGE1_MODEL_WEIGHTS_SHA256=${MATH_STAGE1_MODEL_WEIGHTS_SHA256:-ff8ff12d311bcc862247bd1d13f4380ec53f8af87095b183cf393147222d94b0}
    MATH_STAGE1_SOURCE_JOINT_WEIGHTS_SHA256=${MATH_STAGE1_SOURCE_JOINT_WEIGHTS_SHA256:-a327d9975f9f95d36505fc80fcaf689fe3f13a9a80bd72a74d436e5106a5c850}
    MATH7_VALIDATION_ROOT=${MATH7_VALIDATION_ROOT:-/data-1/dataset/math/qwen3_1p7b_math7_validation_v1}
    export TEST_FILES=${CALLER_TEST_FILES:-"['${MATH7_VALIDATION_ROOT}/aime-2025_with_system_prompt_schema_aligned.parquet','${MATH7_VALIDATION_ROOT}/math500-test_with_system_prompt_schema_aligned.parquet','${MATH7_VALIDATION_ROOT}/amc23-test_with_system_prompt_schema_aligned.parquet','${MATH7_VALIDATION_ROOT}/aqua-test_with_system_prompt_schema_aligned.parquet','${MATH7_VALIDATION_ROOT}/gsm8k-test_with_system_prompt_schema_aligned.parquet','${MATH7_VALIDATION_ROOT}/mawps-test_with_system_prompt_schema_aligned.parquet','${MATH7_VALIDATION_ROOT}/svamp-test_with_system_prompt_schema_aligned.parquet']"}
    export CUSTOM_REWARD_FN_PATH=${CUSTOM_REWARD_FN_PATH:-"${RECIPE_ROOT}/on_policy_wdl_sft/custom_reward_function_latex_verify.py"}
    export CUSTOM_REWARD_FN_NAME=${CUSTOM_REWARD_FN_NAME:-compute_score_latex_verify}
    export WANDB_PROJECT=${CALLER_WANDB_PROJECT:-StandardGRPO-Qwen3-1P7B-Math}
    export BASE_CKPT_DIR=${BASE_CKPT_DIR:-/data-2/checkpoints/standard_grpo_qwen3_1p7b_math}
    export LOG_DIR=${LOG_DIR:-/data-2/logs/standard_grpo_qwen3_1p7b_math}
    macro_overrides=(
      "+trainer.validation_macro_average_sources=${MATH7_MACRO_SOURCES}"
      "+trainer.validation_macro_average_name=math7_macro"
      "+trainer.validation_macro_average_metric=acc/mean@3"
    )
    ;;
  code)
    # shellcheck disable=SC1091
    if [ "${GRPO_CONFIG_ONLY:-0}" = 1 ]; then
      export STAGE123_VALIDATE_EXTERNAL_ASSETS=0
    fi
    source "${RECIPE_ROOT}/on_policy_wdl_sft/code_task/qwen3_1p7b_stage123_resource_profile.sh"
    DATASET_ROOT=${DATASET_ROOT:-/data-2/dataset/code/verl_rl/qwen3_1p7b_code_stage123_author_signature_v2_seed20260706}
    COLD_START_MODEL_PATH=${COLD_START_MODEL_PATH:-/data-1/model_weights/code_task/qwen3_1p7b_cold_start_cotmask_v3_author_signature_v2_steps/candidates/step_20}
    STAGE1_MODEL_PATH=${STAGE1_MODEL_PATH:-/data-2/model_weights/code_task/qwen3_1p7b_stage123_cotmask_v3_author_signature_v2_step20/b0-stage1/final_model}
    CODE_ONLINE_HUMANEVAL_PLUS_VAL_FILE=${CODE_ONLINE_HUMANEVAL_PLUS_VAL_FILE:-/data-1/dataset/code/verl_rl/online_full_humaneval_plus/official_humaneval_plus_val.parquet}
    CODE_ONLINE_MBPP_PLUS_VAL_FILE=${CODE_ONLINE_MBPP_PLUS_VAL_FILE:-/data-1/dataset/code/verl_rl/online_full_mbpp_plus/official_mbpp_plus_val.parquet}
    CODE_ONLINE_LCB_V5_SUBSET_VAL_FILE=${CODE_ONLINE_LCB_V5_SUBSET_VAL_FILE:-/data-1/dataset/code/verl_rl/online_full_livecodebench_v5/official_livecodebench_val.parquet}
    export TEST_FILES=${CALLER_TEST_FILES:-"['${CODE_ONLINE_HUMANEVAL_PLUS_VAL_FILE}','${CODE_ONLINE_MBPP_PLUS_VAL_FILE}','${CODE_ONLINE_LCB_V5_SUBSET_VAL_FILE}']"}
    CODE3_MACRO_SOURCES=${CODE3_MACRO_SOURCES:-"[HumanEval+,MBPP+,LiveCodeBench]"}
    export CUSTOM_REWARD_FN_PATH=${CUSTOM_REWARD_FN_PATH:-"${RECIPE_ROOT}/on_policy_wdl_sft/code_task/official_aligned_reward.py"}
    export CUSTOM_REWARD_FN_NAME=${CUSTOM_REWARD_FN_NAME:-compute_score_code_official_aligned}
    export WANDB_PROJECT=${CALLER_WANDB_PROJECT:-StandardGRPO-Qwen3-1P7B-Code}
    export BASE_CKPT_DIR=${BASE_CKPT_DIR:-/data-2/checkpoints/standard_grpo_qwen3_1p7b_code}
    export LOG_DIR=${LOG_DIR:-/data-2/logs/standard_grpo_qwen3_1p7b_code}
    export PYTHONPATH="${REPO_PYTHONPATH_ROOT:-/workspace/verl}:${CODE_EVAL_OFFICIAL_SITE:-/data-1/code_eval_envs/official_site}:${LCB_REPO_DIR:-/data-1/code_eval_envs/LiveCodeBench}:${PYTHONPATH:-}"
    macro_overrides=(
      "+trainer.validation_macro_average_sources=${CODE3_MACRO_SOURCES}"
      "+trainer.validation_macro_average_name=code3_macro"
      "+trainer.validation_macro_average_metric=acc/mean@3"
    )
    ;;
  *) echo "ERROR: unsupported TASK=${TASK}" >&2; exit 2 ;;
esac

export TRAIN_PROMPT_MINI_BSZ=${STANDARD_GRPO_TRAIN_PROMPT_MINI_BSZ}

case "${PIPELINE}" in
  stage1_grpo)
    if [ -z "${STAGE1_MODEL_PATH}" ]; then
      echo "ERROR: STAGE1_MODEL_PATH is required for ${TASK} Stage1 + GRPO" >&2
      exit 2
    fi
    export INIT_MODEL_PATH=${INIT_MODEL_PATH:-${STAGE1_MODEL_PATH}}
    export TRAIN_FILE=${TRAIN_FILE:-${DATASET_ROOT}/stage1_control_stage2_then_stage3.parquet}
    export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-60}
    export PROTECTED_CKPT_STEPS=${PROTECTED_CKPT_STEPS:-'[20,40,60]'}
    export RUN_PREFIX=${RUN_PREFIX:-"${TASK^^}-QWEN3-1P7B-STAGE1-GRPO"}
    ;;
  cold_start_grpo)
    export INIT_MODEL_PATH=${INIT_MODEL_PATH:-${COLD_START_MODEL_PATH}}
    export TRAIN_FILE=${TRAIN_FILE:-${DATASET_ROOT}/cold_start_grpo_stage1_stage2_stage3.parquet}
    export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
    export PROTECTED_CKPT_STEPS=${PROTECTED_CKPT_STEPS:-'[40,60,80,100]'}
    export RUN_PREFIX=${RUN_PREFIX:-"${TASK^^}-QWEN3-1P7B-COLD-START-GRPO"}
    ;;
  *) echo "ERROR: unsupported PIPELINE=${PIPELINE}" >&2; exit 2 ;;
esac

if [ "${GRPO_CONFIG_ONLY:-0}" = 1 ]; then
  printf '%s\n' \
    "task=${TASK}" "pipeline=${PIPELINE}" "run_prefix=${RUN_PREFIX}" \
    "wandb_project=${WANDB_PROJECT}" \
    "init_model_path=${INIT_MODEL_PATH}" "train_file=${TRAIN_FILE}" \
    "total_training_steps=${TOTAL_TRAINING_STEPS}" "train_prompt_bsz=${TRAIN_PROMPT_BSZ}" \
    "rollout_n=${ROLLOUT_N}" "responses_per_step=$((TRAIN_PROMPT_BSZ * ROLLOUT_N))" \
    "ppo_mini_batch_size=${TRAIN_PROMPT_MINI_BSZ}" "learning_rate=${LR}" \
    "actor_grad_clip=${ACTOR_GRAD_CLIP}" \
    "loss_mode=${LOSS_MODE}" "loss_agg_mode=${LOSS_AGG_MODE}" \
    "norm_adv_by_std_in_grpo=${NORM_ADV_BY_STD_IN_GRPO}" \
    "use_kl_in_reward=${USE_KL_IN_REWARD}" "use_kl_loss=${USE_KL_LOSS}" \
    "kl_loss_coef=${KL_LOSS_COEF}" "kl_loss_type=${KL_LOSS_TYPE}" \
    "ref_log_prob_micro_batch_size=${REF_LOG_PROB_MICRO_BATCH_SIZE}" \
    "ref_log_prob_max_token_len_per_gpu=${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}}" \
    "rollout_is=${ROLLOUT_IS}" "enable_thinking=${ENABLE_THINKING}" \
    "data_shuffle=${DATA_SHUFFLE}" "protected_ckpt_steps=${PROTECTED_CKPT_STEPS}" \
    "protected_ckpt_strip_optimizer=${PROTECTED_CKPT_STRIP_OPTIMIZER}"
  exit 0
fi

if [ "${GRPO_LAUNCH_ALLOWED:-0}" != 1 ]; then
  echo "ERROR: formal launch requires GRPO_LAUNCH_ALLOWED=1" >&2
  exit 2
fi
if [ -z "${TMUX:-}" ] && [ "${GRPO_SCHEDULER_MANAGED:-0}" != 1 ]; then
  echo "ERROR: formal GRPO training must run inside tmux or an admitted scheduler-managed worker" >&2
  exit 2
fi
for required_path in "${INIT_MODEL_PATH}" "${TRAIN_FILE}"; do
  if [ ! -e "${required_path}" ]; then
    echo "ERROR: required input does not exist: ${required_path}" >&2
    exit 2
  fi
done

if [ "${TASK}" = math ] && [ "${PIPELINE}" = stage1_grpo ]; then
  if [ ! -f "${MATH_STAGE1_MODEL_PROVENANCE_PATH}" ]; then
    echo "ERROR: Math S1-P0 provenance receipt missing: ${MATH_STAGE1_MODEL_PROVENANCE_PATH}" >&2
    exit 2
  fi
  python3 - "${INIT_MODEL_PATH}" "${MATH_STAGE1_MODEL_PROVENANCE_PATH}" \
    "${MATH_STAGE1_MODEL_WEIGHTS_SHA256}" "${MATH_STAGE1_SOURCE_JOINT_WEIGHTS_SHA256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

model_path = Path(sys.argv[1])
receipt_path = Path(sys.argv[2])
expected_model_sha, expected_source_sha = sys.argv[3:]
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
if receipt.get("operation") != "extract_joint_submodel" or receipt.get("source", {}).get("sub_model_index") != 1:
    raise SystemExit("Math S1-P0 receipt is not a Model2 joint-submodel extraction")
source_hashes = {item.get("sha256") for item in receipt.get("source", {}).get("safetensors", [])}
if expected_source_sha not in source_hashes:
    raise SystemExit("Math S1-P0 source joint hash mismatch")
files = {item.get("name"): item for item in receipt.get("output", {}).get("files", [])}
model_entry = files.get("model.safetensors")
if not model_entry or model_entry.get("sha256") != expected_model_sha:
    raise SystemExit("Math S1-P0 model hash mismatch in provenance receipt")
for name, item in files.items():
    path = model_path / name
    if not path.is_file() or path.stat().st_size != item.get("size_bytes"):
        raise SystemExit(f"Math S1-P0 file missing or size mismatch: {name}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != item.get("sha256"):
        raise SystemExit(f"Math S1-P0 file hash mismatch: {name}")
PY
fi

if [ "${GRPO_PREFLIGHT_ONLY:-0}" = 1 ]; then
  echo "GRPO preflight passed: task=${TASK} pipeline=${PIPELINE}"
  exit 0
fi

common_overrides=(
  "data.shuffle=${DATA_SHUFFLE}"
  "+data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING}"
  "actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE}"
)
# shellcheck disable=SC1090
source "${COMMON_LAUNCHER}" "${common_overrides[@]}" "${macro_overrides[@]}" "$@"
