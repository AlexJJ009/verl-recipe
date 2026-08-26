#!/usr/bin/env bash
set -euo pipefail

# Single implementation for the four friendly task/pipeline entry points.
# Formal launches are gated; GRPO_CONFIG_ONLY=1 prints the frozen contract.

: "${TASK:?TASK must be math or code}"
: "${PIPELINE:?PIPELINE must be stage1_grpo, cold_start_grpo, or c_wdl_p60_grpo}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMMON_LAUNCHER="${RECIPE_ROOT}/on_policy_wdl_sft/ablation_single_model/_common_ablation.sh"
CALLER_WANDB_PROJECT=${WANDB_PROJECT-}
CALLER_TEST_FILES=${TEST_FILES-}

export LOSS_MODE=vanilla
export LR=${LR:-1e-6}
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
export ROLLOUT_DO_SAMPLE=${ROLLOUT_DO_SAMPLE:-True}
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
export PPO_EPOCHS=${PPO_EPOCHS:-1}
export ACTOR_SHUFFLE=${ACTOR_SHUFFLE:-False}
# Match the actual A/C/D0 resolved contract. These are separate random sources:
# actor/FSDP and PPO data-loader use 42, vLLM used its legacy base default 0,
# and the frozen task dataset owns its preparation/data seed.
export TRAINING_SEED=${TRAINING_SEED:-42}
export ROLLOUT_SEED=${ROLLOUT_SEED:-0}
GRPO_SEED_REPLICATE=${GRPO_SEED_REPLICATE:-1}
case "${GRPO_SEED_REPLICATE}" in
  1) GRPO_EXPECTED_ACTOR_SEED=42; GRPO_EXPECTED_ROLLOUT_SEED=0 ;;
  2) GRPO_EXPECTED_ACTOR_SEED=43; GRPO_EXPECTED_ROLLOUT_SEED=1 ;;
  3) GRPO_EXPECTED_ACTOR_SEED=44; GRPO_EXPECTED_ROLLOUT_SEED=2 ;;
  *) echo "ERROR: GRPO_SEED_REPLICATE must be 1, 2, or 3" >&2; exit 2 ;;
esac
export RESUME_MODE=${RESUME_MODE:-disable}
GRPO_ADMISSION_CHECKER=${GRPO_ADMISSION_CHECKER:-"${SCRIPT_DIR}/../../../scripts/grpo_retrain_admission.py"}
GRPO_EXPECTED_IMAGE_DIGEST=${GRPO_EXPECTED_IMAGE_DIGEST:-sha256:c9d525a1f4b33267bd00be60fe00693338253537cac78151e4c55a6d3a7e5708}

case "${TASK}" in
  math)
    # shellcheck disable=SC1091
    source "${RECIPE_ROOT}/on_policy_wdl_sft/math_task/qwen3_1p7b_math_stage123_resource_profile.sh"
    GRPO_EXPECTED_DATA_SEED=20260719
    DATASET_ROOT=${DATASET_ROOT:-/data-2/dataset/math/qwen3_1p7b_stage123_seed20260719}
    COLD_START_MODEL_PATH=${COLD_START_MODEL_PATH:-/data-2/model_weights/math_task/qwen3_1p7b_cold_start_cotmask_v3/candidates/step_20}
    STAGE1_MODEL_PATH=${STAGE1_MODEL_PATH:-}
    MATH_STAGE1_MODEL_PROVENANCE_PATH=${MATH_STAGE1_MODEL_PROVENANCE_PATH:-${STAGE1_MODEL_PATH}/model_input_provenance.json}
    # ff8... is the recovered S1-P0 Model2, not the trained C-P60 Model2.
    # It was extracted from C's immutable prepared-input joint cache (a327...)
    # before C's first optimizer step.  The standalone safetensors serialization
    # differs from the deleted original file hash f069..., while every source
    # joint-cache Model2 tensor is exactly equal.  C-P60 is separately a6d8....
    MATH_STAGE1_MODEL_WEIGHTS_SHA256=${MATH_STAGE1_MODEL_WEIGHTS_SHA256:-ff8ff12d311bcc862247bd1d13f4380ec53f8af87095b183cf393147222d94b0}
    MATH_STAGE1_SOURCE_JOINT_WEIGHTS_SHA256=${MATH_STAGE1_SOURCE_JOINT_WEIGHTS_SHA256:-a327d9975f9f95d36505fc80fcaf689fe3f13a9a80bd72a74d436e5106a5c850}
    MATH_COLD_START_MODEL_WEIGHTS_SHA256=${MATH_COLD_START_MODEL_WEIGHTS_SHA256:-9ef4bee31240d3bd8de17d5e7ea2d74b1b8b78b3797f56fe440b0170d53bc207}
    MATH7_VALIDATION_ROOT=${MATH7_VALIDATION_ROOT:-/data-1/dataset/math/qwen3_1p7b_math7_validation_v1}
    export TEST_FILES=${CALLER_TEST_FILES:-"['${MATH7_VALIDATION_ROOT}/aime-2025_with_system_prompt_schema_aligned.parquet','${MATH7_VALIDATION_ROOT}/math500-test_with_system_prompt_schema_aligned.parquet','${MATH7_VALIDATION_ROOT}/amc23-test_with_system_prompt_schema_aligned.parquet','${MATH7_VALIDATION_ROOT}/aqua-test_with_system_prompt_schema_aligned.parquet','${MATH7_VALIDATION_ROOT}/gsm8k-test_with_system_prompt_schema_aligned.parquet','${MATH7_VALIDATION_ROOT}/mawps-test_with_system_prompt_schema_aligned.parquet','${MATH7_VALIDATION_ROOT}/svamp-test_with_system_prompt_schema_aligned.parquet']"}
    # Match the frozen A/C/D0 reward contract. The older scorer under
    # on_policy_wdl_sft/ accepts a correct boxed answer outside <answer>.
    export CUSTOM_REWARD_FN_PATH=${CUSTOM_REWARD_FN_PATH:-"${RECIPE_ROOT}/joint_training/custom_reward_function_latex_verify.py"}
    export CUSTOM_REWARD_FN_NAME=${CUSTOM_REWARD_FN_NAME:-compute_score_latex_verify}
    MATH_REWARD_CONTRACT_CHECKER=${MATH_REWARD_CONTRACT_CHECKER:-"${SCRIPT_DIR}/../../../scripts/check_math_reward_contract.py"}
    export WANDB_PROJECT=${CALLER_WANDB_PROJECT:-StandardGRPO-Qwen3-1P7B-Math}
    export BASE_CKPT_DIR=${BASE_CKPT_DIR:-/data-2/checkpoints/standard_grpo_qwen3_1p7b_math}
    export LOG_DIR=${LOG_DIR:-/data-2/logs/standard_grpo_qwen3_1p7b_math}
    export BEST_CKPT_METRIC_KEY=${BEST_CKPT_METRIC_KEY:-val-core/math7_macro/acc/mean@3}
    GRPO_EXPECTED_REWARD_SHA256=${GRPO_EXPECTED_REWARD_SHA256:-6fc2364da021bc5d14e1e3e8788d52cd49a3036088cacbb96d4eb5535e4473e5}
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
    export DATA_SEED=${DATA_SEED:-20260706}
    GRPO_EXPECTED_DATA_SEED=20260706
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
    export BEST_CKPT_METRIC_KEY=${BEST_CKPT_METRIC_KEY:-val-core/code3_macro/acc/mean@3}
    CODE_COLD_START_MODEL_WEIGHTS_SHA256=${CODE_COLD_START_MODEL_WEIGHTS_SHA256:-8330ea21b26e3d6f780df10e473dbf4393bea0438ccf861d1d994eb0a8abc955}
    CODE_STAGE1_MODEL_WEIGHTS_SHA256=${CODE_STAGE1_MODEL_WEIGHTS_SHA256:-a6c69262975ada9e1bc5054128d9f6f79b14167653ba817809bc771799d43c74}
    GRPO_EXPECTED_REWARD_SHA256=${GRPO_EXPECTED_REWARD_SHA256:-bbe6cafb0e3dcaf63bb9ca26df6d377e76077a9f5d2a6e11b1ab009bee23088a}
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

case "${TASK}:${PIPELINE}" in
  math:stage1_grpo|math:c_wdl_p60_grpo)
    # VERL filters 48 overlength prompts from the 3,840-row post-Stage1
    # parquet with the frozen Math tokenizer/template, then drops the trailing
    # 16 samples per epoch. The actual dataloader is therefore 59 batches/epoch.
    GRPO_EXPECTED_FILTERED_TRAIN_ROWS=${GRPO_EXPECTED_FILTERED_TRAIN_ROWS:-3792}
    GRPO_EXPECTED_DATALOADER_STEPS_PER_EPOCH=${GRPO_EXPECTED_DATALOADER_STEPS_PER_EPOCH:-59}
    ;;
  math:cold_start_grpo)
    # The 6,400-row Stage1->Stage2->Stage3 Math parquet filters to 6,324 rows;
    # drop_last=True yields 98 optimizer batches per dataloader epoch.
    GRPO_EXPECTED_FILTERED_TRAIN_ROWS=${GRPO_EXPECTED_FILTERED_TRAIN_ROWS:-6324}
    GRPO_EXPECTED_DATALOADER_STEPS_PER_EPOCH=${GRPO_EXPECTED_DATALOADER_STEPS_PER_EPOCH:-98}
    ;;
  *)
    GRPO_EXPECTED_FILTERED_TRAIN_ROWS=${GRPO_EXPECTED_FILTERED_TRAIN_ROWS:-}
    GRPO_EXPECTED_DATALOADER_STEPS_PER_EPOCH=${GRPO_EXPECTED_DATALOADER_STEPS_PER_EPOCH:-}
    ;;
esac

require_equal() {
  local name=$1 actual=$2 expected=$3
  if [ "${actual}" != "${expected}" ]; then
    echo "ERROR: canonical GRPO contract requires ${name}=${expected}, got ${actual}" >&2
    exit 2
  fi
}

case "${LR}" in
  5e-7|1e-6) ;;
  *) echo "ERROR: canonical GRPO LR must be one of 5e-7 or 1e-6, got ${LR}" >&2; exit 2 ;;
esac
require_equal LOSS_MODE "${LOSS_MODE}" vanilla
require_equal LR_WARMUP_STEPS "${LR_WARMUP_STEPS}" 0
require_equal TRAIN_PROMPT_BSZ "${TRAIN_PROMPT_BSZ}" 64
require_equal ROLLOUT_N "${ROLLOUT_N}" 8
require_equal TRAIN_PROMPT_MINI_BSZ "${TRAIN_PROMPT_MINI_BSZ}" 64
require_equal LOSS_AGG_MODE "${LOSS_AGG_MODE}" seq-mean-token-mean
require_equal ACTOR_GRAD_CLIP "${ACTOR_GRAD_CLIP}" 1.0
require_equal NORM_ADV_BY_STD_IN_GRPO "${NORM_ADV_BY_STD_IN_GRPO}" True
require_equal CLIP_RATIO_LOW "${CLIP_RATIO_LOW}" 0.2
require_equal CLIP_RATIO_HIGH "${CLIP_RATIO_HIGH}" 0.2
require_equal USE_KL_IN_REWARD "${USE_KL_IN_REWARD}" False
require_equal USE_KL_LOSS "${USE_KL_LOSS}" True
require_equal KL_LOSS_COEF "${KL_LOSS_COEF}" 0.001
require_equal KL_LOSS_TYPE "${KL_LOSS_TYPE}" low_var_kl
require_equal ROLLOUT_IS "${ROLLOUT_IS}" null
require_equal DATA_SHUFFLE "${DATA_SHUFFLE}" False
require_equal ENABLE_THINKING "${ENABLE_THINKING}" True
require_equal TEMPERATURE "${TEMPERATURE}" 1.0
require_equal TOP_P "${TOP_P}" 1.0
require_equal TOP_K "${TOP_K}" -1
require_equal ROLLOUT_DO_SAMPLE "${ROLLOUT_DO_SAMPLE}" True
require_equal VAL_N "${VAL_N}" 3
require_equal VAL_TEMPERATURE "${VAL_TEMPERATURE}" 0.2
require_equal VAL_TOP_P "${VAL_TOP_P}" 0.95
require_equal TEST_FREQ "${TEST_FREQ}" 5
require_equal SAVE_FREQ "${SAVE_FREQ}" 5
require_equal PPO_EPOCHS "${PPO_EPOCHS}" 1
require_equal ACTOR_SHUFFLE "${ACTOR_SHUFFLE}" False
require_equal VAL_DO_SAMPLE "${VAL_DO_SAMPLE}" True
require_equal TRAINING_SEED "${TRAINING_SEED}" "${GRPO_EXPECTED_ACTOR_SEED}"
require_equal ROLLOUT_SEED "${ROLLOUT_SEED}" "${GRPO_EXPECTED_ROLLOUT_SEED}"
require_equal DATA_SEED "${DATA_SEED}" "${GRPO_EXPECTED_DATA_SEED}"
require_equal RESUME_MODE "${RESUME_MODE}" disable
if [ "${TASK}" = math ]; then
  require_equal CUSTOM_REWARD_FN_PATH "${CUSTOM_REWARD_FN_PATH}" "${RECIPE_ROOT}/joint_training/custom_reward_function_latex_verify.py"
  require_equal CUSTOM_REWARD_FN_NAME "${CUSTOM_REWARD_FN_NAME}" compute_score_latex_verify
else
  require_equal CUSTOM_REWARD_FN_PATH "${CUSTOM_REWARD_FN_PATH}" "${RECIPE_ROOT}/on_policy_wdl_sft/code_task/official_aligned_reward.py"
  require_equal CUSTOM_REWARD_FN_NAME "${CUSTOM_REWARD_FN_NAME}" compute_score_code_official_aligned
fi

case "${PIPELINE}" in
  stage1_grpo)
    if [ -z "${STAGE1_MODEL_PATH}" ]; then
      echo "ERROR: STAGE1_MODEL_PATH is required for ${TASK} Stage1 + GRPO" >&2
      exit 2
    fi
    export INIT_MODEL_PATH=${INIT_MODEL_PATH:-${STAGE1_MODEL_PATH}}
    export TRAIN_FILE=${TRAIN_FILE:-${DATASET_ROOT}/stage1_control_stage2_then_stage3.parquet}
    export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-60}
    if [ "${TASK}" = math ]; then
      export TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
    else
      export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
    fi
    # local P20 is the Stage2 -> Stage3 boundary. The terminal local P60 is
    # retained independently as latest, so it need not be duplicated here.
    export PROTECTED_CKPT_STEPS=${PROTECTED_CKPT_STEPS:-'[20]'}
    export RUN_PREFIX=${RUN_PREFIX:-"${TASK^^}-QWEN3-1P7B-STAGE1-GRPO"}
    if [ "${TASK}" = math ]; then
      GRPO_EXPECTED_INIT_MODEL_SHA256=${GRPO_EXPECTED_INIT_MODEL_SHA256:-${MATH_STAGE1_MODEL_WEIGHTS_SHA256}}
      GRPO_EXPECTED_TRAIN_SHA256=${GRPO_EXPECTED_TRAIN_SHA256:-88d3accf25f54933b5776bfb0a4c07f5719a25199abc0ed800ccfc68eae15d66}
    else
      GRPO_EXPECTED_INIT_MODEL_SHA256=${GRPO_EXPECTED_INIT_MODEL_SHA256:-${CODE_STAGE1_MODEL_WEIGHTS_SHA256}}
      GRPO_EXPECTED_TRAIN_SHA256=${GRPO_EXPECTED_TRAIN_SHA256:-686856aa28c5928abf20922f3e4d4cec1ddd47e71650c1a62d4d592d83035f42}
    fi
    ;;
  cold_start_grpo)
    export INIT_MODEL_PATH=${INIT_MODEL_PATH:-${COLD_START_MODEL_PATH}}
    export TRAIN_FILE=${TRAIN_FILE:-${DATASET_ROOT}/cold_start_grpo_stage1_stage2_stage3.parquet}
    export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
    if [ "${TASK}" = math ]; then
      export TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
    else
      export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
    fi
    # P40/P60 are the Stage1 -> Stage2 and Stage2 -> Stage3 boundaries.
    # Terminal P100 is retained independently as latest.
    export PROTECTED_CKPT_STEPS=${PROTECTED_CKPT_STEPS:-'[40,60]'}
    export RUN_PREFIX=${RUN_PREFIX:-"${TASK^^}-QWEN3-1P7B-COLD-START-GRPO"}
    if [ "${TASK}" = math ]; then
      GRPO_EXPECTED_INIT_MODEL_SHA256=${GRPO_EXPECTED_INIT_MODEL_SHA256:-${MATH_COLD_START_MODEL_WEIGHTS_SHA256}}
      GRPO_EXPECTED_TRAIN_SHA256=${GRPO_EXPECTED_TRAIN_SHA256:-b17d6531ae9226b3fd6b5755423707060c1b7887c1a5542b5eef192e99a9d0d2}
    else
      GRPO_EXPECTED_INIT_MODEL_SHA256=${GRPO_EXPECTED_INIT_MODEL_SHA256:-${CODE_COLD_START_MODEL_WEIGHTS_SHA256}}
      GRPO_EXPECTED_TRAIN_SHA256=${GRPO_EXPECTED_TRAIN_SHA256:-053d60f602dbd5526f11dfdda8e55b3dc3a50a7b3f52bcd6c973586de14a1c7f}
    fi
    ;;
  c_wdl_p60_grpo)
    if [ "${TASK}" != math ]; then
      echo "ERROR: c_wdl_p60_grpo is defined only for Math C-P60" >&2
      exit 2
    fi
    if [ -z "${STAGE1_MODEL_PATH}" ]; then
      echo "ERROR: STAGE1_MODEL_PATH must identify the extracted C-P60 Model2" >&2
      exit 2
    fi
    export INIT_MODEL_PATH=${INIT_MODEL_PATH:-${STAGE1_MODEL_PATH}}
    export TRAIN_FILE=${TRAIN_FILE:-${DATASET_ROOT}/stage1_control_stage2_then_stage3.parquet}
    export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
    export TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
    export PROTECTED_CKPT_STEPS=${PROTECTED_CKPT_STEPS:-'[20,40,60,80]'}
    export RUN_PREFIX=${RUN_PREFIX:-MATH-QWEN3-1P7B-C-WDL-P60-THEN-GRPO}
    if [ "${GRPO_CONFIG_ONLY:-0}" = 1 ]; then
      GRPO_EXPECTED_INIT_MODEL_SHA256=${GRPO_EXPECTED_INIT_MODEL_SHA256:-CONFIG_ONLY}
    else
      : "${GRPO_EXPECTED_INIT_MODEL_SHA256:?C-P60 launch requires its admitted Model2 sha256}"
    fi
    GRPO_EXPECTED_TRAIN_SHA256=${GRPO_EXPECTED_TRAIN_SHA256:-88d3accf25f54933b5776bfb0a4c07f5719a25199abc0ed800ccfc68eae15d66}
    ;;
  *) echo "ERROR: unsupported PIPELINE=${PIPELINE}" >&2; exit 2 ;;
esac

if [ "${GRPO_CONFIG_ONLY:-0}" = 1 ]; then
  printf '%s\n' \
    "task=${TASK}" "pipeline=${PIPELINE}" "run_prefix=${RUN_PREFIX}" \
    "wandb_project=${WANDB_PROJECT}" \
    "custom_reward_fn_path=${CUSTOM_REWARD_FN_PATH}" "custom_reward_fn_name=${CUSTOM_REWARD_FN_NAME}" \
    "init_model_path=${INIT_MODEL_PATH}" "train_file=${TRAIN_FILE}" \
    "total_training_steps=${TOTAL_TRAINING_STEPS}" "train_prompt_bsz=${TRAIN_PROMPT_BSZ}" \
    "rollout_n=${ROLLOUT_N}" "responses_per_step=$((TRAIN_PROMPT_BSZ * ROLLOUT_N))" \
    "ppo_mini_batch_size=${TRAIN_PROMPT_MINI_BSZ}" "learning_rate=${LR}" \
    "ppo_epochs=${PPO_EPOCHS}" "actor_seed=${TRAINING_SEED}" \
    "rollout_seed=${ROLLOUT_SEED}" "seed_replicate=${GRPO_SEED_REPLICATE}" "data_seed=${DATA_SEED}" \
    "actor_shuffle=${ACTOR_SHUFFLE}" \
    "resume_mode=${RESUME_MODE}" "total_epochs=${TOTAL_EPOCHS}" \
    "expected_filtered_train_rows=${GRPO_EXPECTED_FILTERED_TRAIN_ROWS}" \
    "expected_dataloader_steps_per_epoch=${GRPO_EXPECTED_DATALOADER_STEPS_PER_EPOCH}" \
    "actor_grad_clip=${ACTOR_GRAD_CLIP}" \
    "loss_mode=${LOSS_MODE}" "loss_agg_mode=${LOSS_AGG_MODE}" \
    "norm_adv_by_std_in_grpo=${NORM_ADV_BY_STD_IN_GRPO}" \
    "use_kl_in_reward=${USE_KL_IN_REWARD}" "use_kl_loss=${USE_KL_LOSS}" \
    "kl_loss_coef=${KL_LOSS_COEF}" "kl_loss_type=${KL_LOSS_TYPE}" \
    "ref_log_prob_micro_batch_size=${REF_LOG_PROB_MICRO_BATCH_SIZE}" \
    "ref_log_prob_max_token_len_per_gpu=${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}}" \
    "rollout_is=${ROLLOUT_IS}" "enable_thinking=${ENABLE_THINKING}" \
    "data_shuffle=${DATA_SHUFFLE}" "protected_ckpt_steps=${PROTECTED_CKPT_STEPS}" \
    "protected_ckpt_strip_optimizer=${PROTECTED_CKPT_STRIP_OPTIMIZER}" \
    "best_ckpt_metric_key=${BEST_CKPT_METRIC_KEY}"
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
: "${GRPO_RUNTIME_IMAGE_DIGEST:?formal launch requires the observed runtime image digest}"
: "${GRPO_ADMISSION_RECEIPT:?formal launch requires a unique external admission receipt path}"
if [ ! -x "${GRPO_ADMISSION_CHECKER}" ]; then
  echo "ERROR: GRPO admission checker missing or not executable: ${GRPO_ADMISSION_CHECKER}" >&2
  exit 2
fi

if [ "${TASK}" = math ]; then
  if [ ! -f "${MATH_REWARD_CONTRACT_CHECKER}" ]; then
    echo "ERROR: Math reward contract checker missing: ${MATH_REWARD_CONTRACT_CHECKER}" >&2
    exit 2
  fi
  PYTHONPATH="${REPO_PYTHONPATH_ROOT:-${SCRIPT_DIR}/../../..}:${PYTHONPATH:-}" \
  python3 "${MATH_REWARD_CONTRACT_CHECKER}" \
    --reward-path "${CUSTOM_REWARD_FN_PATH}" \
    --function "${CUSTOM_REWARD_FN_NAME}"
fi

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

admission_source_args=()
if [ "${GRPO_SCHEDULER_MANAGED:-0}" = 1 ]; then
  : "${GRPO_ROOT_COMMIT:?scheduler launch requires the admitted root commit}"
  : "${GRPO_RECIPE_COMMIT:?scheduler launch requires the admitted recipe commit}"
  : "${GRPO_SNAPSHOT_DIGEST:?scheduler launch requires the admitted snapshot digest}"
  admission_source_args=(
    --scheduler-managed
    --root-commit "${GRPO_ROOT_COMMIT}"
    --recipe-commit "${GRPO_RECIPE_COMMIT}"
    --snapshot-digest "${GRPO_SNAPSHOT_DIGEST}"
  )
fi
admission_data_args=()
if [ -n "${GRPO_EXPECTED_FILTERED_TRAIN_ROWS}" ]; then
  admission_data_args+=(--expected-filtered-train-rows "${GRPO_EXPECTED_FILTERED_TRAIN_ROWS}")
fi
if [ -n "${GRPO_EXPECTED_DATALOADER_STEPS_PER_EPOCH}" ]; then
  admission_data_args+=(--expected-dataloader-steps-per-epoch "${GRPO_EXPECTED_DATALOADER_STEPS_PER_EPOCH}")
fi

python3 "${GRPO_ADMISSION_CHECKER}" \
  --task "${TASK}" \
  --pipeline "${PIPELINE}" \
  --model-path "${INIT_MODEL_PATH}" \
  --train-file "${TRAIN_FILE}" \
  --reward-path "${CUSTOM_REWARD_FN_PATH}" \
  --expected-model-sha256 "${GRPO_EXPECTED_INIT_MODEL_SHA256}" \
  --expected-train-sha256 "${GRPO_EXPECTED_TRAIN_SHA256}" \
  --expected-reward-sha256 "${GRPO_EXPECTED_REWARD_SHA256}" \
  --runtime-image-digest "${GRPO_RUNTIME_IMAGE_DIGEST}" \
  --expected-image-digest "${GRPO_EXPECTED_IMAGE_DIGEST}" \
  --actor-seed "${TRAINING_SEED}" \
  --rollout-seed "${ROLLOUT_SEED}" \
  --seed-replicate "${GRPO_SEED_REPLICATE}" \
  --data-seed "${DATA_SEED}" \
  --data-shuffle "${DATA_SHUFFLE}" \
  --train-prompt-bsz "${TRAIN_PROMPT_BSZ}" \
  --total-training-steps "${TOTAL_TRAINING_STEPS}" \
  --total-epochs "${TOTAL_EPOCHS}" \
  "${admission_data_args[@]}" \
  "${admission_source_args[@]}" \
  --receipt "${GRPO_ADMISSION_RECEIPT}"

if [ "${GRPO_PREFLIGHT_ONLY:-0}" = 1 ]; then
  echo "GRPO preflight passed: task=${TASK} pipeline=${PIPELINE}"
  exit 0
fi

common_overrides=(
  "hydra.run.dir=${LOG_DIR}/hydra/${SLURM_JOB_ID:-manual}"
  "data.shuffle=${DATA_SHUFFLE}"
  "+data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING}"
  "actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE}"
)
# shellcheck disable=SC1090
source "${COMMON_LAUNCHER}" "${common_overrides[@]}" "${macro_overrides[@]}" "$@"
