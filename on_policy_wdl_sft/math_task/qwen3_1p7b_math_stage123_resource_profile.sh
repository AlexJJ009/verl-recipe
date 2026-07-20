#!/usr/bin/env bash
set -euo pipefail

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-500}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-4596}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-4596}
export ACTOR_PPO_MAX_TOKEN_LEN=${ACTOR_PPO_MAX_TOKEN_LEN:-4596}
export GENERATION_MICRO_BATCH_SIZE=${GENERATION_MICRO_BATCH_SIZE:-32}
export LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-8}
export REF_LOG_PROB_MICRO_BATCH_SIZE=${REF_LOG_PROB_MICRO_BATCH_SIZE:-1}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60}
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
export TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-64}
export ROLLOUT_N=${ROLLOUT_N:-8}
export TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ:-$((TRAIN_PROMPT_BSZ * ROLLOUT_N))}
export TEST_FREQ=${TEST_FREQ:-5}
export SAVE_FREQ=${SAVE_FREQ:-5}
export VAL_N=${VAL_N:-3}
export VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.2}
export VAL_TOP_P=${VAL_TOP_P:-0.95}
export VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
export DATA_SEED=${DATA_SEED:-20260719}
export DATA_SHUFFLE=${DATA_SHUFFLE:-False}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
export FUSION_LAMBDA=${FUSION_LAMBDA:-0.8}
export MATH7_VAL_FILES=${MATH7_VAL_FILES:-"['/data-1/dataset/AIME-2025/aime-2025_with_system_prompt.parquet','/data-1/dataset/MATH-500/math500-test_with_system_prompt.parquet','/data-1/dataset/AMC23/amc23-test_with_system_prompt.parquet','/data-1/dataset/AQUA/aqua-test_with_system_prompt.parquet','/data-1/dataset/gsm8k/gsm8k-test_with_system_prompt.parquet','/data-1/dataset/MAWPS/mawps-test_with_system_prompt.parquet','/data-1/dataset/SVAMP/svamp-test_with_system_prompt.parquet']"}
export TEST_FILES="$MATH7_VAL_FILES"
export MATH7_MACRO_SOURCES=${MATH7_MACRO_SOURCES:-"[aime25,HuggingFaceH4/MATH-500,zwhe99/amc23,deepmind/aqua_rat,openai/gsm8k,mwpt5/MAWPS,ChilleD/SVAMP]"}
export BEST_CKPT_METRIC_KEY=${BEST_CKPT_METRIC_KEY:-val-core/math7_macro/acc/mean@3}
export WANDB_PROJECT=${WANDB_PROJECT:-OnPolicyWDLSFT-Math-1P7B}
export WANDB_MODE=${WANDB_MODE:-offline}

math_stage123_macro_overrides() {
    printf '%s\n' \
        "+trainer.validation_macro_average_sources=${MATH7_MACRO_SOURCES}" \
        "+trainer.validation_macro_average_name=math7_macro" \
        "+trainer.validation_macro_average_metric=acc/mean@3"
}
