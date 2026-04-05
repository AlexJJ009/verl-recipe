#!/bin/bash
# ==============================================================================
# A800 Tuning — Iter 2: HSDP + faster prefill
#
# Iter 1 result: 190.0s/step (-3.4% from baseline 196.7s)
#   - Actor update still 112s (59%) — dominated by FSDP communication + padding
#   - flash_attn unavailable → cannot enable USE_REMOVE_PADDING
#
# This iteration tries:
#   1. HSDP (fsdp_size=4): 2 data-parallel groups of 4 GPUs each
#      → All-gather within 4 GPUs (not 8) → ~14% less communication
#      → Extra gradient all-reduce between groups is lightweight
#      → Memory: +4 GB/GPU for larger shards → 48 GB total (still well within 72 GB)
#   2. ROLLOUT_MAX_NUM_BATCHED_TOKENS: 4596 → 16384 (faster vLLM prefill scheduling)
#   3. Keep Iter 1 log-prob improvements (LOG_PROB_MAX_TOKEN_LEN=18384)
#
# Expected: actor update 112s → ~104s, gen 44s → ~42s, step time 190s → ~175s
# ==============================================================================

export TOTAL_TRAINING_STEPS=5
export TEST_FREQ=999
export SAVE_FREQ=999
export VAL_BEFORE_TRAIN=False

export RUN_PREFIX="a800-tune-iter2"
export WANDB_RUN_NAME="a800-tune-iter2-$(date +%s)"
export WANDB_MODE=offline

# Iter 1 improvements (keep)
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.7
export LOG_PROB_MICRO_BATCH_SIZE=16
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18384

# Iter 2 new: faster prefill
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=16384

exec bash /workspace/verl/recipe/joint_training/run_baseline_minirl_qwen3_4b_math.sh \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=18384 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=18384 \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=4
