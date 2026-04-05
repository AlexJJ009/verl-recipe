#!/bin/bash
# ==============================================================================
# A800 Tuning — Iter 1 (v3): Fix old_log_prob bottleneck only
#
# Previous attempts OOM'd because ppo_mini_batch_size=16 with USE_REMOVE_PADDING=False
# causes each GPU to process 16 padded sequences (16×4596=73K tokens → 76 GB → OOM).
#
# This iteration focuses ONLY on the old_log_prob bottleneck (31.4s, 16% of step):
#   - LOG_PROB_MAX_TOKEN_LEN_PER_GPU: 4596 → 18384 (4x more tokens per log-prob batch)
#     → ~9 FSDP rounds instead of ~36 → massive reduction on slow A800 NVLink
#   - LOG_PROB_MICRO_BATCH_SIZE: 4 → 16 (match higher token budget)
#   - ROLLOUT_GPU_MEMORY_UTILIZATION: 0.4 → 0.7 (more KV cache during rollout)
#
# Training params UNCHANGED (ppo_mini_batch_size=8, ACTOR_PPO_MAX_TOKEN_LEN=18384)
# to avoid OOM from padding overhead.
#
# Expected: old_log_prob 31.4s → ~8-12s, step time 196.7s → ~173-177s
# ==============================================================================

export TOTAL_TRAINING_STEPS=5
export TEST_FREQ=999
export SAVE_FREQ=999
export VAL_BEFORE_TRAIN=False

export RUN_PREFIX="a800-tune-iter1"
export WANDB_RUN_NAME="a800-tune-iter1-$(date +%s)"
export WANDB_MODE=offline

# Key changes (log-prob + rollout only, training unchanged)
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.7
export LOG_PROB_MICRO_BATCH_SIZE=16
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18384

exec bash /workspace/verl/recipe/joint_training/run_baseline_minirl_qwen3_4b_math.sh \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=18384 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=18384
