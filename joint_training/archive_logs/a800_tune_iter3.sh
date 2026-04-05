#!/bin/bash
# ==============================================================================
# A800 Tuning — Iter 3: torch.compile for actor training
#
# Iter 2 showed HSDP (fsdp_size=4) adds memory but doesn't help speed.
# Revert to fsdp_size=-1 and try torch.compile for training kernel fusion.
#
# Changes from Iter 1 (best so far):
#   use_torch_compile: False → True (fuse training kernels, reduce overhead)
#
# Keep Iter 1 improvements:
#   LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18384
#   LOG_PROB_MICRO_BATCH_SIZE=16
#   ROLLOUT_GPU_MEMORY_UTILIZATION=0.7
#
# Risk: torch.compile step 1 will be slow due to compilation.
#   Steps 2-5 are the true measurement.
# ==============================================================================

export TOTAL_TRAINING_STEPS=5
export TEST_FREQ=999
export SAVE_FREQ=999
export VAL_BEFORE_TRAIN=False

export RUN_PREFIX="a800-tune-iter3"
export WANDB_RUN_NAME="a800-tune-iter3-$(date +%s)"
export WANDB_MODE=offline

# Iter 1 improvements (keep)
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.7
export LOG_PROB_MICRO_BATCH_SIZE=16
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18384

exec bash /workspace/verl/recipe/joint_training/run_baseline_minirl_qwen3_4b_math.sh \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=18384 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=18384 \
  actor_rollout_ref.actor.use_torch_compile=True
