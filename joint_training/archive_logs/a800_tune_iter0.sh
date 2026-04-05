#!/bin/bash
# ==============================================================================
# A800 Tuning — Iter 0: Baseline
# Purpose: Establish A800 baseline timing with current config (no changes)
# ==============================================================================

# Override schedule: 5 steps, no validation, no checkpointing
export TOTAL_TRAINING_STEPS=5
export TEST_FREQ=999
export SAVE_FREQ=999
export VAL_BEFORE_TRAIN=False

# Unique run name for log isolation
export RUN_PREFIX="a800-tune-iter0"
export WANDB_RUN_NAME="a800-tune-iter0-$(date +%s)"
export WANDB_MODE=offline

exec bash /workspace/verl/recipe/joint_training/run_baseline_minirl_qwen3_4b_math.sh
