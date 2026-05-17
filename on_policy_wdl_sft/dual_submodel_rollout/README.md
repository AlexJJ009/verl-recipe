# Dual-Submodel Rollout WDL-SFT

This recipe enables the first dual-rollout implementation for the joint Qwen3
WDL-SFT experiments.

Default data flow:

1. Generate `rollout.n=8` responses from `sub_model_0`.
2. Generate `rollout.n=8` responses from `sub_model_1` using the same prompt batch.
3. Score both source batches for diagnostics.
4. Select only the `sub_model_1` batch for training.
5. Recompute `old_log_probs` under the fused joint policy.
6. Update the actor through fused joint logits with both submodels trainable.

Run locally inside the project Docker environment:

```bash
TOTAL_TRAINING_STEPS=1 \
TRAIN_PROMPT_BSZ=2 \
TRAIN_PROMPT_MINI_BSZ=1 \
ROLLOUT_AGENT_NUM_WORKERS=1 \
VAL_BEFORE_TRAIN=False \
TEST_FREQ=-1 \
SAVE_FREQ=1 \
bash recipe/on_policy_wdl_sft/dual_submodel_rollout/run_3a_model2_rollout_beta0.sh
```

`run_3b_model2_rollout_beta01.sh` is identical except for
`WDL_SFT_BETA=0.1`.
