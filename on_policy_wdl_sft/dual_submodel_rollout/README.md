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

Default checkpoint retention keeps only the latest checkpoint plus the best
checkpoint selected by `val-core/HuggingFaceH4/MATH-500/acc/mean@1`.
The launch fails fast unless `BASE_CKPT_DIR` has at least
`MIN_FREE_GB_FOR_CKPT=160` GiB free by default; override this only for short
smoke runs or when checkpoint size is known.
