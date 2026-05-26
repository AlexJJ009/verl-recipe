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

Run the 4A learning-signal smoke in tmux. This smoke intentionally keeps the
real generation settings from `run_4a_model2_group_adv_is.sh`; only the number
of update steps and operational paths/cadence should be shortened.

```bash
tmux new-session -s dual_model2_group_adv_is_learning_smoke
TOTAL_TRAINING_STEPS=3 \
VAL_BEFORE_TRAIN=False \
TEST_FREQ=-1 \
SAVE_FREQ=1 \
MIN_FREE_GB_FOR_CKPT=1 \
bash recipe/on_policy_wdl_sft/dual_submodel_rollout/run_4a_model2_group_adv_is.sh
```

Do not lower `MAX_RESPONSE_LENGTH` or `N_RESP_PER_PROMPT` for an algorithm
acceptance smoke. This method's real defaults are `MAX_RESPONSE_LENGTH=4096`
and `N_RESP_PER_PROMPT=8`. A short smoke such as `MAX_RESPONSE_LENGTH=256`
usually truncates every math response before EOS; the reward function marks
truncated responses as `-1`, all prompt groups become all-incorrect, group
advantages become zero, and `actor/grad_norm` can be exactly `0.0`. Such a run
is useful only as a plumbing check.

The learning-signal smoke should be considered valid only if the metrics show
non-degenerate reward groups: at least one logged step with
`mixed_group_fraction > 0` or `all_correct_fallback_group_fraction > 0`, plus
nonzero advantage/gradient evidence such as `critic/advantages/max > 0` and
finite `actor/grad_norm > 0`.

`run_4a_model2_group_adv_is.sh` is the revised algorithm entrypoint:

1. Generate responses only from `sub_model_1`.
2. Preserve the model2 rollout log-probs as `log_pi_model2_rollout`.
3. Recompute `old_log_probs` under the fused joint policy.
4. Train with `loss_mode=dual_model2_group_adv_is`.
5. Use group advantages with the all-correct positive fallback
   `GAMMA_POS_SFT=1.0`, detached TIS weight capped by `TIS_THRESHOLD=5.0`,
   and binary staleness masks from current fused / old fused.

The 4A default training set is
`/data-1/dataset/math/train_rl_format.parquet`. The 4A production launcher
defaults to one filtered MATH epoch: `TOTAL_TRAINING_STEPS=115` and
`TOTAL_EPOCHS=1`.

Default validation uses `actor_rollout_ref.rollout.val_kwargs.n=3`.
Default checkpoint retention keeps only the latest checkpoint plus the best
checkpoint selected by `val-core/HuggingFaceH4/MATH-500/acc/mean@3`.
The launch fails fast unless `BASE_CKPT_DIR` has at least
`MIN_FREE_GB_FOR_CKPT=160` GiB free by default; override this only for short
smoke runs or when checkpoint size is known.

## Meituan AFO

4A is reachable through the unified Meituan entry:

```bash
cd platform/hope_on_policy_wdl_sft
./submit_batch.sh --dry-run 4a
```

The platform dispatcher routes `EXPERIMENT=4a` to
`recipe/on_policy_wdl_sft/dual_submodel_rollout/meituan/jupyter.sh`, which
sources `meituan/env.sh` and then launches `run_4a_model2_group_adv_is.sh`.

On Meituan, `TRAIN_FILE` defaults to
`$LGX/verl-exp/data/math/train_rl_format.parquet`, validation data defaults to
`$LGX/verl-exp/data/MATH-500` and `$LGX/verl-exp/data/AIME-2025`, and logs /
checkpoints / wandb land under `$LGX/verl-exp/`.

## Queue Monitor

`monitor_4abc_math_queue.sh` is a thin wrapper around the project-level generic
monitor `scripts/training_queue_monitor.sh`. It only defines the 4A -> 4B ->
4C queue and the local resource gates.
