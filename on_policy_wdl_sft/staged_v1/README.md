# Staged v1 On-Policy SFT -> WDL-SFT

The current executable scope is Stage 1 only:

1. `run_s1_base_sft.sh`: common single-model On-Policy SFT launcher from Qwen3-4B-Base.
2. `run_s1_beta_*.sh`: Stage 1 beta grid wrappers for `WDL_SFT_BETA=0.0..1.0`.
3. `run_stage1_beta_search_queue.sh`: local sequential queue/monitor for the Stage 1 beta grid.

The scripts intentionally reuse the existing `loss_mode=wdl_sft` implementation. Stage 1 keeps `joint_training=False`, `rollout_is=null`, `rollout_rs=null`, and KL disabled.

Default validation/checkpoint cadence is dense (`TEST_FREQ=5`, `SAVE_FREQ=5`, `VAL_N=3`) while checkpoint retention keeps only the latest full checkpoint and the best checkpoint. `VAL_BEFORE_TRAIN=False` is the staged v1 default, because the Base model should not repeatedly run full step-0 validation. Training data order is fixed with `DATA_SEED=20260528` so short runs consume the same prompt subset across beta variants unless explicitly overridden.

W&B defaults:

```text
WANDB_PROJECT=OnPolicySFT-Then-WDLSFT-StagedV1
WANDB_MODE=offline
```

After a run completes or is intentionally stopped, sync the offline W&B run:

```bash
WANDB_SYNC_DIR=/path/to/wandb/offline-run-* bash recipe/on_policy_wdl_sft/staged_v1/sync_wandb_offline.sh
```

Meituan launch path:

```text
platform/hope_staged_v1 -> recipe/on_policy_wdl_sft/staged_v1/meituan -> run_*.sh
```

Stage 1 beta grid:

```text
0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
```

## Stage 2 Fast Validation

Current Stage 2 development target:

```text
docs/joint_training/plans/active/stage2_model2_rollout_fused_loss_fast_validation.md
```

Stage 2 fast validation uses Stage 1 beta `0.0` and `0.1` best checkpoints as
Model 2, original Qwen3-4B-Base as Model 1, Model2-only rollout, fused joint
WDL-SFT loss, and updates both submodels. The local-only queue is:

```text
recipe/on_policy_wdl_sft/staged_v1/run_stage2_fast_validation_queue.sh
```

The Stage 2 data shard must be generated and verified before launch:

```bash
python3 recipe/on_policy_wdl_sft/staged_v1/create_stage2_nonoverlap_shard.py
python3 recipe/on_policy_wdl_sft/staged_v1/create_stage2_nonoverlap_shard.py --verify-only
```

Legacy `run_s2_beta_*.sh` wrappers are not the fast-validation entry points.
Use `run_s2_from_s1_beta*_beta*.sh` for this plan.
