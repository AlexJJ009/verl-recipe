# Staged v1 On-Policy SFT -> WDL-SFT

The current executable scope is the boxed-prompt rerun for two matched chains:

1. `run_s1_base_sft.sh`: common single-model On-Policy SFT launcher from Qwen3-4B-Base.
2. `run_s1_beta_0.sh` and `run_s1_beta_01.sh`: Stage 1 boxed-prompt wrappers for `WDL_SFT_BETA=0.0` and `0.1`.
3. `run_stage1_beta_search_queue.sh`: local sequential queue/monitor for the two boxed Stage 1 runs.
4. `run_boxed_matched_chain_queue.sh`: current primary local queue; runs Stage 1, fixed Model2 merge, then matched Stage 2 for beta `0.0`, followed by the same chain for beta `0.1`.
5. `monitor_boxed_matched_chain_notify.sh`: optional external WxPusher monitor for the full chain.

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

Current boxed Stage 1 queue:

```text
0.0, 0.1
```

## Stage 2 Fast Validation

Current Stage 2 development target:

```text
docs/joint_training/plans/active/stage2_model2_rollout_fused_loss_fast_validation.md
```

Stage 2 fast validation uses Stage 1 beta `0.0` and `0.1` best checkpoints as
Model 2, original Qwen3-4B-Base as Model 1, Model2-only rollout, fused joint
WDL-SFT loss, and updates both submodels. It only runs matched beta chains:

```text
Stage1 beta=0.0 -> Stage2 beta=0.0
Stage1 beta=0.1 -> Stage2 beta=0.1
```

The local-only queue is:

```text
recipe/on_policy_wdl_sft/staged_v1/run_boxed_matched_chain_queue.sh
```

The fixed Model2 merge outputs are:

```text
/data-1/model_weights/staged_v1/boxed_matched/model2-from-s1-boxed-beta0-best
/data-1/model_weights/staged_v1/boxed_matched/model2-from-s1-boxed-beta01-best
```

Run the chain queue and monitor in separate tmux sessions:

```bash
tmux new-session -s staged_v1_boxed_matched_chain_queue
bash recipe/on_policy_wdl_sft/staged_v1/run_boxed_matched_chain_queue.sh

tmux new-session -s staged_v1_boxed_matched_chain_monitor
bash recipe/on_policy_wdl_sft/staged_v1/monitor_boxed_matched_chain_notify.sh
```

The Stage 2 data shard must be generated and verified before launch:

```bash
python3 recipe/on_policy_wdl_sft/staged_v1/create_stage2_nonoverlap_shard.py
python3 recipe/on_policy_wdl_sft/staged_v1/create_stage2_nonoverlap_shard.py --verify-only
```

Legacy `run_s2_beta_*.sh` wrappers are not the fast-validation entry points.
Use `run_s2_from_s1_beta*_beta*.sh` for this plan.
