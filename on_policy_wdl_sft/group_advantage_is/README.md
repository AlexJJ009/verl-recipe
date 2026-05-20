# WDL Group-Advantage IS

This recipe family launches the joint-model `wdl_group_adv_is` method:

- GRPO outcome advantages with `algorithm.norm_adv_by_std_in_grpo=false`
- all-correct positive-SFT fallback with coefficient `1.0`
- detached old/current token IS inside the policy loss
- no `rollout_is_weights`, no KL reward penalty, no actor KL loss, no beta
- `seq-mean-token-sum` aggregation

Local run:

```bash
bash recipe/on_policy_wdl_sft/group_advantage_is/run_1a_group_adv_is.sh
```

Small local smoke:

```bash
TOTAL_TRAINING_STEPS=1 TRAIN_PROMPT_BSZ=2 TRAIN_PROMPT_MINI_BSZ=1 \
PPO_EPOCHS=1 ROLLOUT_AGENT_NUM_WORKERS=1 VAL_BEFORE_TRAIN=False \
TEST_FREQ=-1 SAVE_FREQ=1 \
bash recipe/on_policy_wdl_sft/group_advantage_is/run_1a_group_adv_is.sh
```

Meituan/AFO runs use the layered path under
`platform/hope_group_advantage_is/` and `meituan/`.
