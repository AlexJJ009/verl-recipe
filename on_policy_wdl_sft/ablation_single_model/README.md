# Ablation: Single-Model WDL-SFT / Group-Advantage IS

Scripts that test whether the `wdl_sft_is` loss works on its own — i.e. **without**
the joint-model + fused-logit rollout used in 1A/1B/1C. Every script here uses
a standard single-model setup (one Qwen3 backbone, rollout sampled from that
model's own distribution).

## Design

The only two variables that differ from the joint baseline are:

| Variable | Joint (1A/B/C) | Ablation (this folder) |
|---|---|---|
| Model architecture | `QwenJointForCausalLM` (model1+model2 fused logits) | `Qwen3ForCausalLM` (single backbone) |
| Rollout source | `P_mix = Softmax((1−λ)·z_weak + λ·z_strong)` | `P = Softmax(z)` of the single model |

Everything else is held fixed: data, prompts, batch size (64), rollouts per
prompt (8), mini-batch size (8), steps (300), optimizer, lr schedule, β,
IS-weights / binary-mask clip, eval cadence, seed, reward function. **We
control data budget (steps × batch × N = 153,600 responses/run) — NOT
GPU-hours.** Joint has ~2× the parameters of single, so wall-clock differs by
design; equalizing GPU-hour would mean different step counts and pollute the
comparison.

## Scripts

| Script | Init | Loss | β | lr | Compared to |
|---|---|---|---|---|---|
| `run_2a_base.sh` | Qwen3-4B-Base | `wdl_sft_is` | 0   | 5e-7 | 1A |
| `run_2a_sft.sh`  | Qwen3-4B-Base-SFT-stage-1 | `wdl_sft_is` | 0   | 5e-7 | 1A |
| `run_2b_base.sh` | Qwen3-4B-Base | `wdl_sft_is` | 0.1 | 5e-7 | 1B |
| `run_2b_sft.sh`  | Qwen3-4B-Base-SFT-stage-1 | `wdl_sft_is` | 0.1 | 5e-7 | 1B |
| `run_2c_base.sh` | Qwen3-4B-Base | `wdl_sft_is` | 0   | 1e-6 | 1C |
| `run_2c_sft.sh`  | Qwen3-4B-Base-SFT-stage-1 | `wdl_sft_is` | 0   | 1e-6 | 1C |
| `run_2z_base.sh` | Qwen3-4B-Base | `minirl` (pure RL) | —   | 5e-7 | reference baseline |
| `run_2z_sft.sh`  | Qwen3-4B-Base-SFT-stage-1 | `minirl` (pure RL) | —   | 5e-7 | reference baseline |
| `run_2g_base.sh` | Qwen3-4B-Base | `vanilla` (PPO-clip, GRPO) | — | 5e-7 | canonical GRPO baseline |
| `run_2g_sft.sh`  | Qwen3-4B-Base-SFT-stage-1 | `vanilla` (PPO-clip, GRPO) | — | 5e-7 | canonical GRPO baseline |
| `run_4b_math_base.sh` | Qwen3-4B-Base | `wdl_group_adv_is` | — | 5e-7 | 4A single-model Base ablation |
| `run_4c_math_sft.sh` | Qwen3-4B-Base-SFT-stage-1 | `wdl_group_adv_is` | — | 5e-7 | 4A single-model SFT ablation |

After the 2026-04-27 reward-label fix for `wdl_sft_is`, all `run_2a_*`,
`run_2b_*`, and `run_2c_*` wrappers default their `RUN_PREFIX` to
`WDL-SFT-...-LABELFIX`. This deliberately prevents new spec-correct runs from
auto-resuming the pre-fix checkpoints under the old prefixes. The `minirl`
(`2z`) and `vanilla` (`2g`) baselines are not affected by this bug and keep
their original prefixes.

`_common_ablation.sh` holds the shared env setup, checkpoint/resume logic, and
the Hydra launch command. Each `run_2*.sh` is a thin 10-line wrapper that
exports 4–5 knobs and sources the common launcher.

The 4B/4C MATH scripts reuse `/data-1/dataset/math/train_rl_format.parquet`,
`TOTAL_TRAINING_STEPS=115`, `TOTAL_EPOCHS=1`, validation `VAL_N=3`, and
`BEST_CKPT_METRIC_KEY=val-core/HuggingFaceH4/MATH-500/acc/mean@3`. They use
`LOSS_MODE=wdl_group_adv_is` with `ROLLOUT_IS=null` because the loss owns its
detached IS term internally.

**`run_2g_*` vs `run_2z_*`**: both are reference baselines from the same init,
but they exercise different loss families:

| Knob | `run_2z_*` (MiniRL) | `run_2g_*` (canonical GRPO) |
|---|---|---|
| `LOSS_MODE` | `minirl` | `vanilla` (standard PPO-clip) |
| `clip_ratio_low` / `high` | 0.2 / 0.27 (asymmetric) | 0.2 / 0.2 (symmetric) |
| `norm_adv_by_std_in_grpo` | False | **True** (standard DeepSeek-style GRPO) |
| `rollout_is` correction | token-level IS weights | token-level IS weights (same) |

Everything else — `adv_estimator=grpo`, `clip_ratio_c=10.0`, lr, batch, rollout
N, eval cadence — is identical. The 2G ↔ 2Z delta is exactly "which loss
family" on the same data; useful as a second reference floor to triangulate
against the MiniRL-centric 2X series.

## What we can decompose from the result

- **Pair (2X-SFT vs 1X)**: isolates the "fusion" contribution with init held fixed.
- **Pair (2X-BASE vs 2X-SFT)**: isolates the "SFT init quality" contribution.
- **Pair (2X vs 2Z, same init)**: isolates the "loss design" contribution (wdl_sft_is − minirl).
- Triangle closes: joint 1X contribution = (2Z → 2X loss lift) + (2X-single → 1X-joint fusion lift).

## Usage

### Local physical machine (default `/data-1/...` paths)

```bash
cd /data-1/verl07/verl
tmux new-session -s ablation_2a_base
bash recipe/on_policy_wdl_sft/ablation_single_model/run_2a_base.sh
# Ctrl-B D to detach; tmux attach -t ablation_2a_base to re-attach
```

Or override knobs on the command line:

```bash
LR=7e-7 TOTAL_TRAINING_STEPS=400 bash run_2a_sft.sh
```

### Meituan MLP (AFO) — see `meituan/` subfolder

The **same** `run_2X_*.sh` and `run_4{b,c}_*.sh` scripts run on Meituan MLP. The `meituan/` subfolder
provides a thin path-override layer that redirects all `/data-1/...` defaults
to dolphinfs paths:

```
meituan/
├── env.sh       # HF_HOME, TRAIN_FILE, BASE_CKPT_DIR, ... → /mnt/dolphinfs/.../lgx/...
├── jupyter.sh   # AFO worker entry: sources env.sh, dispatches on EXPERIMENT
└── run.hope     # AFO job config template
```

Workflow:
1. Copy the repo branch to `$LGX/verl08/verl-v0.7-feature-on-policy-wdl-sft/`
2. Upload `Qwen3-4B-Base-SFT-stage-1` (only needed for `-sft` variants) and the
   three parquet datasets (EnsembleLLM train + MATH-500 + AIME-2025 val) to
   the paths shown in `meituan/env.sh`
3. Copy `meituan/run.hope` to `$LGX/hope_dir/`, fill `afo.docker.image.name`,
   set `afo.app.env.EXPERIMENT=2z-base` (or `4b-math-base` / `4c-math-sft`)
4. `hope submit run.hope`

**Currently runnable on Meituan** (base model uploaded, SFT pending):
- `2z-base` — MiniRL baseline. **Recommended first smoke test.**
- `2g-base` — canonical GRPO baseline (vanilla PPO-clip).
- `2a-base`, `2b-base`, `2c-base` — WDL-SFT-IS variants on Base init
- `4b-math-base` — current `wdl_group_adv_is` single-model Base ablation on
  MATH train

**Blocked until SFT model uploaded**:
- `2a-sft`, `2b-sft`, `2c-sft`, `2z-sft`, `2g-sft` — all `-sft` variants
- `4c-math-sft` — current `wdl_group_adv_is` single-model SFT ablation on
  MATH train

`meituan/jupyter.sh` fails fast with a clear error if the required init model
or dataset isn't present at the expected dolphinfs path.

## Important caveats

1. **Base init cold-start**: `run_2*_base.sh` start from Qwen3-4B-Base with no
   prior instruction tuning. Early-step rollouts will be mostly incorrect,
   making the `C` set tiny. Expect slow first ~50 steps. If 300 steps isn't
   enough to see the trend, extend via `TOTAL_TRAINING_STEPS=500`.

2. **2B single-model risk**: v1-era single-model β>0 runs (EXP-12 M5 at β=0.1)
   diverged. v2's binary-mask lower-clip is the hypothesized countermeasure;
   2B tests whether that holds on a single model. Watch grad norm / ratio clip
   rate early.

3. **Checkpoint storage**: `/data-1` was recently near-full. Single-model
   checkpoints are ~45 GB each (vs joint's ~93 GB), 12 checkpoints per run at
   save_freq=25 over 300 steps → ~540 GB per run. Pre-check `df -h /data-1`
   before launching and clean old checkpoints if needed.

## Related docs

- Plan: `docs/joint_training/plans/active/ablation_single_model.md`
- v2 loss spec: `docs/joint_training/specs/wdl_sft_is.md`
- Joint runs (reference): `recipe/on_policy_wdl_sft/run_on_policy_wdl_sft_qwen3_4b_math_1{a,b,c}.sh`
