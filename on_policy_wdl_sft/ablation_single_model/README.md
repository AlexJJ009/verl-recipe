# Ablation: Single-Model WDL-SFT-IS (Series 2X)

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

`_common_ablation.sh` holds the shared env setup, checkpoint/resume logic, and
the Hydra launch command. Each `run_2*.sh` is a thin 10-line wrapper that
exports 4–5 knobs and sources the common launcher.

## What we can decompose from the result

- **Pair (2X-SFT vs 1X)**: isolates the "fusion" contribution with init held fixed.
- **Pair (2X-BASE vs 2X-SFT)**: isolates the "SFT init quality" contribution.
- **Pair (2X vs 2Z, same init)**: isolates the "loss design" contribution (wdl_sft_is − minirl).
- Triangle closes: joint 1X contribution = (2Z → 2X loss lift) + (2X-single → 1X-joint fusion lift).

## Usage

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
