# Experiment Index — On-Policy WDL-SFT

This table tracks all training experiments for the On-Policy WDL-SFT branch, their logs, checkpoints, and evaluation results.

Branch: `feature/on-policy-wdl-sft`

> **Note on numbering**: EXP-01 through EXP-11 live in `recipe/joint_training/EXPERIMENT_INDEX.md` (joint-training parent branch). EXP-12 through EXP-15 are the On-Policy WDL-SFT runs, ordered chronologically. Short smoke-test / launch-failure runs (the un-suffixed initial run, M5.7 dataset-missing failure, M5.8 killed at step 3) are intentionally omitted — their logs remain on disk under `recipe/on_policy_wdl_sft/` but they produced no meaningful training signal.

> **On "eval" semantics**: For EXP-12 through EXP-14, "Online Val Progression" refers to the trainer's `val-core/...acc/mean@1` metrics logged every validation step during training (MATH-500 500 samples, AIME-2025 26 samples, n=1). These are joint-model fused-rollout results. Offline vLLM evaluation (extracting model1/model2 and re-running with n=3 across 7 benchmarks) has only been performed for EXP-15 step 125 so far — see EVAL-10 / EVAL-11 in `INFERENCE_RESULTS.md`.

---

## EXP-12: WDL-SFT-M5 — First Long Run (lr=1e-6, β=0.1, bidirectional)

| Field | Value |
|---|---|
| **Script** | `run_on_policy_wdl_sft_qwen3_4b_math.sh` with RUN_PREFIX=`WDL-SFT-Qwen3-4B-MATH-M5` |
| **Goal** | First real training run: bidirectional WDL-SFT at lr=1e-6, β=0.1, full 1745-step horizon, to see whether reverse SFT improves over forward-only. |
| **Algorithm** | WDL-SFT bidirectional, β=0.1, loss_mode=wdl_sft, seq-mean-token-sum |
| **Model** | QwenJoint-4B (weak=Qwen3-4B-Base, strong=Qwen3-4B-Base-SFT-stage-1, λ=0.5) |
| **Dataset** | EnsembleLLM MATH / MATH-500 + AIME-2025 (val) |
| **Key Params** | lr=1e-6, warmup=5, batch=64, n_resp=8, max_resp_len=4096, grad_clip=500.0, weight_decay=0.1, total_training_steps=1745, val_freq=100, 8 GPUs |
| **Logs** | `WDL-SFT-Qwen3-4B-MATH-M5_1775597477.log` (ran to step ~1043, killed before reaching 1745) |
| **Validation Dir** | `validation/WDL-SFT-Qwen3-4B-MATH-M5_1775597477/` (step 0, 100, 200, …, 1000) |
| **Checkpoint Dir** | Not retained on `/data-2` (discarded after divergence was confirmed) |
| **Online Val Progression (MATH-500 / AIME-2025)** | step 0: 58.47% / 15.38% → step 100: **66.94%** / 15.38% → step 200: 65.73% / 15.38% → step 300: 48.79% / 7.69% → step 400: 63.91% / 11.54% → step 500: 63.31% / 11.54% → step 600: 43.15% / 0.00% → step 700: 55.65% / 7.69% → step 800: 60.08% / 3.85% → step 900: 51.61% / 3.85% → step 1000: 42.74% / 7.69% |
| **Inference** | None |
| **W&B** | Project: `OnPolicyWDLSFT`, Run: `WDL-SFT-Qwen3-4B-MATH-M5_1775597477` |
| **Status** | **Diverged** — peaked at step 100 (66.94% MATH-500), then bidirectional reverse SFT at lr=1e-6 destabilized the model. AIME-2025 collapsed to 0% at step 600. This run motivated the switch to forward-only (β=0) and a smaller lr (5e-7) in M5.5. |

---

## EXP-13: WDL-SFT-M5.5 — Forward-Only Baseline (lr=5e-7, β=0, 300 steps) ★

| Field | Value |
|---|---|
| **Script** | `run_on_policy_wdl_sft_qwen3_4b_math_m5_5.sh` |
| **Goal** | Establish a stable baseline by (a) disabling reverse SFT (β=0) and (b) halving the learning rate to 5e-7. Short 300-step horizon to finish quickly and confirm whether forward-only WDL-SFT converges cleanly. **This is the reference baseline for all subsequent LR-search and ablation work.** |
| **Algorithm** | WDL-SFT forward-only (β=0), loss_mode=wdl_sft, seq-mean-token-sum |
| **Model** | QwenJoint-4B (λ=0.5) |
| **Dataset** | EnsembleLLM MATH / MATH-500 + AIME-2025 (val) |
| **Key Params** | lr=5e-7, warmup=5, batch=64, n_resp=8, max_resp_len=4096, grad_clip=500.0, weight_decay=0.1, total_training_steps=300, val_freq=25, 8 GPUs |
| **Logs** | `WDL-SFT-Qwen3-4B-MATH-M5-5_1775980322.log` (steps 0 → 300 complete) |
| **Validation Dir** | `validation/WDL-SFT-Qwen3-4B-MATH-M5-5_1775980322/` (13 jsonls: step 0, 25, 50, …, 300) |
| **Checkpoint Dir** | `/data-2/checkpoints/WDL-SFT-Qwen3-4B-MATH-M5-5_1775980322/` (12 checkpoints: steps 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300) — migrated from `/data-1` |
| **Best Checkpoint** | **step 300** — online MATH-500 = **67.94%**, AIME-2025 = 26.92% |
| **Online Val Progression (MATH-500 / AIME-2025)** | step 0: 59.68% / 11.54% → step 25: 63.51% / 15.38% → step 50: 63.91% / 11.54% → step 75: 64.11% / 15.38% → step 100: 63.51% / 15.38% → step 125: 65.32% / 19.23% → step 150: 65.12% / 19.23% → step 175: 67.74% / 15.38% → step 200: 65.73% / 15.38% → step 225: 67.54% / 15.38% → step 250: 66.94% / **26.92%** → step 275: 67.74% / 19.23% → **step 300: 67.94% / 26.92%** |
| **Inference** | Pending — offline vLLM eval on extracted model1/model2 not yet run |
| **W&B** | Project: `OnPolicyWDLSFT`, Run: `WDL-SFT-Qwen3-4B-MATH-M5-5_1775980322` |
| **Status** | **Complete & stable** ✓ — monotonically improved from 59.68% → 67.94% on MATH-500, AIME-2025 climbed to 26.92%. First run to confirm that forward-only (β=0) WDL-SFT at lr=5e-7 is stable. Checkpoints migrated to `/data-2` to free `/data-1` for the LR-search grid. |
| **Context** | Referenced in `docs/joint_training/plans/active/lr_search.md` as the baseline lr=5e-7 anchor. |

---

## EXP-14: WDL-SFT-M5.6 — Reverse SFT Re-Test (lr=5e-7, β=0.1)

| Field | Value |
|---|---|
| **Script** | `run_on_policy_wdl_sft_qwen3_4b_math_m5_6.sh` |
| **Goal** | After M5.5 confirmed forward-only stability, re-enable reverse SFT (β=0.1) at the lower lr=5e-7 to test whether the M5 instability was caused by the lr rather than by the reverse term itself. |
| **Algorithm** | WDL-SFT bidirectional, β=0.1, loss_mode=wdl_sft, seq-mean-token-sum |
| **Model** | QwenJoint-4B (λ=0.5) |
| **Dataset** | EnsembleLLM MATH / MATH-500 + AIME-2025 (val) |
| **Key Params** | lr=5e-7, warmup=5, batch=64, n_resp=8, max_resp_len=4096, grad_clip=500.0, weight_decay=0.1, total_training_steps=300, val_freq=25, 8 GPUs |
| **Logs** | `WDL-SFT-Qwen3-4B-MATH-M5-6_1776095760.log` (initial; steps 0 → 236, crashed) |
| | `WDL-SFT-Qwen3-4B-MATH-M5-6_1776095760_resumed_1776202941.log` (resumed from step 225, continued past 300 — total cap raised to 1745 on resume, ran to step ~458 before manual stop) |
| **Validation Dir** | `validation/WDL-SFT-Qwen3-4B-MATH-M5-6_1776095760/` (12 jsonls: step 0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 300, 400) |
| **Checkpoint Dir** | `/data-2/checkpoints/WDL-SFT-Qwen3-4B-MATH-M5-6_1776095760/` (11 checkpoints: steps 25, 50, 75, 100, 125, 150, 175, 200, 225, 300, 400) |
| **Best Checkpoint** | **step 300** — online MATH-500 = **68.15%**, AIME-2025 = 19.23% |
| **Online Val Progression (MATH-500 / AIME-2025)** | (initial run, val every 25) step 0: 63.10% / 15.38% → step 25: 63.10% / 15.38% → step 50: 61.69% / 23.08% → step 75: 64.31% / **26.92%** → step 100: 66.13% / 19.23% → step 125: 65.52% / 19.23% → step 150: 65.93% / **26.92%** → step 175: 65.12% / 15.38% → step 200: 64.92% / 15.38% → step 225: 64.52% / 23.08% → (initial crash near step 236, resumed) → step 225 (revalidate): 67.34% / 19.23% → **step 300: 68.15% / 19.23%** → step 400: 67.54% / 23.08% |
| **Inference** | Pending |
| **W&B** | Project: `OnPolicyWDLSFT`, Run: `WDL-SFT-Qwen3-4B-MATH-M5-6_1776095760` |
| **Status** | **Completed with one mid-run crash (step 236), successfully resumed to step ~458.** Contrary to earlier characterizations, M5.6 at lr=5e-7 with β=0.1 did **not** diverge — it matched or slightly exceeded M5.5 (peak 68.15% MATH-500 vs 67.94%). The original "reverse SFT unstable" conclusion was driven by M5 (lr=1e-6), not by M5.6. |
| **Context** | The project currently treats reverse SFT as "abandoned" (see CLAUDE.md), but M5.6 results suggest that claim deserves revisiting — divergence in M5 may have been an lr artifact. |

---

## EXP-15: WDL-SFT-LR3 — On-Policy WDL-SFT, lr=1e-6 (forward-only)

| Field | Value |
|---|---|
| **Script** | `run_on_policy_wdl_sft_qwen3_4b_math_lr3.sh` |
| **Goal** | LR search point: test lr=1e-6 (doubled from M5.5 baseline lr=5e-7) with forward-only On-Policy WDL-SFT |
| **Algorithm** | On-Policy WDL-SFT, forward-only (β=0), loss_mode=wdl_sft, seq-mean-token-sum |
| **Model** | QwenJoint-4B (weak=Qwen3-4B-Base, strong=Qwen3-4B-Base-SFT-stage-1, λ=0.5) |
| **Dataset** | EnsembleLLM MATH (`train_rl_format.parquet`, ~111K entries) / 7 benchmarks with system prompt (offline test) |
| **Key Params** | lr=1e-6, warmup=5, batch=64 prompts, n_resp=8, max_resp_len=4096, grad_clip=500.0, weight_decay=0.1, 8 GPUs |
| **Logs** | `WDL-SFT-Qwen3-4B-MATH-LR3_1776359574.log` (steps 1–125, initial run) |
| | `WDL-SFT-Qwen3-4B-MATH-LR3_1776359574_resumed_1776445423.log` (first resume) |
| | `WDL-SFT-Qwen3-4B-MATH-LR3_1776359574_resumed_1776445477.log` (second resume, steps to 274, killed) |
| **Checkpoint Dir** | `/data-1/checkpoints/WDL-SFT-Qwen3-4B-MATH-LR3_1776359574/` (steps 25, 50, 75, 100, 125, 150, 175, 200, 225, 250) |
| **Best Checkpoint** | **step 125** — online MATH-500 acc/mean@1 = **68.15%**, AIME-2025 = 19.23% |
| **Online Val Progression** | step 25: 63.3% / step 50: 65.9% / step 75: 65.7% / step 100: 67.3% / **step 125: 68.2%** / step 150: 67.9% / step 175: 60.5% / step 200: 66.5% / step 225: 62.5% / step 250: 58.9% |
| **Inference** | EVAL-10 in `INFERENCE_RESULTS.md` (step 125, model2, 7 benchmarks, n=3) |
| | EVAL-11 in `INFERENCE_RESULTS.md` (step 125, model1, 7 benchmarks, n=3) |
| **W&B** | Project: `OnPolicyWDLSFT`, Run: `WDL-SFT-Qwen3-4B-MATH-LR3_1776359574` (offline, not yet synced) |
| **Status** | **Stopped early** — killed at step 274. Training diverged after step 125: MATH-500 online dropped from 68.15% (step 125) to 58.87% (step 250). lr=1e-6 ramps up faster than M5.5 (lr=5e-7) but is unstable in later training. |
| **Context** | Part of LR search grid; see `docs/joint_training/plans/active/lr_search.md`. M5.5 baseline (lr=5e-7) achieved stable 67.94% at step 300. |

---

## Cross-Experiment Comparison (On-Policy WDL-SFT — Online Val, MATH-500 / AIME-2025)

| Experiment | lr | β | Steps | Best MATH-500 (step) | Best AIME-2025 (step) | Status |
|---|---|---|---|---|---|---|
| EXP-12 (M5) | 1e-6 | 0.1 | ~1043 | 66.94% (step 100) | 15.38% (steps 0–200) | Diverged |
| **EXP-13 (M5.5)** | **5e-7** | **0** | **300 complete** | **67.94% (step 300)** | **26.92% (steps 250, 300)** | **Stable baseline ★** |
| EXP-14 (M5.6) | 5e-7 | 0.1 | ~458 (1 resume) | 68.15% (step 300) | 26.92% (steps 75, 150) | Stable (peak ≥ M5.5) |
| EXP-15 (LR3) | 1e-6 | 0 | ~274 | 68.15% (step 125) | 23.08% (step 125) | Peaked early, then diverged |

**Observations**:
- Forward-only (β=0) at lr=5e-7 (EXP-13) is the only **monotonically improving** run through 300 steps.
- Doubling lr to 1e-6 (EXP-15) ramps faster (hits 68.15% by step 125) but destabilizes after step 150 — same directional effect as EXP-12 (M5 bidirectional at lr=1e-6, peak at step 100).
- EXP-14 (β=0.1 at lr=5e-7) matched EXP-13's peak, suggesting the earlier blanket "reverse SFT unstable" conclusion was really an artifact of lr=1e-6, not of the reverse term itself.

---

## Checkpoint Inventory

| Checkpoint Path | Experiment | Steps Saved | Status |
|---|---|---|---|
| `/data-2/checkpoints/WDL-SFT-Qwen3-4B-MATH-M5-5_1775980322/` | EXP-13 (M5.5) | 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300 | Complete ✓ (migrated from `/data-1`) |
| `/data-2/checkpoints/WDL-SFT-Qwen3-4B-MATH-M5-6_1776095760/` | EXP-14 (M5.6) | 25, 50, 75, 100, 125, 150, 175, 200, 225, 300, 400 | Complete (migrated from `/data-1`) |
| `/data-1/checkpoints/WDL-SFT-Qwen3-4B-MATH-LR3_1776359574/` | EXP-15 (LR3) | 25, 50, 75, 100, 125, 150, 175, 200, 225, 250 | Stopped early (best: step 125) |

EXP-12 (M5) did not retain its checkpoint directory — discarded after divergence was confirmed.

---

## Extracted Weights

| Path | Source | Step | Sub-Model | Notes |
|---|---|---|---|---|
| `/data-1/eval_results/wdl-sft-lr3-step125_merged_joint/` | EXP-15 | 125 | Joint (both sub-models) | Merged from FSDP shards |
| `/data-1/eval_results/wdl-sft-lr3-step125_model2/` | EXP-15 | 125 | model2 (strong/trainable, index=1) | Extracted from joint |
| `/data-1/eval_results/wdl-sft-lr3-step125_model1/` | EXP-15 | 125 | model1 (weak/anchor, index=0) | Extracted from joint |
