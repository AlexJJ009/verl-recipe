# Experiment Index — On-Policy WDL-SFT

This table tracks all training experiments for the On-Policy WDL-SFT branch, their logs, checkpoints, and evaluation results.

Branch: `feature/on-policy-wdl-sft`

> **Note on numbering**: EXP-01 through EXP-11 live in `recipe/joint_training/EXPERIMENT_INDEX.md` (joint-training parent branch). EXP-12 through EXP-18 are the joint-model On-Policy WDL-SFT runs, ordered chronologically. EXP-12–15 used v1 loss (`loss_mode=wdl_sft`); EXP-16–18 are v2 runs (`loss_mode=wdl_sft_is`, IS/clip-corrected). **The single-model ablation series (2X, plan: `docs/joint_training/plans/active/ablation_single_model.md`) uses a separate prefix `ABL-MINIRL-NN`** — same data budget / config as paired joint runs, standard single Qwen3 backbone instead of `QwenJointForCausalLM`. The prefix switch (rather than continuing EXP-N) is deliberate: it keeps the joint-family (EXP-N) and the ablation-family (ABL-MINIRL-N) visually distinct when scanning registries. ABL-MINIRL-01 = 2Z-SFT (MiniRL from SFT-stage-1 init), ABL-MINIRL-02 = 2A-SFT, and ABL-MINIRL-03/04/05/06/07 = the 2Z/2G/2A/2C/2B-BASE Meituan-AFO batch launched 2026-04-23 (ordered chronologically by launch timestamp). Short smoke-test / launch-failure runs (the un-suffixed initial run, M5.7 dataset-missing failure, M5.8 killed at step 3, the 1A first-launch attempt at `_1776591102`, the 1B first-launch attempt at `_1776683653` killed at step 26 before the val-path semantics were re-confirmed, and the 2Z-BASE `_1776928234` first attempt killed at init by a `trl` import incompatibility before any step trained) are intentionally omitted — their logs remain on disk or have been cleaned.

> **On "eval" semantics**: "Online Val Progression" refers to the trainer's `val-core/...acc/mean@1` metrics logged every validation step during training (MATH-500 500 samples, AIME-2025 26 samples, n=1). For **v2 runs (EXP-16 onward)**, `ray_trainer._validate()` invokes `checkpoint_manager.update_weights(eval_only=True)` → `extract_sub_model_weights(sub_model_index=1)`, so the online val is **model2-only** (not fused). Verified in 1A/1B logs via `[WDL-SFT VERIFY] extracting model2-only weights`. For v1 runs (EXP-12–15) the online val reported in this index is joint-fused. Offline vLLM evaluation (n=3 across 7 benchmarks on extracted model1/model2) has been performed for EXP-15 step 125 (EVAL-10 / EVAL-11), EXP-13 step 300 (EVAL-12 / EVAL-13), EXP-14 step 300 (EVAL-14 / EVAL-15), EXP-16 step 225 model2 (**EVAL-20**, MATH-500 mean@3 = 83.1% — v2 breaks v1 ~79% ceiling), EXP-17 step 275 / step 300 (EVAL-16/17 and EVAL-18/19), and **EXP-18 step 150 / step 300 (EVAL-21/22/23/24, 2026-04-22)** — step 150 m2 MATH-500 = 82.5%, step 300 m2 = 78.1% (offline confirms online peak→drift shape; does not exceed 1A's 83.1% ceiling). **EXP-16 step 225 model1 offline eval is explicitly deferred (low priority)** — extraction done on both machines 2026-04-20, but no inference run; the clean β=0 anchor baseline is now provided by EVAL-22/24 (1C m1) instead.

> **Reward-label bug note (2026-04-27)**: `wdl_sft_is` training before 2026-04-27 used GRPO-centered `advantages` as reward labels instead of raw `token_level_scores.sum(dim=-1)`. This affects EXP-16/17/18 and ABL-MINIRL-02/05/06/07, plus any pre-fix runs from `run_2b_sft.sh` / `run_2c_sft.sh` if they exist outside this index. Treat those as **pre-fix `wdl_sft_is` results**, not spec-correct WDL-SFT-IS. The launch scripts now default to `RUN_PREFIX` values ending in `-LABELFIX` for post-fix reruns. `wdl_sft` v1, `minirl`, and `vanilla` / GRPO runs are not affected by this bug.

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

## EXP-16: WDL-SFT-1A — v2 IS-Corrected (lr=5e-7, β=0, wdl_sft_is) ★

| Field | Value |
|---|---|
| **Script** | `run_on_policy_wdl_sft_qwen3_4b_math_1a.sh` |
| **Goal** | First v2 run: test whether IS/clip correction (`loss_mode=wdl_sft_is`) breaks through the v1 model2 online mean@1 ceiling of ~68% observed in M5.5. Fair A/B against M5.5 — identical lr, β, batch, horizon; only diff is v2 loss (binary-mask clip + token-level `rollout_is_weights`). |
| **Algorithm** | On-Policy WDL-SFT v2, forward-only (β=0), loss_mode=**wdl_sft_is**, seq-mean-token-sum, clip_ratio_low/high=0.2/0.27, rollout_is=token, rollout_is_threshold=5.0 |
| **Model** | QwenJoint-4B (weak=Qwen3-4B-Base, strong=Qwen3-4B-Base-SFT-stage-1, λ=0.5) |
| **Dataset** | EnsembleLLM MATH (`train_rl_format.parquet`) / MATH-500 + AIME-2025 (val) |
| **Key Params** | lr=5e-7, warmup=5, batch=64 prompts, n_resp=8, max_resp_len=4096, grad_clip=500.0, weight_decay=0.1, total_training_steps=300, val_freq=25, save_freq=25, 8 GPUs |
| **Logs** | `WDL-SFT-Qwen3-4B-MATH-1A_1776594597.log` (steps 0 → 275, crashed on disk-full at ckpt save) |
| | `WDL-SFT-Qwen3-4B-MATH-1A_1776594597_resumed_1776671476.log` (resumed from step 250, completed 275 + 300) |
| **Validation Dir** | `validation/WDL-SFT-Qwen3-4B-MATH-1A_1776594597/` (step 0, 25, 50, …, 300) |
| **Checkpoint Dir** | `/data-1/checkpoints/WDL-SFT-Qwen3-4B-MATH-1A_1776594597/` (12 checkpoints: steps 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300) |
| **Best Checkpoint** | **step 225** — online MATH-500 = **71.37%**, AIME-2025 = 23.08% (online peak); step 300 (final) = 70.36% / 23.08% |
| **Online Val Progression (MATH-500 / AIME-2025)** | step 0: 58.87% / 15.38% → step 25: 65.93% / 15.38% → step 50: 66.33% / 15.38% → step 75: 66.33% / 15.38% → step 100: 69.76% / 19.23% → step 125: 69.15% / 19.23% → step 150: 69.96% / 23.08% → step 175: 68.55% / 23.08% → step 200: 69.56% / 19.23% → **step 225: 71.37% / 23.08%** → step 250: 69.56% / **26.92%** → step 275: 70.36% / 15.38% → **step 300: 70.36% / 23.08%** |
| **Inference** | **Partial** — model2 step 225 offline eval complete 2026-04-20 → **EVAL-20** (MATH-500 mean@3 = **83.1%**, AQUA 70.2%, GSM8K 91.3%, MAWPS 95.4%, SVAMP 93.7%; breaks v1 m2 ceiling by +3.5 pp). model1 offline eval **explicitly deferred (low priority)** — weights extracted to `/data-1/model_weights/WDL-SFT-4B-MATH-1A/step_225_model1/` on both machines 2026-04-20 but inference not run; scheduled after higher-priority experiments. Under β=0 the reverse-SFT term has zero gradient so m1 should be ≈ untouched Qwen3-4B-Base; not required for the v2 headline result. |
| **W&B** | Project: `OnPolicyWDLSFT`, Run: `WDL-SFT-Qwen3-4B-MATH-1A_1776594597` (offline; `offline-run-20260419_103345-g9docxqz` initial + `offline-run-20260420_075458-irihn0a8` resume; synced 2026-04-20) |
| **Status** | **Complete & stable ✓** — full 300-step run under v2 IS-corrected loss. Interrupted once at step 275 (disk full during ckpt save, not a training failure); resumed cleanly from step 250. Online MATH-500 peaked at step 225 (71.37%), regressed slightly to 70.36% at step 300. **Exceeds v1 baseline M5.5** at matched step 300 by **+2.4 pp** (70.36% vs 67.94%). `critic/rewards/mean` climbed 0.29 → 0.68 over the last 50 steps, consistent with stable on-policy learning. |
| **Context** | First v2 experiment. See `docs/joint_training/plans/active/wdl_sft_is.md` (experiment 1a) and `docs/joint_training/specs/wdl_sft_is.md`. Offline mean@3 confirmed v2 also breaks the v1 model2 mean@3 ceiling (~79.6% in EVAL-10) — EVAL-20 lands at 83.1% (+3.5 pp). m1 offline eval is deferred. |

---

## EXP-17: WDL-SFT-1B — v2 IS-Corrected + Reverse SFT Re-Test (lr=5e-7, β=0.1, wdl_sft_is) ★

| Field | Value |
|---|---|
| **Script** | `run_on_policy_wdl_sft_qwen3_4b_math_1b.sh` |
| **Goal** | Re-test reverse SFT (β=0.1) under v2 loss at the same lr as 1A. Under v1, M5.6 (same lr, same β) trained through 300 steps but OFFLINE EVAL-15 revealed model1 format-compliance collapse (MATH-500 −21.6%, extraction_fail uniform 24–28%) — interpreted as the reverse term eroding anchor format tokens. v2 adds a lower-bound binary mask on negative samples (`ratio < 1 − clip_ratio_low` → zero gradient), which should contain the mechanism. Decision criterion: 1B online ≥ 1A → reverse SFT has incremental value under v2; pending offline eval on model1 to confirm format compliance is preserved. |
| **Algorithm** | On-Policy WDL-SFT v2, bidirectional (β=0.1), loss_mode=**wdl_sft_is**, seq-mean-token-sum, clip_ratio_low/high=0.2/0.27, rollout_is=token, rollout_is_threshold=5.0 |
| **Model** | QwenJoint-4B (weak=Qwen3-4B-Base, strong=Qwen3-4B-Base-SFT-stage-1, λ=0.5) |
| **Dataset** | EnsembleLLM MATH (`train_rl_format.parquet`) / MATH-500 + AIME-2025 (val) |
| **Key Params** | lr=5e-7, warmup=5, batch=64 prompts, n_resp=8, max_resp_len=4096, grad_clip=500.0, weight_decay=0.1, total_training_steps=300, val_freq=25, save_freq=25, 8 GPUs |
| **Logs** | `WDL-SFT-Qwen3-4B-MATH-1B_1776695220.log` (steps 0 → 300, complete in ~20h, no resumes) |
| **Validation Dir** | `validation/WDL-SFT-Qwen3-4B-MATH-1B_1776695220/` (step 0, 25, 50, …, 300) |
| **Checkpoint Dir** | `/data-1/checkpoints/WDL-SFT-Qwen3-4B-MATH-1B_1776695220/` (all 12 FSDP ckpts — steps 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300 — **deleted 2026-04-21 on both machines**, ~1.1T local + ~186G remote, after step_275/300 trio weights were double-mirrored and EVAL-16/17/18/19 completed; the surviving artifacts are `/data-1/model_weights/WDL-SFT-4B-MATH-1B/step_{275,300}{,_model1,_model2}/`) |
| **Best Checkpoint** | **step 225** and **step 275** tied — online MATH-500 = **70.97%** (model2-only). Step 300 final = 70.36% / 15.38%. |
| **Online Val Progression (MATH-500 / AIME-2025, model2-only)** | step 0: 40.73% / 7.69% → step 25: 62.70% / 19.23% → step 50: 64.92% / 11.54% → step 75: 67.94% / **26.92%** → step 100: 67.14% / 23.08% → step 125: 67.94% / **26.92%** → step 150: 70.16% / 23.08% → step 175: 70.56% / 19.23% → step 200: 70.77% / 23.08% → **step 225: 70.97% / 19.23%** → step 250: 70.16% / 19.23% → **step 275: 70.97% / 19.23%** → step 300: 70.36% / 15.38% |
| **Inference** | ✓ Complete (2026-04-21). Offline vLLM n=3, 7 benchmarks, tp=8 on Eval machine L40S. **Step 275** → EVAL-16 (m2, MATH-500 **82.5%**) / EVAL-17 (m1, MATH-500 38.7%, ext_fail 40–49%). **Step 300** → EVAL-18 (m2, MATH-500 **82.9%**) / EVAL-19 (m1, MATH-500 37.9%, ext_fail 42–49%). **m2 breaks v1 ceiling** (matches 1A prelim ~83%); **m1 collapse is MORE severe than v1 EVAL-15** — v2's lower-bound clip does NOT rescue the anchor under β>0. See INFERENCE_RESULTS.md for full tables. |
| **W&B** | Project: `OnPolicyWDLSFT`, Run: `WDL-SFT-Qwen3-4B-MATH-1B_1776695220` (offline; `offline-run-20260420_143041-dp91giwo`; pending sync) |
| **Status** | **Complete & stable (training) ✓ | m1 format collapse WORSE than v1 ✗** — training ran clean for 300 steps with zero resumes; online model2 trajectory within 0.5 pp of 1A at every val point. Offline eval (2026-04-21) cleanly separates the two failure modes: m2 is healthy (82.5%/82.9% MATH-500, matches 1A v2 ceiling), m1 is *more* broken than v1 EVAL-15 (extraction_fail 37–49% uniform vs v1's 24–28%). **The v2 lower-bound clip does not rescue model1** under β>0. |
| **Context** | Experiment 1b in `docs/joint_training/plans/active/wdl_sft_is.md`. Per §4 decision matrix: 1B online ≈ 1A (within 0.5% throughout) → reverse SFT does NOT hurt model2 under v2 at either training or offline eval. **But offline EVAL-17/19 refutes the hypothesis that v2's lower-bound clip prevents model1 format collapse.** Practical implication: β>0 is only useful when m1 is discarded and only m2 is deployed, which nullifies the reverse-SFT ablation's motivation. Forward-only (β=0, EXP-16 1A) remains the recommended default. |

---

## EXP-18: WDL-SFT-1C — v2 IS-Corrected at Higher LR (lr=1e-6, β=0, wdl_sft_is) ★

| Field | Value |
|---|---|
| **Script** | `run_on_policy_wdl_sft_qwen3_4b_math_1c.sh` |
| **Goal** | Re-test lr=1e-6 under v2. Under v1, EXP-15 (LR3) at lr=1e-6 peaked at step 125 (68.15%) then drifted. Disambiguate whether the drift was caused by the higher lr itself or by v1's missing stability mechanisms. Decision criterion: 1C stable through 300 steps AND reaches/exceeds 1A peak → drift was loss-layer; safe to use higher lr. Still drifts → lr=1e-6 is genuinely past the stable region. |
| **Algorithm** | On-Policy WDL-SFT v2, forward-only (β=0), loss_mode=**wdl_sft_is**, seq-mean-token-sum, clip_ratio_low/high=0.2/0.27, rollout_is=token, rollout_is_threshold=5.0 |
| **Model** | QwenJoint-4B (weak=Qwen3-4B-Base, strong=Qwen3-4B-Base-SFT-stage-1, λ=0.5) |
| **Dataset** | EnsembleLLM MATH (`train_rl_format.parquet`) / MATH-500 + AIME-2025 (val) |
| **Key Params** | lr=**1e-6** (doubled vs 1A/1B), warmup=5, batch=64 prompts, n_resp=8, max_resp_len=4096, grad_clip=500.0, weight_decay=0.1, total_training_steps=300, val_freq=25, save_freq=25, 8 GPUs |
| **Logs** | `WDL-SFT-Qwen3-4B-MATH-1C_1776768784.log` (steps 0 → 300, complete in ~18.5h, no resumes) |
| **Validation Dir** | `validation/WDL-SFT-Qwen3-4B-MATH-1C_1776768784/` (13 jsonls: step 0, 25, 50, …, 300) |
| **Checkpoint Dir** | `/data-1/checkpoints/WDL-SFT-Qwen3-4B-MATH-1C_1776768784/` (12 checkpoints: steps 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300) |
| **Best Checkpoint** | **step 150** — online MATH-500 = **71.98%** (model2-only, peak across all v2 runs); AIME-2025 15.38%. Step 300 final = 67.34% / 7.69%. |
| **Online Val Progression (MATH-500 / AIME-2025, model2-only)** | step 0: 60.89% / 15.38% → step 25: 65.32% / 15.38% → step 50: 68.55% / 19.23% → step 75: 69.56% / 19.23% → step 100: 70.77% / 15.38% → step 125: 70.56% / **23.08%** → **step 150: 71.98% / 15.38%** → step 175: 68.55% / 15.38% → step 200: 71.17% / 19.23% → step 225: 69.76% / 15.38% → step 250: 69.56% / 15.38% → step 275: 68.75% / 19.23% → step 300: 67.34% / 7.69% |
| **Inference** | ✅ Complete (2026-04-22, Eval machine L40S tp=8): 4 EVAL entries covering step 150 + step 300 × model1 + model2. **Best: EVAL-21 step 150 m2, MATH-500 mean@3 82.5%, GSM8K 91.7%, AMC 63.3%, AIME-2025 18.9%**. See EVAL-21 (step 150 m2 ★), EVAL-22 (step 150 m1), EVAL-23 (step 300 m2), EVAL-24 (step 300 m1) in INFERENCE_RESULTS.md. |
| **W&B** | Project: `OnPolicyWDLSFT`, Run: `WDL-SFT-Qwen3-4B-MATH-1C_1776768784` |
| **Status** | **Complete & stable ✓ (training & offline eval both done)** — full 300-step run under v2 IS-corrected loss at lr=1e-6, zero resumes or interventions. v1 LR3's post-step-125 collapse does **not** reproduce under v2: 1C reached its online peak at step 150 (**71.98%**, the highest MATH-500 point across all v2 runs, above 1A's 71.37% and 1B's 70.97%), then oscillated in the 67–71% band without the monotonic downward drift that defined v1 LR3. End-of-run is weaker than 1A/1B: step 300 MATH-500 = 67.34% (vs 1A/1B's 70.36%) and AIME = 7.69% (vs 1A's 23.08% / 1B's 15.38%). **Offline m2 mean@3 mirrors the online shape**: step 150 m2 MATH-500 = 82.5% (= 1B step 275), step 300 m2 = 78.1% (−4.4 pp peak-to-end). 1C m2 peak does **not** exceed 1A m2 step 225 (83.1%) — higher lr does not raise the v2 offline ceiling. **Surprise offline m1 finding**: 1C m1 IMPROVES step 150 → step 300 (MATH +12, GSM8K +22 pp) under β=0, reverse of 1B's β=0.1 m1 degradation. Extract recommendation: pull m2 from step 150 (not step 300). |
| **Context** | Experiment 1c in `docs/joint_training/plans/active/wdl_sft_is.md`. Decision-criterion verdict: **v2 clip+IS contains lr=1e-6 drift at the training level** (v1 LR3's crash was loss-layer, not lr-layer) AND **the extra stability does not translate to an offline win over 1A's lr=5e-7** — offline m2 MATH-500 ceiling is ~83% across v2 runs regardless of lr. Recommendation going forward: stay at lr=5e-7, β=0 (1A recipe) as the safe default; 1C is the decisive negative result for "double the lr under v2 to push ceiling higher." |

---

# Ablation Series 2X — Single-Model (plan: `docs/joint_training/plans/active/ablation_single_model.md`)

The 2X series isolates the loss-vs-fusion-vs-init axes of the 1A/1B/1C findings by dropping the joint model + fused-logit rollout. Each 2X run uses a standard single Qwen3 backbone with identical data budget (300 × 64 × 8 = 153,600 responses/run) to its paired 1X joint run. Scripts: `recipe/on_policy_wdl_sft/ablation_single_model/`.

## ABL-MINIRL-01: MINIRL-2Z-SFT — Single-Model MiniRL Baseline from SFT Init (lr=5e-7) ★

| Field | Value |
|---|---|
| **Script** | `recipe/on_policy_wdl_sft/ablation_single_model/run_2z_sft.sh` (via `/data-1/verl07/run_train.sh`, docker `verl-harness`) |
| **Goal** | Reference floor for the 2X series: pure RL (MiniRL) from the SFT-stage-1 init that 1X's model2 uses. Quantifies $L_\text{loss}$ = score(2X-SFT) − score(2Z-SFT), and tests H3 (init-dominant) — if 2Z-SFT online ≈ 1X online, the joint + wdl_sft_is machinery is not the main driver of 1X's lift. |
| **Algorithm** | MiniRL (PG with clip + IS), loss_mode=**minirl**, seq-mean-token-sum, clip_ratio_low/high=0.2/0.27, rollout_is=token, rollout_is_threshold=5.0 |
| **Model** | Qwen3-4B (single backbone) initialized from Qwen3-4B-Base-SFT-stage-1 |
| **Dataset** | EnsembleLLM MATH (`train_rl_format.parquet`) / MATH-500 + AIME-2025 (val) |
| **Key Params** | lr=5e-7, warmup=5, batch=64 prompts, n_resp=8, mini_bsz=8, max_resp_len=4096, grad_clip=500.0, weight_decay=0.1, total_training_steps=300, val_freq=25, save_freq=25, 8 GPUs |
| **Logs** | `ablation_single_model/MINIRL-Qwen3-4B-MATH-2Z-SFT_1776855436.log` (steps 0 → 300, complete in ~14.4h, no resumes) |
| **Validation Dir** | `ablation_single_model/validation/MINIRL-Qwen3-4B-MATH-2Z-SFT_1776855436/` (step 0, 25, 50, …, 300) |
| **Checkpoint Dir** | `/data-1/checkpoints/MINIRL-Qwen3-4B-MATH-2Z-SFT_1776855436/` (12 checkpoints: steps 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300) |
| **Best Checkpoint** | **step 275** — online MATH-500 = **70.56%**, AIME-2025 = 15.38% (online peak). Step 300 final = 70.16% / 11.54%. |
| **Online Val Progression (MATH-500 / AIME-2025)** | step 25: 66.73% / 15.38% → step 50: 66.94% / **23.08%** → step 75: 66.73% / **23.08%** → step 100: 67.14% / 15.38% → step 125: 67.34% / 11.54% → step 150: 66.53% / 19.23% → step 175: 67.14% / 15.38% → step 200: 69.76% / 19.23% → step 225: 68.55% / **23.08%** → step 250: 68.75% / 19.23% → **step 275: 70.56% / 15.38%** → step 300: 70.16% / 11.54% |
| **Training Dynamics** | entropy stable 1808–2929 (no collapse); grad_norm 430–685 (clipped at 500 but not runaway); pg_clipfrac ≤0.0015; rollout_corr/kl ≈1e-3. reward/mean climbed 0.156 → 0.531 monotonically — healthy MiniRL signal. |
| **Inference** | **Complete (2026-04-22)** — offline n=3 vLLM eval on Eval machine (L40S, tp=8). **EVAL-25** (step 275): MATH-500 mean@3 = **79.6%**, AIME-2025 21.1%, GSM8K 92.5%, MAWPS 95.6%, SVAMP 94.1%, AQUA 82.3%, AMC23 61.7%. **EVAL-26** (step 300): MATH-500 mean@3 = **80.7%**, AIME-2025 13.3%, GSM8K 91.9%, MAWPS 95.7%, SVAMP 93.7%, AQUA 82.0%, AMC23 60.8%. Extraction_fail ≤ 3% on MATH-500 / ≤ 0.1% on GSM8K/MAWPS/SVAMP — clean format compliance, no β>0 style collapse. |
| **W&B** | Project: `OnPolicyWDLSFT`, Run: `MINIRL-Qwen3-4B-MATH-2Z-SFT_1776855436` (offline) |
| **Status** | **Complete & stable ✓** — first ABL-MINIRL run, pure-RL reference floor. Online MATH-500 peak 70.56% (step 275) lands within 0.4–0.8 pp of 1A/1B v2 joint peaks (71.37% / 70.97%). **Offline decides H3 decisively**: step_300 MATH-500 mean@3 **80.7%** trails 1A step_225 m2 (**83.1%**, EVAL-20) by **−2.4 pp** and 1B step_300 m2 (82.9%, EVAL-18) by **−2.2 pp**. The v2 joint machinery (fused-logit rollout + wdl_sft_is loss + joint param sharing) adds ~2.4 pp on MATH-500 over single-model MiniRL from the same SFT init. H3 (init-dominant) is partially supported — online val fails to separate joint and single-model runs, but offline mean@3 does separate them by a consistent ~2 pp margin. |
| **Context** | Reference floor for the 2X series. Plan: `docs/joint_training/plans/active/ablation_single_model.md` §5 — $L_\text{loss}$ = score(2A-SFT) − score(2Z-SFT), interpreted together with $L_\text{fusion}$ = score(1A) − score(2A-SFT). |

---

## ABL-MINIRL-02: WDL-SFT-2A-SFT — Single-Model + wdl_sft_is Loss from SFT Init (lr=5e-7)

| Field | Value |
|---|---|
| **Model** | Qwen3-4B (single backbone) initialized from Qwen3-4B-Base-SFT-stage-1 |
| **Dataset** | EnsembleLLM MATH (`train_rl_format.parquet`) / MATH-500 + AIME-2025 (val) |
| **Key Params** | lr=5e-7, `loss_mode=wdl_sft_is` (single-model v2 loss), batch=64 prompts, n_resp=8, mini_bsz=8, max_resp_len=4096, total_training_steps=300, val_freq=25, save_freq=25, 8 GPUs |
| **Logs** | `ablation_single_model/WDL-SFT-Qwen3-4B-MATH-2A-SFT_1776892819.log` |
| **Validation Dir** | `ablation_single_model/validation/WDL-SFT-Qwen3-4B-MATH-2A-SFT_1776892819/` |
| **Checkpoint Dir** | `/data-1/checkpoints/WDL-SFT-Qwen3-4B-MATH-2A-SFT_1776892819/` (12 FSDP ckpts: steps 25/50/…/300; both machines after 2026-04-23 rsync) |
| **Best Checkpoint** | **step 300** (online MATH-500 = 70.4%) — tied with step 50 (70.2%) at top of online range. |
| **Online Val Progression (MATH-500 / AIME-2025)** | step 25: 64.9% / 15.4% → step 50: **70.2%** / 19.2% → step 75: 67.1% / 15.4% → step 100: 68.8% / 11.5% → step 125: 69.2% / 19.2% → step 150: 69.0% / **23.1%** → step 175: 68.8% / 15.4% → step 200: 67.9% / 15.4% → step 225: 69.2% / 15.4% → step 250: 68.3% / **26.9%** → step 275: 69.0% / 15.4% → step 300: **70.4%** / 7.7% |
| **Training Dynamics** | Clean 300-step run at lr=5e-7 with `loss_mode=wdl_sft_is` on single Qwen3-4B. Same step budget, batch size, init, and val schedule as ABL-MINIRL-01 (2Z-SFT) — the two are a matched pair for isolating the **loss term** contribution. |
| **Inference** | **Complete (2026-04-23)** — offline n=3 vLLM eval on Eval machine (L40S, tp=8). **EVAL-27** (step 275): MATH-500 mean@3 = **80.1%** / pass@3 = **88.4%**, AIME-2025 11.1%/16.7%, AMC23 55.8%/75.0%, AQUA 66.3%/79.1%, GSM8K 90.6%/93.6%, MAWPS 94.3%/95.2%, SVAMP 92.8%/95.0%. **EVAL-28** (step 300): MATH-500 mean@3 = **80.1%** / pass@3 = **88.6%**, AIME-2025 10.0%/16.7%, AMC23 60.0%/**82.5%**, AQUA 57.0%/68.9%, GSM8K 90.4%/93.7%, MAWPS 94.5%/95.5%, SVAMP 91.7%/95.7%. Extraction_fail clean on MATH-500 (4.8%) and easy benches (≤0.1%). |
| **W&B** | Project: `OnPolicyWDLSFT`, Run: `WDL-SFT-Qwen3-4B-MATH-2A-SFT_1776892819` (offline) |
| **Status** | **Complete & stable ✓** — second ABL-MINIRL run. **L_loss decomposition (2A − 2Z)**: MATH-500 mean@3 ≈ 0 at step 300 (80.1 vs 80.7, −0.6 pp); pass@3 ≈ 0 (88.6 vs 89.2). **AQUA regression is striking: −25 pp mean@3 / −21 pp pass@3 at step 300**, not format-driven (ext_fail comparable). Easy benches small −1 to −2 pp. AMC-2023 step 300 pass@3 **82.5%** is a new v2-family best, beating 1B step 300 m2 (80.0%). **Plan §5 resolution**: with L_loss ≈ 0 on MATH-500, the full joint-vs-single MATH-500 lift collapses onto **L_fusion** (joint architecture + fused-logit rollout + parameter sharing), measured as +3.0 pp mean@3 / +1.6 pp pass@3 (EVAL-20 1A m2 vs EVAL-28). |
| **Context** | L_loss isolation for the 2X series — matched pair with ABL-MINIRL-01 (2Z-SFT). Plan: `docs/joint_training/plans/active/ablation_single_model.md` §5. The joint contribution on MATH-500 is now attributed to architecture, not loss. Next natural step: **joint + MiniRL loss (no wdl_sft_is)** — would complete the 2×2 (loss × architecture) grid and test whether the `pass@3 − mean@3` tightness (6.1–6.7 pp for v2 joint m2 vs 8.3–8.8 pp for both ABLs) is architecture-driven. |

---

## ABL-MINIRL-03: MINIRL-2Z-BASE — Single-Model MiniRL Baseline from Base Init (lr=5e-7) [Meituan]

| Field | Value |
|---|---|
| **Script** | `recipe/on_policy_wdl_sft/ablation_single_model/run_2z_base.sh` (Meituan AFO, dolphinfs) |
| **Goal** | Reference floor for the Base-init arm of the 2X series: pure RL (MiniRL) from Qwen3-4B-Base. Pairs with 2Z-SFT (ABL-MINIRL-01) to isolate the **init contribution** L_init = score(2Z-SFT) − score(2Z-BASE), and with 2A-BASE to isolate the loss term on Base init. |
| **Algorithm** | MiniRL (PG with clip + IS), loss_mode=**minirl**, seq-mean-token-sum, clip_ratio_low/high=0.2/0.27, rollout_is=token, rollout_is_threshold=5.0 |
| **Model** | Qwen3-4B (single backbone) initialized from **Qwen3-4B-Base** (no SFT) |
| **Dataset** | EnsembleLLM MATH (`train_rl_format.parquet`) / MATH-500 + AIME-2025 (val) |
| **Key Params** | lr=5e-7, warmup=5, batch=64 prompts, n_resp=8, mini_bsz=8, max_resp_len=4096, grad_clip=500.0, weight_decay=0.1, total_training_steps=300, val_freq=25, save_freq=25, 8 GPUs |
| **Logs** | `ablation_single_model/MINIRL-Qwen3-4B-MATH-2Z-BASE_1776928234.log` (trl-import crash at init, zero steps trained) |
| | `ablation_single_model/MINIRL-Qwen3-4B-MATH-2Z-BASE_1776928234_resumed_1776932913.log` (clean restart from step 0, ran to step 300) |
| **Validation Dir** | `ablation_single_model/validation/MINIRL-Qwen3-4B-MATH-2Z-BASE_1776928234/` (13 jsonls: step 0, 25, 50, …, 300) |
| **Metrics** | `ablation_single_model/metrics/OnPolicyWDLSFT/MINIRL-Qwen3-4B-MATH-2Z-BASE_1776928234.jsonl` |
| **Checkpoint Dir** | **Meituan dolphinfs** `/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yangfengkai02/lgx/verl-exp/checkpoints/MINIRL-Qwen3-4B-MATH-2Z-BASE_1776928234/` (12 FSDP ckpts: steps 25/50/…/300). Not yet transferred to local `/data-1`. |
| **Best Checkpoint** | **step 100** — online MATH-500 = **76.21%** (peak); AIME-2025 step_275 = 15.38%. Step 300 final = 69.35% / 11.54%. |
| **Online Val Progression (MATH-500 / AIME-2025)** | step 0: 29.44% / 3.85% → step 25: 70.77% / 15.38% → step 50: 73.19% / 15.38% → step 75: 75.20% / 0.00% → **step 100: 76.21% / 11.54%** → step 125: 75.81% / 11.54% → step 150: 74.60% / 7.69% → step 175: 67.74% / 3.85% → step 200: 66.13% / **19.23%** → step 225: 64.52% / 11.54% → step 250: 66.73% / 15.38% → step 275: 68.75% / 15.38% → step 300: 69.35% / 11.54% |
| **Training Dynamics** | entropy 101–3280 (no collapse); grad_norm 126–497 (well inside the 500 clip, no clipping in practice); pg_clipfrac ≤ 0.0013; rollout_corr/kl ≈ 5e-4. reward/mean climbed −0.93 → 0.61 (peak 0.73) with clean monotonicity through step 100, then oscillates as online MATH-500 drifts. |
| **Inference** | Pending — offline n=3 vLLM eval not yet run; model weights still on Meituan dolphinfs. |
| **W&B** | Project: `OnPolicyWDLSFT`, Run: `MINIRL-Qwen3-4B-MATH-2Z-BASE_1776928234` (offline; not yet synced from Meituan) |
| **Status** | **Complete (online) ✓ with mid-run online drift** — first-launch attempt crashed at init before step 0 (trl-import incompatibility, unrelated to training); second attempt restarted from step 0 and ran 300 steps clean. Online MATH-500 **peaks at step 100 (76.21%)** — the highest point across the whole Base-init quintet — then drifts ~12 pp down to 64.52% by step 225 before partially recovering to 69.35% at step 300. AIME weak throughout (peak 19.23% step 200). This is the same "learn fast then drift" shape that 2Z-SFT did NOT exhibit (2Z-SFT was monotone ~66 → 70%) — so the Base-init + MiniRL combination is less stable than SFT-init + MiniRL. |
| **Context** | Base-init reference floor for the 2X-BASE sub-series. Plan: `docs/joint_training/plans/active/ablation_single_model.md` §3.3 (2Z-base row). Surprising observation: **2Z-BASE online peak (76.21%) > 2Z-SFT online peak (70.56%)** at matched step budget and loss — online val does not support H3 (init-dominant) in the direction originally hypothesized. Offline eval is required before drawing conclusions (the drifted end-of-run ckpt may not carry this online peak through to mean@3). |

---

## ABL-MINIRL-04: GRPO-2G-BASE — Canonical GRPO Baseline from Base Init (lr=5e-7) [Meituan]

| Field | Value |
|---|---|
| **Script** | `recipe/on_policy_wdl_sft/ablation_single_model/run_2g_base.sh` (Meituan AFO, dolphinfs) |
| **Goal** | Canonical GRPO baseline from Base init. Pairs with 2Z-BASE (MiniRL from same init): isolates the **loss family** (PPO-clip + norm_adv_by_std vs MiniRL IS + binary-mask) at matched init/lr/data budget. Reference for "what does vanilla GRPO get on Base init". |
| **Algorithm** | Canonical GRPO (PPO-clip), loss_mode=**vanilla**, seq-mean-token-sum, clip_ratio_low/high=0.2/**0.2** (symmetric), `norm_adv_by_std_in_grpo=True`, no IS correction (`rollout_is=null`) |
| **Model** | Qwen3-4B (single backbone) initialized from **Qwen3-4B-Base** |
| **Dataset** | EnsembleLLM MATH (`train_rl_format.parquet`) / MATH-500 + AIME-2025 (val) |
| **Key Params** | lr=5e-7, warmup=5, batch=64 prompts, n_resp=8, mini_bsz=8, max_resp_len=4096, grad_clip=500.0, weight_decay=0.1, total_training_steps=300, val_freq=25, save_freq=25, 8 GPUs |
| **Logs** | `ablation_single_model/GRPO-Qwen3-4B-MATH-2G-BASE_1776937259.log` (steps 0 → 300, no resumes) |
| **Validation Dir** | `ablation_single_model/validation/GRPO-Qwen3-4B-MATH-2G-BASE_1776937259/` (13 jsonls: step 0, 25, …, 300) |
| **Metrics** | `ablation_single_model/metrics/OnPolicyWDLSFT/GRPO-Qwen3-4B-MATH-2G-BASE_1776937259.jsonl` |
| **Checkpoint Dir** | **Meituan dolphinfs** `/mnt/dolphinfs/ssd_pool/.../verl-exp/checkpoints/GRPO-Qwen3-4B-MATH-2G-BASE_1776937259/` (12 FSDP ckpts: steps 25/50/…/300). Not yet transferred. |
| **Best Checkpoint** | **step 275** — online MATH-500 = **71.17%**, AIME-2025 = 7.69%. Step 300 final = 70.16% / 15.38%. |
| **Online Val Progression (MATH-500 / AIME-2025)** | step 0: 30.44% / 3.85% → step 25: 68.55% / 7.69% → step 50: 71.77% / 3.85% → step 75: 69.96% / 15.38% → step 100: 70.77% / 11.54% → step 125: 70.77% / 7.69% → step 150: 69.56% / **19.23%** → step 175: 69.35% / 15.38% → step 200: 68.95% / **19.23%** → step 225: 70.36% / **19.23%** → step 250: 69.56% / 15.38% → **step 275: 71.17% / 7.69%** → step 300: 70.16% / 15.38% |
| **Training Dynamics** | entropy 25–4297 (healthy range); grad_norm 158–812 (under the 500 clip most of the time, rare brief spikes above); pg_clipfrac ≤ 0.0017 (symmetric 0.2 clip rarely binds); rollout_corr/kl ≈ 3e-4. reward/mean −0.90 → 0.65 monotonically; reward peak 0.83 (the highest across the Base-init quintet). |
| **Inference** | Pending — offline n=3 vLLM eval not yet run; weights on Meituan dolphinfs. |
| **W&B** | Project: `OnPolicyWDLSFT`, Run: `GRPO-Qwen3-4B-MATH-2G-BASE_1776937259` (offline; not yet synced) |
| **Status** | **Complete & most-stable-in-series ✓** — full 300 steps without drift. Online MATH-500 stays in the 68.5–71.2% band from step 25 onward, peak 71.17% (step 275), end 70.16% (step 300). No monotonic drift like 2Z-BASE; no β>0 collapse like 2B-BASE. Lower ceiling than 2A-BASE / 2Z-BASE online (peak 71.2% vs 76.2% for both) but higher step-300 final than 2Z-BASE (70.16% vs 69.35%) because it avoids the drift. |
| **Context** | Canonical GRPO reference on Base init — fills the "loss family" cell next to 2Z-BASE's MiniRL. Plan: `docs/joint_training/plans/active/ablation_single_model.md` (2G added post-hoc to the matrix; plan §3.3 should be extended). Early reading: **vanilla PPO-clip is more stable than MiniRL on Base init** (no drift), but has a lower online ceiling — the IS/asymmetric-clip machinery buys headroom at the cost of late-training stability. Offline eval will confirm whether "more stable but lower ceiling" survives multi-sample. |

---

## ABL-MINIRL-05: WDL-SFT-2A-BASE — Single-Model + wdl_sft_is Loss from Base Init (lr=5e-7, β=0) [Meituan] ★

| Field | Value |
|---|---|
| **Script** | `recipe/on_policy_wdl_sft/ablation_single_model/run_2a_base.sh` (Meituan AFO, dolphinfs) |
| **Goal** | Does the `wdl_sft_is` loss alone (without fused-logit rollout, without SFT init) lift a Base model? Matches 1A on β/lr/batch/N/steps; only differs in model (single Qwen3-4B-Base vs JointQwen3 with m2 from SFT) and rollout source (single-model vs fused logits). Pairs with 2A-SFT (ABL-MINIRL-02) for init-ablation; with 2Z-BASE for loss-ablation on Base init. |
| **Algorithm** | On-Policy WDL-SFT v2 (single-model), forward-only (β=0), loss_mode=**wdl_sft_is**, seq-mean-token-sum, clip_ratio_low/high=0.2/0.27, rollout_is=token, rollout_is_threshold=5.0 |
| **Model** | Qwen3-4B (single backbone) initialized from **Qwen3-4B-Base** |
| **Dataset** | EnsembleLLM MATH (`train_rl_format.parquet`) / MATH-500 + AIME-2025 (val) |
| **Key Params** | lr=5e-7, warmup=5, batch=64 prompts, n_resp=8, mini_bsz=8, max_resp_len=4096, grad_clip=500.0, weight_decay=0.1, total_training_steps=300, val_freq=25, save_freq=25, 8 GPUs |
| **Logs** | `ablation_single_model/WDL-SFT-Qwen3-4B-MATH-2A-BASE_1776937268.log` (steps 0 → 300, no resumes) |
| **Validation Dir** | `ablation_single_model/validation/WDL-SFT-Qwen3-4B-MATH-2A-BASE_1776937268/` (13 jsonls: step 0, 25, …, 300) |
| **Metrics** | `ablation_single_model/metrics/OnPolicyWDLSFT/WDL-SFT-Qwen3-4B-MATH-2A-BASE_1776937268.jsonl` |
| **Checkpoint Dir** | **Meituan dolphinfs** `/mnt/dolphinfs/ssd_pool/.../verl-exp/checkpoints/WDL-SFT-Qwen3-4B-MATH-2A-BASE_1776937268/` (12 FSDP ckpts: steps 25/50/…/300). Not yet transferred. |
| **Best Checkpoint** | **step 225** — online MATH-500 = **76.21%**, AIME-2025 = 0.00%. Step 300 final = 74.80% / 7.69%. Other online high points: step 150 76.01% / 19.23% (best AIME of run), step 250 75.00% / 11.54%. |
| **Online Val Progression (MATH-500 / AIME-2025)** | step 0: 30.65% / 3.85% → step 25: 72.58% / 0.00% → step 50: 71.17% / 7.69% → step 75: 71.17% / 7.69% → step 100: 73.39% / 0.00% → step 125: 74.19% / 7.69% → step 150: 76.01% / **19.23%** → step 175: 75.60% / 7.69% → step 200: 73.19% / 7.69% → **step 225: 76.21% / 0.00%** → step 250: 75.00% / 11.54% → step 275: 73.99% / 7.69% → step 300: 74.80% / 7.69% |
| **Training Dynamics** | entropy 15–3767 (wide range but no collapse); grad_norm 461–1169 (hits the 500 clip regularly — typical for base-init early training); pg_clipfrac ≤ 0.0029; rollout_corr/kl ≈ 4e-4. reward/mean −0.91 → 0.39 (peak 0.66) — note reward/mean is lower than 2Z-BASE's 0.61 at step 300 despite higher MATH-500 accuracy, consistent with wdl_sft_is including negative-sample penalty terms that depress raw reward. |
| **Inference** | Pending — offline n=3 vLLM eval not yet run; weights on Meituan dolphinfs. |
| **W&B** | Project: `OnPolicyWDLSFT`, Run: `WDL-SFT-Qwen3-4B-MATH-2A-BASE_1776937268` (offline; not yet synced) |
| **Status** | **Complete & stable ✓ — highest online (N=1) MATH-500 tail across every N=1-comparable run in this project.** Unlike 2Z-BASE, 2A-BASE does not drift: it climbs monotonically through step 225 (peak 76.21%) and holds 74–75% through step 300. **N=1-to-N=1 deltas at step 300**: +5.45 pp vs 2Z-BASE (same init, minirl loss), +4.37 pp vs 2A-SFT (same loss/β/lr, SFT init), +4.44 pp vs 1A model2-only (same loss/β/lr, joint), +4.64 pp vs 2G-BASE (GRPO baseline), +7.06 pp vs 2B-BASE (β=0.1). **Peak 76.21% @ step 225 is also the highest online peak across all 1X + 2X runs** (next-best online peaks: 1C 71.98%, 1A 71.37%, 2Z-BASE 76.21% but only at step 100 then drift). **All figures are online mean@1 (trainer val) — 2A-BASE has not been offline-evaluated (n=3) yet; do not cross-compare to the offline n=3 mean@3 numbers cited elsewhere in this doc until an n=3 eval on 2A-BASE is run.** |
| **Context** | Loss-effect isolation on Base init. Plan: `docs/joint_training/plans/active/ablation_single_model.md` §5 — **N=1-to-N=1** online L_loss: on Base init (2A-BASE − 2Z-BASE step 300 online) = **+5.45 pp**; on SFT init (2A-SFT − 2Z-SFT step 300 online = 70.4 − 70.16) = **+0.24 pp**. ABL-MINIRL-02's offline (n=3) likewise found L_loss_SFT ≈ −0.6 pp. Hypothesis: SFT init already captures most of what wdl_sft_is does via its pretrained format/anchor compliance, so the loss-delta is near zero from SFT init; Base init has room for the wdl_sft_is stability to matter. **All Base-side claims here are online N=1 only — offline n=3 mean@3 on 2A-BASE + 2Z-BASE is required before this finding is promoted out of "tentative" status** (online N=1 has historically diverged from offline n=3, e.g. 1B m2 online 70.97% → offline 82.9%). |

---

## ABL-MINIRL-06: WDL-SFT-2C-BASE — Single-Model + wdl_sft_is at Higher LR from Base Init (lr=1e-6, β=0) [Meituan]

| Field | Value |
|---|---|
| **Script** | `recipe/on_policy_wdl_sft/ablation_single_model/run_2c_base.sh` (Meituan AFO, dolphinfs) |
| **Goal** | Does doubling lr (1e-6 vs 2A-BASE's 5e-7) help or hurt single-model wdl_sft_is from Base init? Pairs with 1C (joint, same lr) for the "higher-lr" corner of the decomposition. |
| **Algorithm** | On-Policy WDL-SFT v2 (single-model), forward-only (β=0), loss_mode=**wdl_sft_is**, seq-mean-token-sum, clip_ratio_low/high=0.2/0.27, rollout_is=token, rollout_is_threshold=5.0 |
| **Model** | Qwen3-4B (single backbone) initialized from **Qwen3-4B-Base** |
| **Dataset** | EnsembleLLM MATH (`train_rl_format.parquet`) / MATH-500 + AIME-2025 (val) |
| **Key Params** | lr=**1e-6** (doubled vs 2A-BASE), warmup=5, batch=64 prompts, n_resp=8, mini_bsz=8, max_resp_len=4096, grad_clip=500.0, weight_decay=0.1, total_training_steps=300, val_freq=25, save_freq=25, 8 GPUs |
| **Logs** | `ablation_single_model/WDL-SFT-Qwen3-4B-MATH-2C-BASE_1776938745.log` (steps 0 → 300, no resumes) |
| **Validation Dir** | `ablation_single_model/validation/WDL-SFT-Qwen3-4B-MATH-2C-BASE_1776938745/` (13 jsonls: step 0, 25, …, 300) |
| **Metrics** | `ablation_single_model/metrics/OnPolicyWDLSFT/WDL-SFT-Qwen3-4B-MATH-2C-BASE_1776938745.jsonl` |
| **Checkpoint Dir** | **Meituan dolphinfs** `/mnt/dolphinfs/ssd_pool/.../verl-exp/checkpoints/WDL-SFT-Qwen3-4B-MATH-2C-BASE_1776938745/` (12 FSDP ckpts: steps 25/50/…/300). Not yet transferred. |
| **Best Checkpoint** | **step 100** — online MATH-500 = **74.19%**, AIME-2025 = 3.85%. Step 300 final = 72.18% / 3.85%. |
| **Online Val Progression (MATH-500 / AIME-2025)** | step 0: 30.65% / 0.00% → step 25: 72.78% / 7.69% → step 50: 71.57% / **19.23%** → step 75: 72.98% / **23.08%** → **step 100: 74.19% / 3.85%** → step 125: 73.19% / 7.69% → step 150: 72.18% / 11.54% → step 175: 70.97% / 11.54% → step 200: 72.38% / 11.54% → step 225: 71.37% / **19.23%** → step 250: 72.18% / 11.54% → step 275: 69.35% / 11.54% → step 300: 72.18% / 3.85% |
| **Training Dynamics** | entropy 18–3733; grad_norm 427–3188 (brief spikes above 1k — more volatile than 2A-BASE's 461–1169, consistent with doubled lr); pg_clipfrac ≤ 0.0039; rollout_corr/kl ≈ 8e-4. reward/mean −0.87 → 0.32 (peak 0.64). |
| **Inference** | Pending — offline n=3 vLLM eval not yet run; weights on Meituan dolphinfs. |
| **W&B** | Project: `OnPolicyWDLSFT`, Run: `WDL-SFT-Qwen3-4B-MATH-2C-BASE_1776938745` (offline; not yet synced) |
| **Status** | **Complete & stable ✓** — 300 steps with no resumes. lr=1e-6 on Base + single-model does **not** reproduce v1 LR3's monotonic drift: 2C-BASE peaks at step 100 (74.19%), oscillates in the 69–73% band, ends at 72.18%. However, **lr=1e-6 loses to lr=5e-7 on Base init**: step 300 72.18% (2C) vs 74.80% (2A) = **−2.62 pp online**; peak 74.19% (2C) vs 76.21% (2A) = **−2.02 pp online**. Same sign as the joint result (1C's ceiling did not beat 1A's). Higher lr reaches the plateau slightly earlier (step 75 at 72.98%) at the cost of lower steady-state. |
| **Context** | Higher-lr regime on Base init. Plan: `docs/joint_training/plans/active/ablation_single_model.md` §3.3 (2C-base row). Combined verdict across joint+single: **lr=5e-7 is the default for both joint and single-model v2 from either init**; lr=1e-6 does not buy ceiling in any tested configuration. |

---

## ABL-MINIRL-07: WDL-SFT-2B-BASE — Single-Model + wdl_sft_is with Reverse SFT from Base Init (lr=5e-7, β=0.1) [Meituan]

| Field | Value |
|---|---|
| **Script** | `recipe/on_policy_wdl_sft/ablation_single_model/run_2b_base.sh` (Meituan AFO, dolphinfs) |
| **Goal** | Does β=0.1 reverse SFT stay stable when applied to a single Base model under v2? Under v1 (EXP-12 M5, lr=1e-6) β>0 diverged catastrophically; on joint v2 1B matched 1A online but collapsed m1 offline. This is the first v2 single-model + β>0 + Base-init test. |
| **Algorithm** | On-Policy WDL-SFT v2 (single-model), bidirectional (β=0.1), loss_mode=**wdl_sft_is**, seq-mean-token-sum, clip_ratio_low/high=0.2/0.27, rollout_is=token, rollout_is_threshold=5.0 |
| **Model** | Qwen3-4B (single backbone) initialized from **Qwen3-4B-Base** |
| **Dataset** | EnsembleLLM MATH (`train_rl_format.parquet`) / MATH-500 + AIME-2025 (val) |
| **Key Params** | lr=5e-7, warmup=5, batch=64 prompts, n_resp=8, mini_bsz=8, max_resp_len=4096, grad_clip=500.0, weight_decay=0.1, total_training_steps=300, val_freq=25, save_freq=25, 8 GPUs |
| **Logs** | `ablation_single_model/WDL-SFT-Qwen3-4B-MATH-2B-BASE_1776942366.log` (steps 0 → 300, no resumes) |
| **Validation Dir** | `ablation_single_model/validation/WDL-SFT-Qwen3-4B-MATH-2B-BASE_1776942366/` (13 jsonls: step 0, 25, …, 300) |
| **Metrics** | `ablation_single_model/metrics/OnPolicyWDLSFT/WDL-SFT-Qwen3-4B-MATH-2B-BASE_1776942366.jsonl` |
| **Checkpoint Dir** | **Meituan dolphinfs** `/mnt/dolphinfs/ssd_pool/.../verl-exp/checkpoints/WDL-SFT-Qwen3-4B-MATH-2B-BASE_1776942366/` (12 FSDP ckpts: steps 25/50/…/300). Not yet transferred. |
| **Best Checkpoint** | **step 75** — online MATH-500 = **73.19%**, AIME-2025 = 7.69%. Step 300 final = 67.74% / 15.38%. |
| **Online Val Progression (MATH-500 / AIME-2025)** | step 0: 30.04% / 0.00% → step 25: 71.77% / 7.69% → step 50: 71.57% / 15.38% → **step 75: 73.19% / 7.69%** → step 100: 72.58% / 11.54% → step 125: 68.55% / 11.54% → step 150: 67.94% / 7.69% → step 175: 67.94% / 15.38% → step 200: 67.14% / 11.54% → step 225: 69.56% / 15.38% → step 250: 68.95% / **19.23%** → step 275: 69.96% / 7.69% → step 300: 67.74% / 15.38% |
| **Training Dynamics** | entropy 19–3647; grad_norm 467–1651 (mid spikes higher than 2A-BASE's 1169 peak — consistent with β>0 adding variance); pg_clipfrac ≤ 0.0021; rollout_corr/kl ≈ 3e-4. reward/mean −0.88 → 0.67 (peak 0.77) — reward trajectory keeps climbing even as online MATH-500 drifts down, indicating the reverse SFT is pushing the policy away from the anchor on training distribution while eroding format/accuracy on val. |
| **Inference** | Pending — critical (this is the m1-style offline test on single-model Base). Offline n=3 vLLM eval needs to probe format compliance (extraction_fail) on MATH-500 to verify whether the β>0 anchor-destruction seen in 1B m1 (ext_fail 37–49%) also shows up here. |
| **W&B** | Project: `OnPolicyWDLSFT`, Run: `WDL-SFT-Qwen3-4B-MATH-2B-BASE_1776942366` (offline; not yet synced) |
| **Status** | **Complete (training) ✓ | Early peak then sustained online drift ✗** — 300 steps with no resumes. MATH-500 peaks early at **step 75 (73.19%)** and drifts down to the 67–69% band for the rest of the run, ending at 67.74% (step 300). **At matched step 300 loses to 2A-BASE by −7.06 pp online** (67.74% vs 74.80%) — the β=0.1 reverse-SFT term hurts a Base-init single model clearly, much more than it hurt 1B on joint (where 1B online tracked 1A within 0.5 pp throughout). |
| **Context** | β>0 isolation on Base init. Plan: `docs/joint_training/plans/active/ablation_single_model.md` §3.3 (2B-base row). **The 1B offline m1 collapse finding now has a single-model analogue at the training level already**: without offline eval we can already see β=0.1 drags online MATH-500 down by 7 pp on Base init, whereas on joint (1B vs 1A) the same β term was invisible online and only surfaced at offline m1. **Tentative implication**: the joint fused-logit rollout absorbs / masks the reverse-SFT damage during training (since model2 benefits from correct-set fusion while model1 absorbs the reverse-set gradient), and removing the joint scaffold exposes the damage directly on the training trajectory. Offline eval pending, but the β>0 abandonment conclusion looks further reinforced. |

---

## Cross-Experiment Comparison (On-Policy WDL-SFT — Online Val, MATH-500 / AIME-2025)

| Experiment | Loss | lr | β | Steps | Best MATH-500 (step) | Best AIME-2025 (step) | Status |
|---|---|---|---|---|---|---|---|
| EXP-12 (M5) | v1 | 1e-6 | 0.1 | ~1043 | 66.94% (step 100) | 15.38% (steps 0–200) | Diverged |
| EXP-13 (M5.5) | v1 | 5e-7 | 0 | 300 complete | 67.94% (step 300) | 26.92% (steps 250, 300) | Stable (v1 baseline) |
| EXP-14 (M5.6) | v1 | 5e-7 | 0.1 | ~458 (1 resume) | 68.15% (step 300) | 26.92% (steps 75, 150) | Stable (peak ≥ M5.5) |
| EXP-15 (LR3) | v1 | 1e-6 | 0 | ~274 | 68.15% (step 125) | 23.08% (step 125) | Peaked early, then diverged |
| **EXP-16 (1A)** | **v2** | **5e-7** | **0** | **300 complete** | **71.37% (step 225)** | **26.92% (step 250)** | **Stable ★ (v2 breaks v1 ceiling, +2.4 pp @ step 300 vs M5.5)** |
| **EXP-17 (1B)** | **v2** | **5e-7** | **0.1** | **300 complete** | **70.97% (step 225 & 275)** | 26.92% (steps 75, 125) | **Training stable; m2 offline MATH-500 82.9% (step 300, EVAL-18); m1 format collapse WORSE than v1 EVAL-15 (ext_fail 37–49% vs 24–28%) — v2 clip does not rescue anchor under β>0** |
| **EXP-18 (1C)** | **v2** | **1e-6** | **0** | **300 complete** | **71.98% (step 150)** | 23.08% (step 125) | **Stable ★ — v2 contains v1 LR3 drift. Offline (EVAL-21/22/23/24): m2 step 150 MATH 82.5%, step 300 78.1% (−4.4 pp drift); online peak translates to offline peak but does NOT exceed 1A 83.1% ceiling. m1 step 300 clean (ext_fail 5–19%, MATH 64.7%, GSM8K 81.9%) — first β=0 v2 m1 baseline.** |
| **ABL-MINIRL-01 (2Z-SFT)** | **minirl (single)** | **5e-7** | **—** | **300 complete** | **70.56% (step 275)** | 23.08% (steps 50/75/225) | **Stable ★ — single-model pure-RL baseline from SFT init. Online peaks within 0.4–0.8 pp of 1A/1B joint v2 peaks. Offline mean@3 (EVAL-25/26, 2026-04-22): step_275 MATH 79.6%, step_300 MATH **80.7%** — trails 1A m2 (83.1%) by −2.4 pp. Joint machinery adds ~2.4 pp over single-model MiniRL from same init.** |
| **ABL-MINIRL-02 (2A-SFT)** | **wdl_sft_is (single)** | **5e-7** | **0** | **300 complete** | **70.4% (step 300)** | 26.92% (step 250) | **Stable ★ — L_loss isolation run (single + `wdl_sft_is` vs 2Z-SFT's pure MiniRL). Offline (EVAL-27/28, 2026-04-23): step 275/300 MATH-500 mean@3 both **80.1%** / pass@3 88.4%/88.6%. **L_loss ≈ 0 on MATH-500** (−0.6 mean@3 vs 2Z step 300). AQUA regression −25 pp at step 300. AMC-2023 pass@3 step 300 **82.5%** sets a new v2-family ceiling. Full joint-vs-single MATH-500 lift collapses onto L_fusion (architectural), not loss.** |
| **ABL-MINIRL-03 (2Z-BASE)** [Meituan] | **minirl (single)** | **5e-7** | **—** | **300 complete** | **76.21% (step 100)** | 19.23% (step 200) | **Meituan AFO run. Online early peak then drift: ramps to 76.21% by step 100, drifts to 64.52% by step 225, recovers to 69.35% by step 300. Base init + MiniRL is LESS stable than SFT init + MiniRL (2Z-SFT was monotone). Offline eval pending (weights on dolphinfs).** |
| **ABL-MINIRL-04 (2G-BASE)** [Meituan] | **vanilla / GRPO (single)** | **5e-7** | **—** | **300 complete** | **71.17% (step 275)** | 19.23% (steps 150/200/225) | **Canonical GRPO (PPO-clip, symmetric 0.2, norm_adv_by_std=True, no IS). Most stable run in Base-init quintet — no drift, 68.5–71.2% band from step 25 onward, end 70.16% (step 300). Lower online ceiling than 2Z-BASE/2A-BASE (71.2% vs 76.2%) but higher step-300 than 2Z-BASE because it avoids drift. Offline pending.** |
| **ABL-MINIRL-05 (2A-BASE)** [Meituan] | **wdl_sft_is (single)** | **5e-7** | **0** | **300 complete** | **76.21% (step 225)** | 19.23% (step 150) | **★ Highest online tail of the Base-init quintet. Monotone climb through step 225, holds 74–75% to step 300 (74.80%). **Beats 2Z-BASE step 300 by +5.45 pp online** (loss contribution on Base init, vs ≈0 pp on SFT init from ABL-MINIRL-02). Also beats 1A step 300 model2-only online (70.36%) by +4.44 pp — treat cautiously pending offline mean@3.** |
| **ABL-MINIRL-06 (2C-BASE)** [Meituan] | **wdl_sft_is (single)** | **1e-6** | **0** | **300 complete** | **74.19% (step 100)** | 23.08% (step 75) | **Doubled lr on Base init: contains drift but loses ceiling. Peak 74.19% (step 100) vs 2A-BASE's 76.21% (step 225) = −2.02 pp; step 300 72.18% vs 74.80% = −2.62 pp. Same sign as 1C vs 1A — lr=5e-7 is the default on Base init too.** |
| **ABL-MINIRL-07 (2B-BASE)** [Meituan] | **wdl_sft_is (single)** | **5e-7** | **0.1** | **300 complete** | **73.19% (step 75)** | 19.23% (step 250) | **β=0.1 on Base single-model: early peak (73.19% step 75) then sustained drift to 67–69% band. Step 300 67.74% = **−7.06 pp vs 2A-BASE** (β=0). The 1B-style β>0 damage — invisible online on joint (1B ≈ 1A within 0.5 pp) — is clearly visible online on single Base. Offline pending; likely anchor/format erosion analogous to 1B m1 EVAL-17/19.** |

**Observations**:
- Forward-only (β=0) at lr=5e-7 is the only setting that produces **monotonically stable** training under both v1 (EXP-13) and v2 (EXP-16) loss.
- Doubling lr to 1e-6 under v1 (EXP-15) ramps faster (hits 68.15% by step 125) but destabilizes after step 150. **EXP-18 (1C) re-tests this under v2 and the v1 collapse does NOT reproduce**: 1C runs the full 300 steps without intervention, peaks at step 150 (71.98% — highest online MATH-500 across all v2 runs), and oscillates in the 67–71% band without monotonic downward drift. End-of-run is weaker than 1A/1B on online MATH-500 (67.34% vs 70.36%). **Offline eval verdict (EVAL-21/23, 2026-04-22)**: online peak translates to offline peak (step 150 m2 MATH-500 82.5%, step 300 m2 78.1% — the 4.4 pp online drop carries through to offline), but **does NOT exceed 1A's 83.1% ceiling** (EVAL-20). The v2 offline m2 ceiling is ~83% regardless of lr ∈ {5e-7, 1e-6}. Higher lr buys an earlier online peak at the cost of late-training drift with zero offline payoff — **1A's lr=5e-7 β=0 remains the default**.
- EXP-14 (β=0.1 at lr=5e-7 under v1) matched EXP-13's peak on model2, suggesting the earlier blanket "reverse SFT unstable" conclusion was really an artifact of lr=1e-6, not of the reverse term itself. (But EVAL-15 showed β>0 destroys model1 format compliance — see INFERENCE_RESULTS.md.)
- **v2 IS-corrected loss (EXP-16) breaks the v1 online AND offline ceiling**: same lr/β/horizon as EXP-13, only diff is `loss_mode=wdl_sft_is`; online peaks at 71.37% (step 225) and lands at 70.36% (step 300) vs M5.5's 67.94% at step 300 (**+2.4 pp online**). Offline mean@3 carries the gain through to multi-sample eval — **EVAL-20** (1A m2 step 225) MATH-500 **83.1%** vs EVAL-12 (M5.5 m2 step 300) 78.6% = **+4.5 pp offline**. The v1 plateau at ~79% MATH-500 mean@3 was loss-bound, not data- or capacity-bound. 1A m1 eval deferred.
- **EXP-17 (1B) — β=0.1 under v2 online-stable BUT offline m1 collapse WORSE than v1**: Online model2-only trajectory tracks 1A within 0.5 pp at all 13 val points; both peak at ~71% around step 225, both end at ~70.4% at step 300. No drift through step 125 (v1's fragile zone). Offline eval (2026-04-21, EVAL-16/17/18/19): **m2 is healthy** (MATH-500 82.5% at step 275, **82.9% at step 300** — matches 1A ceiling) — β=0.1 is a wash on model2 under v2, just as it was under v1. **m1 is more broken than EVAL-15**: extraction_fail 37–49% uniformly (vs v1's 24–28%), MATH-500 38.7%/37.9% (vs v1 M5.6 48.9%). The v2 lower-bound clip does NOT contain the reverse-SFT push-away mechanism on the anchor — it actually leaves the mechanism more room to run. The v1-era "reverse SFT destroys model1" finding is **reinforced**, not refuted, by EXP-17. Forward-only (β=0) remains the recommended default.
- **ABL-MINIRL-01 (2Z-SFT) — online ≈ joint, offline trails joint m2 by ~2.4 pp**: Online MATH-500 peak **70.56%** (step 275) lands within 0.4–0.8 pp of 1A's 71.37% / 1B's 70.97% / 1C's 71.98% online peaks at matched step budget. Step-300 final 70.16% sits 0.2 pp below 1A/1B's 70.36%. Training clean throughout (entropy 1800–2900, no collapse; reward/mean monotonic 0.16 → 0.53). **Offline mean@3 (EVAL-25/26, 2026-04-22)**: step_275 MATH-500 **79.6%** / step_300 **80.7%** — step_300 is the best ABL-MINIRL-01 ckpt and sits **−2.4 pp below 1A step_225 m2 (83.1%, EVAL-20)** and **−2.2 pp below 1B step_300 m2 (82.9%, EVAL-18)**. **Verdict on H3 (init-dominant)**: partially supported — the SFT init gets you ~80% on MATH-500 without any joint machinery, but the last ~2.4 pp ceiling gain requires joint + wdl_sft_is. Online val cannot see this gap (joint and single-model peaks are within 1 pp); only offline mean@3 separates them. Training-cost-adjusted: joint is worth +2.4 pp MATH if you can afford ~2× param memory + extraction overhead, otherwise single-model MiniRL from SFT init is a solid 80% baseline. Late-training drift is also absent here (unlike 1C m2 at lr=1e-6) — lr=5e-7 keeps it monotone.
- **Base-init quintet (ABL-MINIRL-03/04/05/06/07, Meituan AFO, 2026-04-23) — online N=1 only; offline n=3 pending**. All five Base-init runs completed 300 steps on Meituan; weights on dolphinfs, wandb not yet synced. Every number below is trainer `val-core/MATH-500/acc/mean@1` (single-sample, T=1.0, top_p=0.95) — do NOT compare to the offline n=3 mean@3 numbers listed elsewhere in this doc. N=1-comparable reference runs: 1A/1B/1C model2-only online (70.36/70.36/67.34 at step 300), 2A-SFT (70.4), 2Z-SFT (70.16). Against this N=1 reference set:
  - **2A-BASE (wdl_sft_is, β=0, lr=5e-7) is the highest-tail N=1 run across the whole project**: peak 76.21% (step 225), end 74.80% (step 300). Deltas at step 300: **+5.45 pp vs 2Z-BASE, +4.37 pp vs 2A-SFT, +4.44 pp vs 1A (m2-only), +4.64 pp vs 2G-BASE, +7.06 pp vs 2B-BASE**. Peak 76.21% is also the highest online peak in the project (next-best: 1C 71.98% / 1A 71.37% / 2Z-BASE 76.21% briefly at step 100 before drifting). N=1-to-N=1 L_loss on Base init (+5.45 pp) is much larger than L_loss on SFT init (+0.24 pp N=1 / −0.6 pp n=3 offline from ABL-MINIRL-02). Hypothesis: `wdl_sft_is` β=0 is effectively on-policy RFT/rejection-SFT with IS correction; Base init has headroom for the positive-set amplification, SFT init is close to saturated.
  - **H3 "init-dominant" direction is reversed on N=1**: 2A-BASE peak > 2A-SFT peak by +5.81 pp; 2Z-BASE peak > 2Z-SFT peak by +5.65 pp. This could be a genuine Base-wins signal, or it could be a single-sample MATH-500 artifact (Base decoding may happen to produce more valid-format answers at T=1 than SFT decoding). Online N=1 has historically disagreed with offline n=3 in this project (1B m2 online 70.97% → offline n=3 82.9%, +11.9 pp gap), so treat the "Base wins" reading as N=1-only until offline n=3 on 2A-BASE confirms or refutes. **Do not revise §5 of the ablation plan yet.**
  - **β=0.1 is visibly damaging on Base single model online** (unlike on joint): 2B-BASE step 300 = 67.74% online = **−7.06 pp vs 2A-BASE** (same init/loss/lr, only β=0.1 vs β=0). On joint, 1B tracked 1A within 0.5 pp online throughout — the β>0 damage only surfaced at offline m1 (ext_fail 37–49%). Read: joint's fused-logit rollout absorbs the reverse-SFT gradient through model1 while model2 benefits from correct-set fusion; remove the joint scaffold and the damage lands on the val-curve directly. Reinforces β=0 as the default at the N=1 level.
  - **lr=1e-6 on Base (2C-BASE) loses 2.62 pp to lr=5e-7 at step 300 online** (72.18 vs 74.80), same sign as 1C vs 1A. lr=5e-7 is the default regardless of arch/init.
  - **GRPO vs MiniRL vs wdl_sft_is on Base** (all lr=5e-7, β not applicable / β=0):
    - 2G-BASE (vanilla PPO-clip, symmetric 0.2): **most stable** (no drift, 68–71% band from step 25 onward), lowest peak (71.17%), end 70.16%.
    - 2Z-BASE (MiniRL, IS + 0.2/0.27 asymmetric): highest peak (76.21% step 100), but **12 pp online drift** to 64.52% by step 225 before recovering to 69.35%.
    - 2A-BASE (wdl_sft_is β=0, same IS + clip machinery as MiniRL): highest peak AND no drift — ties 2Z-BASE's peak and beats its end by +5.45 pp.
    - Reading: forward-only SFT on the correct set is cleaner than signed-advantage PG when starting from Base, because the Base model's early rollouts contain many incorrect responses whose negative advantage pushes policy in noisy directions. The IS + binary-mask machinery buys headroom over vanilla GRPO, but that headroom is only realized once you also switch off the negative-sample gradient.
- **ABL-MINIRL-02 (2A-SFT) — L_loss isolation says the loss term buys ≈0 on MATH-500**: Online MATH-500 trajectory matches ABL-MINIRL-01 within noise (peak 70.4% at step 300 vs 2Z-SFT's 70.56% at step 275 — same ~70% band). **Offline mean@3 (EVAL-27/28, 2026-04-23)**: step 275 MATH-500 **80.1%** / step 300 **80.1%** (essentially tied); pass@3 88.4% / 88.6%. L_loss = score(2A-SFT) − score(2Z-SFT) at step 300 on MATH-500 mean@3 = **−0.6 pp** (noise); pass@3 = **−0.6 pp**. **The 2.4 pp joint-vs-single MATH-500 gap observed in ABL-MINIRL-01 is therefore NOT attributable to the loss term — it is L_fusion (joint architecture + fused-logit rollout + parameter sharing)**. Measured at step 300: **L_fusion = +3.0 pp mean@3 / +1.6 pp pass@3 on MATH-500** (EVAL-20 1A m2 vs EVAL-28 2A-SFT). Side findings: (1) AQUA regression **−25 pp mean@3 / −21 pp pass@3** in 2A vs 2Z at step 300 — wdl_sft_is single-model hurts multiple-choice discipline, not format-driven (ext_fail comparable); isolate in follow-ups. (2) AMC-2023 pass@3 step 300 **82.5%** is a new v2-family best (previously 80.0% from 1B EVAL-18) — wdl_sft_is broadens hard-math capability but doesn't tighten per-sample consistency. (3) `pass@3 − mean@3` on MATH-500 at 8.3–8.5 pp is identical to 2Z-SFT's 8.5–8.8 pp, so the reasoning-variance tightening from v1 (~8 pp) → v2 joint m2 (~6.5 pp) also collapses onto the architecture, not the loss. **Implied 2×2 ablation to close the decomposition**: joint-arch + MiniRL loss (no wdl_sft_is) would complete the grid and confirm architecture-driven reasoning-variance reduction.

---

## Checkpoint Inventory

| Checkpoint Path | Experiment | Steps Saved | Status |
|---|---|---|---|
| `/data-2/checkpoints/WDL-SFT-Qwen3-4B-MATH-M5-5_1775980322/` | EXP-13 (M5.5) | 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300 | **Deleted 2026-04-20** — all 12 FSDP ckpts pruned after step 300 extracted (`WDL-SFT-4B-MATH-M5-5/step_300{,_model1,_model2}/`) and evaluated (EVAL-12/13) on both machines. |
| `/data-2/checkpoints/WDL-SFT-Qwen3-4B-MATH-M5-6_1776095760/` | EXP-14 (M5.6) | 25, 50, 75, 100, 125, 150, 175, 200, 225, 300, 400 | **Deleted 2026-04-20** — all 11 FSDP ckpts pruned after step 300 extracted (`WDL-SFT-4B-MATH-M5-6/step_300{,_model1,_model2}/`) and evaluated (EVAL-14/15) on both machines. Step 400 not promoted (online MATH-500 67.54% ≤ step 300's 68.15%). |
| `/data-1/checkpoints/WDL-SFT-Qwen3-4B-MATH-LR3_1776359574/` | EXP-15 (LR3) | 25, 50, 75, 100, 125 | **Deleted 2026-04-21** — all 5 remaining FSDP ckpts pruned. step 125 already extracted to `/data-1/model_weights/WDL-SFT-4B-MATH-LR3/step_125{,_model1,_model2}/` and evaluated (EVAL-10/11) on both machines. steps 25/50/75/100 never extracted or referenced. |
| `/data-1/checkpoints/WDL-SFT-Qwen3-4B-MATH-1A_1776594597/` | EXP-16 (1A, v2) | 225 (remaining); 125/150/175/200/250/275/300 deleted 2026-04-21 | step 225 retained as third-redundancy FSDP backup; the extracted step_225 trio is double-mirrored on both machines. Other 7 steps pruned to free /data-1 space for EXP-17 (1B) training. |
| `/data-1/checkpoints/WDL-SFT-Qwen3-4B-MATH-1B_1776695220/` | EXP-17 (1B, v2 β=0.1) | — (all 12 FSDP ckpts deleted 2026-04-21) | **Fully deleted on both machines 2026-04-21.** Training machine: all 12 steps (25/50/…/300, ~1.1 TB). Eval machine: steps 275, 300 (~186 GB; other 10 were never transferred). step_275 and step_300 trios already double-mirrored as extracted weights and evaluated (EVAL-16/17/18/19); other steps never promoted. No FSDP backup retained. Freed 1.1 T locally to unblock EXP-18 (1C) training. |
| `/data-1/checkpoints/MINIRL-Qwen3-4B-MATH-2Z-SFT_1776855436/` | ABL-MINIRL-01 (2Z-SFT) | — (all 12 FSDP ckpts deleted 2026-04-22) | **Fully deleted on both machines 2026-04-22** after EVAL-25/26 complete and merged HF weights double-mirrored (byte-parity verified). Training machine: entire dir ~558 GB freed. Eval machine: steps 275/300 only (~93 GB; other 10 steps were never transferred since they were never promoted). Surviving artifacts: `/data-1/model_weights/MINIRL-Qwen3-4B-MATH-2Z-SFT/step_{275,300}/` (HF weights + `inference_n3/` eval outputs), mirrored both machines. |
| `/data-1/checkpoints/WDL-SFT-Qwen3-4B-MATH-2A-SFT_1776892819/` | ABL-MINIRL-02 (2A-SFT) | — (all 12 FSDP ckpts deleted 2026-04-23) | **Fully deleted on both machines 2026-04-23** after EVAL-27/28 complete and merged HF weights double-mirrored (byte-parity verified). Training machine: entire dir ~558 GB freed. Eval machine: steps 275/300 only (~93 GB; other 10 steps were never transferred since they were never promoted). Surviving artifacts: `/data-1/model_weights/WDL-SFT-Qwen3-4B-MATH-2A-SFT/step_{275,300}/` (HF weights + `inference_n3/` eval outputs), mirrored both machines. |
| **Meituan dolphinfs** `/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yangfengkai02/lgx/verl-exp/checkpoints/MINIRL-Qwen3-4B-MATH-2Z-BASE_1776928234/` | ABL-MINIRL-03 (2Z-BASE) | 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300 | **Live on Meituan AFO dolphinfs (2026-04-23)**; not yet transferred to local `/data-1`. Candidate promotion step: **step 100** (online MATH-500 peak 76.21%). |
| **Meituan dolphinfs** `/mnt/dolphinfs/.../verl-exp/checkpoints/GRPO-Qwen3-4B-MATH-2G-BASE_1776937259/` | ABL-MINIRL-04 (2G-BASE) | 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300 | **Live on Meituan AFO dolphinfs (2026-04-23)**; not yet transferred. Candidate promotion step: **step 275** (online MATH-500 peak 71.17%). |
| **Meituan dolphinfs** `/mnt/dolphinfs/.../verl-exp/checkpoints/WDL-SFT-Qwen3-4B-MATH-2A-BASE_1776937268/` | ABL-MINIRL-05 (2A-BASE) | 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300 | **Live on Meituan AFO dolphinfs (2026-04-23)**; not yet transferred. Candidate promotion steps: **step 225** (online peak 76.21%), **step 150** (second peak 76.01% + best AIME 19.23%), **step 300** (74.80%, end-of-run). |
| **Meituan dolphinfs** `/mnt/dolphinfs/.../verl-exp/checkpoints/WDL-SFT-Qwen3-4B-MATH-2C-BASE_1776938745/` | ABL-MINIRL-06 (2C-BASE) | 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300 | **Live on Meituan AFO dolphinfs (2026-04-23)**; not yet transferred. Candidate promotion step: **step 100** (online peak 74.19%). |
| **Meituan dolphinfs** `/mnt/dolphinfs/.../verl-exp/checkpoints/WDL-SFT-Qwen3-4B-MATH-2B-BASE_1776942366/` | ABL-MINIRL-07 (2B-BASE) | 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300 | **Live on Meituan AFO dolphinfs (2026-04-23)**; not yet transferred. Candidate promotion steps: **step 75** (online peak 73.19% — early peak before β>0 drift), **step 300** (end-of-run 67.74%, for comparison against step 75's post-drift degradation). |

EXP-12 (M5) did not retain its checkpoint directory — discarded after divergence was confirmed.

---

## Extracted Weights

All paths below are mirrored on the Eval machine under the same `/data-1/model_weights/...` path.

| Path | Source | Step | Sub-Model | Notes |
|---|---|---|---|---|
| `/data-1/model_weights/WDL-SFT-4B-MATH-M5-5/step_300/` | EXP-13 | 300 | Joint (both sub-models) | Merged from FSDP shards (17G, intermediate) |
| `/data-1/model_weights/WDL-SFT-4B-MATH-M5-5/step_300_model1/` | EXP-13 | 300 | model1 (weak/anchor, index=0) | Extracted from joint; EVAL-13 results under `inference_n3/` |
| `/data-1/model_weights/WDL-SFT-4B-MATH-M5-5/step_300_model2/` | EXP-13 | 300 | model2 (strong/trainable, index=1) | Extracted from joint; EVAL-12 results under `inference_n3/` |
| `/data-1/model_weights/WDL-SFT-4B-MATH-M5-6/step_300/` | EXP-14 | 300 | Joint (both sub-models) | Merged from FSDP shards (17G, intermediate) |
| `/data-1/model_weights/WDL-SFT-4B-MATH-M5-6/step_300_model1/` | EXP-14 | 300 | model1 (weak/anchor, index=0) | Extracted from joint; EVAL-15 results under `inference_n3/` |
| `/data-1/model_weights/WDL-SFT-4B-MATH-M5-6/step_300_model2/` | EXP-14 | 300 | model2 (strong/trainable, index=1) | Extracted from joint; EVAL-14 results under `inference_n3/` |
| `/data-1/model_weights/WDL-SFT-4B-MATH-LR3/step_125/` | EXP-15 | 125 | Joint (both sub-models) | Merged from FSDP shards |
| `/data-1/model_weights/WDL-SFT-4B-MATH-LR3/step_125_model1/` | EXP-15 | 125 | model1 (weak/anchor, index=0) | Extracted from joint; EVAL-11 results under `inference_n3/` |
| `/data-1/model_weights/WDL-SFT-4B-MATH-LR3/step_125_model2/` | EXP-15 | 125 | model2 (strong/trainable, index=1) | Extracted from joint; EVAL-10 results under `inference_n3/` |
| `/data-1/model_weights/WDL-SFT-4B-MATH-1A/step_225/` | EXP-16 | 225 | Joint (both sub-models) | Merged from FSDP shards on Eval machine (17G) |
| `/data-1/model_weights/WDL-SFT-4B-MATH-1A/step_225_model1/` | EXP-16 | 225 | model1 (weak/anchor, index=0) | Extracted from joint 2026-04-20 (on Eval machine); n=3 offline eval **deferred (low priority)** — no `inference_n3/` subdir on either machine as of 2026-04-22 |
| `/data-1/model_weights/WDL-SFT-4B-MATH-1A/step_225_model2/` | EXP-16 | 225 | model2 (strong/trainable, index=1) | Extracted from joint; **EVAL-20** results under `inference_n3/` (MATH-500 mean@3 = 83.1%) |
| `/data-1/model_weights/WDL-SFT-4B-MATH-1B/step_275/` | EXP-17 | 275 | Joint (both sub-models) | Merged from FSDP shards (17G, intermediate) |
| `/data-1/model_weights/WDL-SFT-4B-MATH-1B/step_275_model1/` | EXP-17 | 275 | model1 (weak/anchor, index=0) | Extracted from joint; EVAL-17 results under `inference_n3/` ⚠⚠ m1 format collapse (MATH-500 38.7%, ext_fail 40–49%) |
| `/data-1/model_weights/WDL-SFT-4B-MATH-1B/step_275_model2/` | EXP-17 | 275 | model2 (strong/trainable, index=1) | Extracted from joint; EVAL-16 results under `inference_n3/` (MATH-500 82.5%) |
| `/data-1/model_weights/WDL-SFT-4B-MATH-1B/step_300/` | EXP-17 | 300 | Joint (both sub-models) | Merged from FSDP shards (17G, intermediate) |
| `/data-1/model_weights/WDL-SFT-4B-MATH-1B/step_300_model1/` | EXP-17 | 300 | model1 (weak/anchor, index=0) | Extracted from joint; EVAL-19 results under `inference_n3/` ⚠⚠ m1 format collapse (MATH-500 37.9%, ext_fail 42–49%) |
| `/data-1/model_weights/WDL-SFT-4B-MATH-1B/step_300_model2/` | EXP-17 | 300 | model2 (strong/trainable, index=1) | Extracted from joint; EVAL-18 results under `inference_n3/` (MATH-500 82.9%) |

---

## Deletion Log

| Date | Item Deleted | Experiment | Reason |
|---|---|---|---|
| 2026-04-20 | `/data-2/checkpoints/WDL-SFT-Qwen3-4B-MATH-M5-5_1775980322/` (all 12 global_step_* FSDP ckpts, ~1.1 TB) | EXP-13 (M5.5) | Step 300 already extracted to `/data-1/model_weights/WDL-SFT-4B-MATH-M5-5/step_300{,_model1,_model2}/` and evaluated on both machines (EVAL-12, EVAL-13). Other 11 steps never extracted or referenced downstream. |
| 2026-04-20 | `/data-2/checkpoints/WDL-SFT-Qwen3-4B-MATH-M5-6_1776095760/` (all 11 global_step_* FSDP ckpts, ~1.0 TB) | EXP-14 (M5.6) | Step 300 already extracted to `/data-1/model_weights/WDL-SFT-4B-MATH-M5-6/step_300{,_model1,_model2}/` and evaluated on both machines (EVAL-14, EVAL-15). Step 400's online MATH-500 (67.54%) was below step 300's (68.15%) — no reason to promote. |
| 2026-04-21 | `/data-1/checkpoints/WDL-SFT-Qwen3-4B-MATH-LR3_1776359574/` (all 5 remaining global_step_* FSDP ckpts, 465 GB) | EXP-15 (LR3) | step 125 already extracted + evaluated (EVAL-10/11), double-mirrored. steps 25/50/75/100 never extracted, not best candidate, never referenced downstream. Freed space for EXP-17 (1B) training. |
| 2026-04-21 | `/data-1/checkpoints/WDL-SFT-Qwen3-4B-MATH-1A_1776594597/global_step_{125,150,175,200,250,275,300}/` (7 FSDP ckpts, 651 GB) | EXP-16 (1A, v2) | step 225 trio double-mirrored on both machines (byte-identical verified); other 7 steps never promoted. Retained `global_step_225/` on training machine as third-redundancy backup. Freed space for EXP-17 (1B) training. |
| 2026-04-21 | **Training machine**: `/data-1/checkpoints/WDL-SFT-Qwen3-4B-MATH-1B_1776695220/` (all 12 global_step_* FSDP ckpts, ~1.1 TB). **Eval machine**: `/data-1/checkpoints/WDL-SFT-Qwen3-4B-MATH-1B_1776695220/global_step_{275,300}/` (2 FSDP ckpts, ~186 GB). | EXP-17 (1B, v2 β=0.1) | step_275 and step_300 trios (joint + model1 + model2) double-mirrored and byte-identical on both machines; EVAL-16 / 17 / 18 / 19 all complete on Eval machine. steps 25/50/…/250 never promoted. Zero FSDP backup retained (unlike 1A step 225) — the extracted weights under `/data-1/model_weights/WDL-SFT-4B-MATH-1B/` are the only surviving artifacts. Freed 1.1T on training machine to unblock EXP-18 (1C) training; freed 186G on Eval machine. |
