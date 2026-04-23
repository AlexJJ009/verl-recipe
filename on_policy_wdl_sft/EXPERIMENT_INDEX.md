# Experiment Index — On-Policy WDL-SFT

This table tracks all training experiments for the On-Policy WDL-SFT branch, their logs, checkpoints, and evaluation results.

Branch: `feature/on-policy-wdl-sft`

> **Note on numbering**: EXP-01 through EXP-11 live in `recipe/joint_training/EXPERIMENT_INDEX.md` (joint-training parent branch). EXP-12 through EXP-18 are the joint-model On-Policy WDL-SFT runs, ordered chronologically. EXP-12–15 used v1 loss (`loss_mode=wdl_sft`); EXP-16–18 are v2 runs (`loss_mode=wdl_sft_is`, IS/clip-corrected). EXP-19 onward is the single-model ablation series (2X, plan: `docs/joint_training/plans/active/ablation_single_model.md`) — same data budget / config as paired joint runs, standard single Qwen3 backbone instead of `QwenJointForCausalLM`. Short smoke-test / launch-failure runs (the un-suffixed initial run, M5.7 dataset-missing failure, M5.8 killed at step 3, the 1A first-launch attempt at `_1776591102`, and the 1B first-launch attempt at `_1776683653` killed at step 26 before the val-path semantics were re-confirmed) are intentionally omitted — their logs remain on disk or have been cleaned.

> **On "eval" semantics**: "Online Val Progression" refers to the trainer's `val-core/...acc/mean@1` metrics logged every validation step during training (MATH-500 500 samples, AIME-2025 26 samples, n=1). For **v2 runs (EXP-16 onward)**, `ray_trainer._validate()` invokes `checkpoint_manager.update_weights(eval_only=True)` → `extract_sub_model_weights(sub_model_index=1)`, so the online val is **model2-only** (not fused). Verified in 1A/1B logs via `[WDL-SFT VERIFY] extracting model2-only weights`. For v1 runs (EXP-12–15) the online val reported in this index is joint-fused. Offline vLLM evaluation (n=3 across 7 benchmarks on extracted model1/model2) has been performed for EXP-15 step 125 (EVAL-10 / EVAL-11), EXP-13 step 300 (EVAL-12 / EVAL-13), EXP-14 step 300 (EVAL-14 / EVAL-15), EXP-16 step 225 model2 (**EVAL-20**, MATH-500 mean@3 = 83.1% — v2 breaks v1 ~79% ceiling), EXP-17 step 275 / step 300 (EVAL-16/17 and EVAL-18/19), and **EXP-18 step 150 / step 300 (EVAL-21/22/23/24, 2026-04-22)** — step 150 m2 MATH-500 = 82.5%, step 300 m2 = 78.1% (offline confirms online peak→drift shape; does not exceed 1A's 83.1% ceiling). **EXP-16 step 225 model1 offline eval is explicitly deferred (low priority)** — extraction done on both machines 2026-04-20, but no inference run; the clean β=0 anchor baseline is now provided by EVAL-22/24 (1C m1) instead.

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

## EXP-19: MINIRL-2Z-SFT — Single-Model MiniRL Baseline from SFT Init (lr=5e-7) ★

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
| **Inference** | **Pending** — step 275 extraction + offline vLLM eval to be run on Eval machine (L40S). Will be registered as EVAL-25 once results land. |
| **W&B** | Project: `OnPolicyWDLSFT`, Run: `MINIRL-Qwen3-4B-MATH-2Z-SFT_1776855436` (offline) |
| **Status** | **Complete & stable ✓ (training) | offline eval pending** — first 2X run, pure-RL reference floor. Online MATH-500 peak 70.56% (step 275) lands within 0.4–0.8 pp of 1A/1B v2 joint peaks (71.37% / 70.97%) and 0.4 pp below 1C's online peak (71.98%). **Training-level H3 signal is strong**: if the joint + wdl_sft_is machinery were the main driver of 1X's online lift, 2Z-SFT (single + minirl, same SFT init) should be meaningfully worse. It isn't. The loss-vs-init decomposition must wait on offline mean@3 — 2Z-SFT step 275 m2 vs 1A step 225 m2 (EVAL-20, MATH 83.1%) is the decisive apples-to-apples. |
| **Context** | Reference floor for the 2X series. Plan: `docs/joint_training/plans/active/ablation_single_model.md` §5 — $L_\text{loss}$ = score(2A-SFT) − score(2Z-SFT), interpreted together with $L_\text{fusion}$ = score(1A) − score(2A-SFT). |

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
| **EXP-19 (2Z-SFT)** | **minirl (single)** | **5e-7** | **—** | **300 complete** | **70.56% (step 275)** | 23.08% (steps 50/75/225) | **Stable ★ — single-model pure-RL baseline from SFT init. Online peak within 0.4–0.8 pp of 1A/1B joint v2 peaks at matched lr. Training-level evidence for H3 (init-dominant). Offline eval on step 275 pending — EVAL-25 will decide.** |

**Observations**:
- Forward-only (β=0) at lr=5e-7 is the only setting that produces **monotonically stable** training under both v1 (EXP-13) and v2 (EXP-16) loss.
- Doubling lr to 1e-6 under v1 (EXP-15) ramps faster (hits 68.15% by step 125) but destabilizes after step 150. **EXP-18 (1C) re-tests this under v2 and the v1 collapse does NOT reproduce**: 1C runs the full 300 steps without intervention, peaks at step 150 (71.98% — highest online MATH-500 across all v2 runs), and oscillates in the 67–71% band without monotonic downward drift. End-of-run is weaker than 1A/1B on online MATH-500 (67.34% vs 70.36%). **Offline eval verdict (EVAL-21/23, 2026-04-22)**: online peak translates to offline peak (step 150 m2 MATH-500 82.5%, step 300 m2 78.1% — the 4.4 pp online drop carries through to offline), but **does NOT exceed 1A's 83.1% ceiling** (EVAL-20). The v2 offline m2 ceiling is ~83% regardless of lr ∈ {5e-7, 1e-6}. Higher lr buys an earlier online peak at the cost of late-training drift with zero offline payoff — **1A's lr=5e-7 β=0 remains the default**.
- EXP-14 (β=0.1 at lr=5e-7 under v1) matched EXP-13's peak on model2, suggesting the earlier blanket "reverse SFT unstable" conclusion was really an artifact of lr=1e-6, not of the reverse term itself. (But EVAL-15 showed β>0 destroys model1 format compliance — see INFERENCE_RESULTS.md.)
- **v2 IS-corrected loss (EXP-16) breaks the v1 online AND offline ceiling**: same lr/β/horizon as EXP-13, only diff is `loss_mode=wdl_sft_is`; online peaks at 71.37% (step 225) and lands at 70.36% (step 300) vs M5.5's 67.94% at step 300 (**+2.4 pp online**). Offline mean@3 carries the gain through to multi-sample eval — **EVAL-20** (1A m2 step 225) MATH-500 **83.1%** vs EVAL-12 (M5.5 m2 step 300) 78.6% = **+4.5 pp offline**. The v1 plateau at ~79% MATH-500 mean@3 was loss-bound, not data- or capacity-bound. 1A m1 eval deferred.
- **EXP-17 (1B) — β=0.1 under v2 online-stable BUT offline m1 collapse WORSE than v1**: Online model2-only trajectory tracks 1A within 0.5 pp at all 13 val points; both peak at ~71% around step 225, both end at ~70.4% at step 300. No drift through step 125 (v1's fragile zone). Offline eval (2026-04-21, EVAL-16/17/18/19): **m2 is healthy** (MATH-500 82.5% at step 275, **82.9% at step 300** — matches 1A ceiling) — β=0.1 is a wash on model2 under v2, just as it was under v1. **m1 is more broken than EVAL-15**: extraction_fail 37–49% uniformly (vs v1's 24–28%), MATH-500 38.7%/37.9% (vs v1 M5.6 48.9%). The v2 lower-bound clip does NOT contain the reverse-SFT push-away mechanism on the anchor — it actually leaves the mechanism more room to run. The v1-era "reverse SFT destroys model1" finding is **reinforced**, not refuted, by EXP-17. Forward-only (β=0) remains the recommended default.
- **EXP-19 (2Z-SFT) — single-model MiniRL from SFT init is training-level competitive with joint v2**: Online MATH-500 peak **70.56%** (step 275) lands within 0.4–0.8 pp of 1A's 71.37% / 1B's 70.97% / 1C's 71.98% online peaks at matched step budget. Step-300 final 70.16% sits 0.2 pp below 1A/1B's 70.36%. Training is clean throughout (entropy 1800–2900, no collapse; reward/mean monotonic 0.16 → 0.53). This is **training-level support for H3 (init-dominant)** from the plan: if the joint-model + fused-logit + wdl_sft_is machinery were the main driver of 1X's online lift, 2Z-SFT (same SFT init, pure RL, single model) should be meaningfully worse — it isn't. The decisive test is offline mean@3 on step 275: if 2Z-SFT m2 matches the v2 m2 ceiling (~83%, EVAL-20), the joint+IS infrastructure is largely redundant for MATH accuracy. If 2Z-SFT m2 stays in the v1 cluster (~79%), the joint+IS loss is doing real work that online val can't see. Result pending — will register as EVAL-25.

---

## Checkpoint Inventory

| Checkpoint Path | Experiment | Steps Saved | Status |
|---|---|---|---|
| `/data-2/checkpoints/WDL-SFT-Qwen3-4B-MATH-M5-5_1775980322/` | EXP-13 (M5.5) | 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300 | **Deleted 2026-04-20** — all 12 FSDP ckpts pruned after step 300 extracted (`WDL-SFT-4B-MATH-M5-5/step_300{,_model1,_model2}/`) and evaluated (EVAL-12/13) on both machines. |
| `/data-2/checkpoints/WDL-SFT-Qwen3-4B-MATH-M5-6_1776095760/` | EXP-14 (M5.6) | 25, 50, 75, 100, 125, 150, 175, 200, 225, 300, 400 | **Deleted 2026-04-20** — all 11 FSDP ckpts pruned after step 300 extracted (`WDL-SFT-4B-MATH-M5-6/step_300{,_model1,_model2}/`) and evaluated (EVAL-14/15) on both machines. Step 400 not promoted (online MATH-500 67.54% ≤ step 300's 68.15%). |
| `/data-1/checkpoints/WDL-SFT-Qwen3-4B-MATH-LR3_1776359574/` | EXP-15 (LR3) | 25, 50, 75, 100, 125 | **Deleted 2026-04-21** — all 5 remaining FSDP ckpts pruned. step 125 already extracted to `/data-1/model_weights/WDL-SFT-4B-MATH-LR3/step_125{,_model1,_model2}/` and evaluated (EVAL-10/11) on both machines. steps 25/50/75/100 never extracted or referenced. |
| `/data-1/checkpoints/WDL-SFT-Qwen3-4B-MATH-1A_1776594597/` | EXP-16 (1A, v2) | 225 (remaining); 125/150/175/200/250/275/300 deleted 2026-04-21 | step 225 retained as third-redundancy FSDP backup; the extracted step_225 trio is double-mirrored on both machines. Other 7 steps pruned to free /data-1 space for EXP-17 (1B) training. |
| `/data-1/checkpoints/WDL-SFT-Qwen3-4B-MATH-1B_1776695220/` | EXP-17 (1B, v2 β=0.1) | — (all 12 FSDP ckpts deleted 2026-04-21) | **Fully deleted on both machines 2026-04-21.** Training machine: all 12 steps (25/50/…/300, ~1.1 TB). Eval machine: steps 275, 300 (~186 GB; other 10 were never transferred). step_275 and step_300 trios already double-mirrored as extracted weights and evaluated (EVAL-16/17/18/19); other steps never promoted. No FSDP backup retained. Freed 1.1 T locally to unblock EXP-18 (1C) training. |
| `/data-1/checkpoints/MINIRL-Qwen3-4B-MATH-2Z-SFT_1776855436/` | EXP-19 (2Z-SFT) | 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300 | **Retained pending eval.** Single-model FSDP ckpts (~47 GB each, 12 ckpts ≈ 564 GB total). step 275 (online peak) scheduled for extraction + transfer to Eval machine for offline vLLM eval (EVAL-25). Non-promoted steps to be pruned after EVAL-25 completes. |

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
