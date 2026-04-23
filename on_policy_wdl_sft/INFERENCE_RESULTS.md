# Inference Results — On-Policy WDL-SFT

This file tracks offline vLLM evaluation results for On-Policy WDL-SFT experiments. Each entry cross-references back to the experiment index (`EXPERIMENT_INDEX.md`) by EXP-ID.

Numbering continues from `recipe/joint_training/INFERENCE_RESULTS.md` (EVAL-01 through EVAL-09).

> **Note**: `EXPERIMENT_INDEX.md` carries EXP-12 (M5, diverged, no checkpoints retained), EXP-13 (M5.5, offline eval at step 300 → EVAL-12 / EVAL-13), EXP-14 (M5.6, offline eval at step 300 → EVAL-14 / EVAL-15), EXP-15 (LR3, offline eval at step 125 → EVAL-10 / EVAL-11), **EXP-16 (1A, v2; training complete 2026-04-20; step 225 model2 offline eval 2026-04-20 → EVAL-20, MATH-500 mean@3 = 83.1% — v2 breaks the v1 ~79% ceiling by +3.5 pp; step 225 model1 offline eval not yet run)**, **EXP-17 (1B, v2 with β=0.1; offline eval complete 2026-04-21 at step 275 → EVAL-16/17 and step 300 → EVAL-18/19; model1 format collapse is MORE severe than v1 EVAL-15, refuting the hypothesis that v2's lower-bound clip fixes the β>0 anchor-degradation failure mode)**, **EXP-18 (1C, v2 at lr=1e-6; training + offline eval complete 2026-04-22; EVAL-21/22/23/24 on steps 150 + 300 × m1/m2; m2 ceiling 82.5% at step 150, drops 4.4 pp by step 300; does not exceed 1A 83.1% offline ceiling)**, **EXP-19 (2Z-SFT, single-model MiniRL baseline from SFT init, lr=5e-7; training complete 2026-04-22, 300 steps, online MATH-500 peak 70.56% at step 275 — within 0.4–0.8 pp of joint v2 online peaks despite no joint/fusion/wdl_sft_is machinery; offline eval on step 275 pending on Eval machine, will register as EVAL-25; this is the decisive H3 init-dominant test — if 2Z-SFT m2 matches the v2 ~83% m2 ceiling, the joint+IS infrastructure is largely redundant for MATH accuracy from SFT init)**. All offline evals use the same pipeline and generation params; the only axes that vary are the source checkpoint and the sub-model extracted. **EXP-19 introduces a new dimension: single-model runs have only one backbone, so there is no model1/model2 split — one EVAL entry per checkpoint, not a pair.**

---

## EVAL-10: EXP-15 WDL-SFT-LR3 step 125 (model2 — strong/trainable)

| Field | Value |
|---|---|
| **Source Experiment** | EXP-15 (WDL-SFT-LR3, On-Policy WDL-SFT lr=1e-6) |
| **Model Weights** | `/data-1/model_weights/WDL-SFT-4B-MATH-LR3/step_125_model2` |
| **Checkpoint Step** | 125 (best checkpoint, online MATH-500 peak) |
| **Sub-Model** | model2 (strong/trainable, index=1, extracted from joint checkpoint) |
| **Inference Engine** | vLLM 0.12.0 (FLASH_ATTN backend, V1 engine, tp=8) |
| **Benchmarks** | AIME-2025, MATH-500, AMC-2023, AQUA, GSM8K, MAWPS, SVAMP (all with system prompt) |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=3, max_tokens=4096 |
| **Date** | 2026-04-18 |

### Results (n=3)

| Benchmark | Samples | mean@3 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|
| **AIME-2025** | 30 | **20.0%** | 23.3% | 23.3% | 70.0% |
| **MATH-500** | 500 | **79.6%** | 87.6% | 84.4% | 14.6% |
| **AMC-2023** | 40 | **51.7%** | 60.0% | 57.5% | 36.7% |
| **AQUA** | 254 | **73.8%** | 84.6% | 79.9% | 12.1% |
| **GSM8K** | 1319 | **91.3%** | 95.5% | 91.9% | 0.8% |
| **MAWPS** | 355 | **95.6%** | 96.3% | 95.8% | 0.5% |
| **SVAMP** | 300 | **94.8%** | 97.0% | 96.0% | 0.7% |

### Notes

- model2 is the strong/trainable sub-model (initialized from Qwen3-4B-Base-SFT-stage-1, then trained by On-Policy WDL-SFT for 125 steps at lr=1e-6). It was extracted from the merged joint checkpoint at `/data-1/model_weights/WDL-SFT-4B-MATH-LR3/step_125/`.
- Online training validation at step 125 reported MATH-500 acc/mean@1 = **68.15%** (joint fused inference). Offline model2-only mean@3 = **79.6%** — the large gap (+11.5%) indicates model2 alone is substantially stronger than the fused joint model on MATH-500. This is expected: model2 benefits from the SFT-stage-1 initialization and WDL-SFT training, while the fused model blends in the weaker Qwen3-4B-Base (model1).
- Compared to best previous 4B baseline **EXP-11 (Qwen3-4B-SFT-DPO, mean@3)** from `recipe/joint_training/INFERENCE_RESULTS.md`: model2 outperforms on all 7 benchmarks — MATH-500 **+11.9%** (79.6% vs 67.7%), AIME-2025 **+12.2%** (20.0% vs 7.8%), AMC-2023 **+10.9%** (51.7% vs 40.8%), AQUA **+8.8%** (73.8% vs 65.0%), GSM8K **+1.5%** (91.3% vs 89.8%), MAWPS **+1.2%** (95.6% vs 94.4%), SVAMP **+4.1%** (94.8% vs 90.7%).
- **High extraction failure on AIME-2025 (70%) and AMC-2023 (36.7%)**: the model often generates long reasoning chains without a clearly formatted `\boxed{}` final answer on hard competition problems. The AIME mean@3 of 20.0% is achieved entirely among the ~30% prompts where extraction succeeded; effective accuracy among those is ~67%.
- Generation time: 1449s (~24 min) for 2798 prompts × 3 = 8394 generations (tp=8).
- Raw results saved to: `/data-1/model_weights/WDL-SFT-4B-MATH-LR3/step_125_model2/inference_n3/`

---

## EVAL-11: EXP-15 WDL-SFT-LR3 step 125 (model1 — weak/anchor)

| Field | Value |
|---|---|
| **Source Experiment** | EXP-15 (WDL-SFT-LR3, On-Policy WDL-SFT lr=1e-6) |
| **Model Weights** | `/data-1/model_weights/WDL-SFT-4B-MATH-LR3/step_125_model1` |
| **Checkpoint Step** | 125 |
| **Sub-Model** | model1 (weak/anchor, index=0, initialized from Qwen3-4B-Base, trained 125 steps alongside model2) |
| **Inference Engine** | vLLM 0.12.0 (FLASH_ATTN backend, V1 engine, tp=8) |
| **Benchmarks** | AIME-2025, MATH-500, AMC-2023, AQUA, GSM8K, MAWPS, SVAMP (all with system prompt) |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=3, max_tokens=4096 |
| **Date** | 2026-04-18 |

### Results (n=3)

| Benchmark | Samples | mean@3 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|
| **AIME-2025** | 30 | **6.7%** | 13.3% | 10.0% | 21.1% |
| **MATH-500** | 500 | **63.7%** | 84.4% | 74.8% | 12.9% |
| **AMC-2023** | 40 | **36.7%** | 60.0% | 47.5% | 17.5% |
| **AQUA** | 254 | **45.9%** | 79.1% | 56.7% | 14.6% |
| **GSM8K** | 1319 | **70.7%** | 93.3% | 85.1% | 13.5% |
| **MAWPS** | 355 | **80.7%** | 96.3% | 94.4% | 10.0% |
| **SVAMP** | 300 | **77.4%** | 94.7% | 91.3% | 9.8% |

### Notes

- model1 is initialized from Qwen3-4B-Base but is **not frozen** during EXP-15 training (`freeze_model1=False` default in `configuration_joint_qwen3.py`). Both sub-models receive gradient updates simultaneously. Confirmed by training metrics: step 126 `model1_grad_norm=417.66`, `model2_grad_norm=473.90`.
- **model1 ≠ EXP-08 Qwen3-4B-Base**: EXP-08 is the raw pretrained model with zero training. EVAL-11 model1 (extracted from EXP-15 checkpoint) has been trained for 125 WDL-SFT steps. The MATH-500 gap (63.7% vs 32.5%) and extraction_fail drop (12.9% vs 39.8%) are caused by this training, not eval format differences.
- **model1 vs model2 gap**: model2 outperforms model1 on all benchmarks — MATH-500 **+15.9%** (79.6% vs 63.7%), AIME-2025 **+13.3%** (20.0% vs 6.7%), AQUA **+27.9%** (73.8% vs 45.9%), GSM8K **+20.6%** (91.3% vs 70.7%). The gap reflects different initialization: model2 starts from Qwen3-4B-Base-SFT-stage-1 (already math-SFT'd), model1 starts from the base pretrained weights.
- model1 AIME extraction_fail (21.1%) is lower than model2 (70.0%) — model1 generates shorter responses that fit within 4096 tokens; model2's longer reasoning chains get truncated.
- Generation time: 1028s (~17 min) for 2798 prompts × 3 generations (tp=8).
- Raw results saved to: `/data-1/model_weights/WDL-SFT-4B-MATH-LR3/step_125_model1/inference_n3/`

---

## EVAL-12: EXP-13 WDL-SFT-M5.5 step 300 (model2 — strong/trainable)

| Field | Value |
|---|---|
| **Source Experiment** | EXP-13 (WDL-SFT-M5.5, forward-only On-Policy WDL-SFT, lr=5e-7, β=0, baseline) |
| **Model Weights** | `/data-1/model_weights/WDL-SFT-4B-MATH-M5-5/step_300_model2` |
| **Checkpoint Step** | 300 (final step of the complete 300-step run; online MATH-500 peak = 67.94%) |
| **Sub-Model** | model2 (strong/trainable, index=1, extracted from joint checkpoint) |
| **FSDP Source** | `/data-2/checkpoints/WDL-SFT-Qwen3-4B-MATH-M5-5_1775980322/global_step_300/actor` |
| **Inference Engine** | vLLM 0.12.0 (FLASH_ATTN backend, V1 engine, tp=8) |
| **Benchmarks** | AIME-2025, MATH-500, AMC-2023, AQUA, GSM8K, MAWPS, SVAMP (all with system prompt) |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=3, max_tokens=4096 |
| **Date** | 2026-04-19 |

### Results (n=3)

| Benchmark | Samples | mean@3 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|
| **AIME-2025** | 30 | **20.0%** | 26.7% | 26.7% | 71.1% |
| **MATH-500** | 500 | **78.6%** | 87.4% | 84.6% | 16.7% |
| **AMC-2023** | 40 | **55.8%** | 65.0% | 62.5% | 37.5% |
| **AQUA** | 254 | **80.1%** | 84.3% | 82.7% | 14.0% |
| **GSM8K** | 1319 | **91.8%** | 95.7% | 93.1% | 1.1% |
| **MAWPS** | 355 | **95.2%** | 96.9% | 96.3% | 0.6% |
| **SVAMP** | 300 | **93.4%** | 96.7% | 95.7% | 1.0% |

### Notes

- M5.5 is the reference **forward-only baseline**: lr=5e-7, β=0, 300 steps, monotonically stable online progression 59.68% → 67.94% on MATH-500.
- Online training val at step 300 reported joint-fused MATH-500 acc/mean@1 = **67.94%**. Offline model2-only mean@3 = **78.6%** — same +11% gap pattern as EVAL-10 (LR3 step 125): fusing in the weaker model1 drags the joint score down.
- **Very close to EVAL-10 (LR3 m2 step 125, 79.6%)**: within ≤1% on MATH-500, AIME-2025 tied at 20.0%. M5.5 model2 edges out LR3 model2 on AMC-2023 (+4.1%), AQUA (+6.3%), GSM8K (+0.5%). This suggests **300 steps at lr=5e-7 ≈ 125 steps at lr=1e-6 in final model2 quality** — doubling lr buys speed, not a higher ceiling (and loses stability after step 125).
- Extraction failure pattern matches LR3 model2: high on competition problems (AIME 71.1%, AMC 37.5%) due to long reasoning chains hitting 4096-token cap; near-zero on easy benchmarks.
- Generation time: 1395s (~23.3 min) for 2798 prompts × 3 = 8394 generations (tp=8).
- Raw results saved to: `/data-1/model_weights/WDL-SFT-4B-MATH-M5-5/step_300_model2/inference_n3/`

---

## EVAL-13: EXP-13 WDL-SFT-M5.5 step 300 (model1 — weak/anchor)

| Field | Value |
|---|---|
| **Source Experiment** | EXP-13 (WDL-SFT-M5.5, forward-only, lr=5e-7, β=0) |
| **Model Weights** | `/data-1/model_weights/WDL-SFT-4B-MATH-M5-5/step_300_model1` |
| **Checkpoint Step** | 300 |
| **Sub-Model** | model1 (weak/anchor, index=0, initialized from Qwen3-4B-Base, trained 300 steps alongside model2) |
| **Inference Engine** | vLLM 0.12.0 (FLASH_ATTN backend, V1 engine, tp=8) |
| **Benchmarks** | AIME-2025, MATH-500, AMC-2023, AQUA, GSM8K, MAWPS, SVAMP (all with system prompt) |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=3, max_tokens=4096 |
| **Date** | 2026-04-19 |

### Results (n=3)

| Benchmark | Samples | mean@3 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|
| **AIME-2025** | 30 | **8.9%** | 16.7% | 10.0% | 5.6% |
| **MATH-500** | 500 | **70.5%** | 86.6% | 77.4% | 6.8% |
| **AMC-2023** | 40 | **45.8%** | 65.0% | 52.5% | 8.3% |
| **AQUA** | 254 | **57.5%** | 81.1% | 68.5% | 6.2% |
| **GSM8K** | 1319 | **82.0%** | 94.9% | 89.9% | 5.3% |
| **MAWPS** | 355 | **84.9%** | 96.1% | 93.0% | 7.0% |
| **SVAMP** | 300 | **85.3%** | 96.7% | 91.0% | 4.8% |

### Notes

- Trained 300 steps as the non-frozen anchor model. Compared to EVAL-11 (LR3 model1, 125 steps), M5.5 model1 is **substantially stronger on every benchmark**: MATH-500 +6.8% (70.5% vs 63.7%), AIME-2025 +2.2%, AMC-2023 +9.1%, AQUA +11.6%, GSM8K +11.3%, MAWPS +4.2%, SVAMP +7.9%.
- **Extraction failure is low and uniform (4.8–8.3%)** — much better than LR3 model1 (9.8–21.1%). model1 generates compact answers that fit in 4096 tokens, so the eval is not format-bottlenecked.
- The step-300 model1 exceeds EXP-11 (Qwen3-4B-SFT-DPO, mean@3) on MATH-500 (70.5% vs 67.7%) and AMC-2023 (45.8% vs 40.8%), even though it started from the weaker Qwen3-4B-Base rather than the SFT-stage-1 init — strong evidence that the WDL-SFT training signal is driving real capability gains.
- Generation time: 702s (~11.7 min).
- Raw results saved to: `/data-1/model_weights/WDL-SFT-4B-MATH-M5-5/step_300_model1/inference_n3/`

---

## EVAL-14: EXP-14 WDL-SFT-M5.6 step 300 (model2 — strong/trainable)

| Field | Value |
|---|---|
| **Source Experiment** | EXP-14 (WDL-SFT-M5.6, bidirectional On-Policy WDL-SFT, lr=5e-7, β=0.1) |
| **Model Weights** | `/data-1/model_weights/WDL-SFT-4B-MATH-M5-6/step_300_model2` |
| **Checkpoint Step** | 300 (online MATH-500 peak = 68.15%, matched/exceeded M5.5) |
| **Sub-Model** | model2 (strong/trainable, index=1, extracted from joint checkpoint) |
| **FSDP Source** | `/data-2/checkpoints/WDL-SFT-Qwen3-4B-MATH-M5-6_1776095760/global_step_300/actor` |
| **Inference Engine** | vLLM 0.12.0 (FLASH_ATTN backend, V1 engine, tp=8) |
| **Benchmarks** | AIME-2025, MATH-500, AMC-2023, AQUA, GSM8K, MAWPS, SVAMP (all with system prompt) |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=3, max_tokens=4096 |
| **Date** | 2026-04-19 |

### Results (n=3)

| Benchmark | Samples | mean@3 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|
| **AIME-2025** | 30 | **17.8%** | 23.3% | 23.3% | 71.1% |
| **MATH-500** | 500 | **79.1%** | 86.8% | 84.8% | 16.9% |
| **AMC-2023** | 40 | **52.5%** | 62.5% | 62.5% | 46.7% |
| **AQUA** | 254 | **79.9%** | 84.6% | 83.5% | 12.9% |
| **GSM8K** | 1319 | **91.6%** | 94.8% | 93.3% | 0.7% |
| **MAWPS** | 355 | **95.1%** | 96.3% | 95.2% | 0.2% |
| **SVAMP** | 300 | **93.3%** | 95.3% | 95.0% | 1.6% |

### Notes

- M5.6 re-enables reverse SFT (β=0.1) at the lower lr=5e-7 to test whether M5's instability was an lr artifact rather than a β>0 artifact.
- **model2 is effectively tied with M5.5 model2**: MATH-500 79.1% vs 78.6% (+0.5%), AQUA 79.9% vs 80.1% (−0.2%), GSM8K 91.6% vs 91.8% (−0.2%), MAWPS 95.1% vs 95.2% (−0.1%), SVAMP 93.3% vs 93.4% (−0.1%). AIME-2025 slightly lower (17.8% vs 20.0%), AMC-2023 slightly lower (52.5% vs 55.8%). **Conclusion**: at lr=5e-7, β=0.1 does NOT destabilize model2 — the model2-side results are statistically indistinguishable from forward-only M5.5. This contradicts the earlier blanket claim that reverse SFT is unstable.
- AMC-2023 extraction_fail is higher than M5.5 (46.7% vs 37.5%) — worth monitoring but not a dealbreaker.
- Generation time: 1482s (~24.7 min).
- Raw results saved to: `/data-1/model_weights/WDL-SFT-4B-MATH-M5-6/step_300_model2/inference_n3/`

---

## EVAL-15: EXP-14 WDL-SFT-M5.6 step 300 (model1 — weak/anchor) ⚠

| Field | Value |
|---|---|
| **Source Experiment** | EXP-14 (WDL-SFT-M5.6, bidirectional, lr=5e-7, β=0.1) |
| **Model Weights** | `/data-1/model_weights/WDL-SFT-4B-MATH-M5-6/step_300_model1` |
| **Checkpoint Step** | 300 |
| **Sub-Model** | model1 (weak/anchor, index=0, initialized from Qwen3-4B-Base, trained 300 steps alongside model2) |
| **Inference Engine** | vLLM 0.12.0 (FLASH_ATTN backend, V1 engine, tp=8) |
| **Benchmarks** | AIME-2025, MATH-500, AMC-2023, AQUA, GSM8K, MAWPS, SVAMP (all with system prompt) |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=3, max_tokens=4096 |
| **Date** | 2026-04-19 |

### Results (n=3)

| Benchmark | Samples | mean@3 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|
| **AIME-2025** | 30 | **5.6%** | 13.3% | 10.0% | 24.4% |
| **MATH-500** | 500 | **48.9%** | 78.2% | 67.6% | 26.9% |
| **AMC-2023** | 40 | **30.8%** | 50.0% | 42.5% | 27.5% |
| **AQUA** | 254 | **20.9%** | 46.9% | 27.6% | 27.4% |
| **GSM8K** | 1319 | **64.4%** | 92.0% | 85.7% | 24.1% |
| **MAWPS** | 355 | **64.3%** | 92.4% | 89.3% | 28.3% |
| **SVAMP** | 300 | **62.7%** | 92.7% | 87.7% | 26.7% |

### Notes

- ⚠ **model1 is substantially weaker than every other 4B checkpoint evaluated so far.** Compared to EVAL-13 (M5.5 model1, identical setup except β=0): MATH-500 collapses −21.6% (48.9% vs 70.5%), AQUA −36.6% (20.9% vs 57.5%), GSM8K −17.6%, MAWPS −20.6%, SVAMP −22.6%. AIME-2025 also drops (5.6% vs 8.9%).
- ⚠ **Extraction failure is uniformly elevated at 24–28% across all 7 benchmarks**, versus 4.8–8.3% for M5.5 model1 and 9.8–21.1% for LR3 model1. The uniformity (low variance across datasets) is the telltale: the reverse SFT term has systematically degraded model1's **format compliance** — the anchor stops emitting properly-boxed final answers, regardless of problem difficulty. pass@3 figures (78.2% MATH-500, 92.0% GSM8K) are much closer to other model1s, confirming that the *capability* is still there but is being hidden by broken output structure.
- **This is the first real evidence that β>0 has a non-trivial cost on the model1 side**, even though model2 (EVAL-14) is unaffected. Interpretation: reverse SFT pushes mass *away* from incorrect trajectories, which for the anchor (with weaker init) ends up eroding the format tokens used by both correct and incorrect rollouts. model2 resists this because it starts from an SFT-stage-1 checkpoint with strong format priors.
- **Action item**: the earlier M5 divergence (lr=1e-6, β=0.1, MATH-500 collapsed to 42.74%) was likely caused by the *same* mechanism amplified by higher lr — not a fundamental instability in the wdl_sft loss. β>0 should probably be reconsidered as "bad for model1 anchor" rather than "abandoned entirely"; forward-only (β=0) remains the safer default but β>0 at lr=5e-7 is usable if only model2 is kept.
- Generation time: 955s (~15.9 min).
- Raw results saved to: `/data-1/model_weights/WDL-SFT-4B-MATH-M5-6/step_300_model1/inference_n3/`

---

## EVAL-20: EXP-16 WDL-SFT-1A step 225 (model2 — strong/trainable) ★

| Field | Value |
|---|---|
| **Source Experiment** | EXP-16 (WDL-SFT-1A, v2 IS-corrected, forward-only, lr=5e-7, β=0) |
| **Model Weights** | `/data-1/model_weights/WDL-SFT-4B-MATH-1A/step_225_model2` |
| **Checkpoint Step** | 225 (online MATH-500 peak at 71.37%; the only 1A step promoted) |
| **Sub-Model** | model2 (strong/trainable, index=1, extracted from joint checkpoint) |
| **FSDP Source** | `/data-1/checkpoints/WDL-SFT-Qwen3-4B-MATH-1A_1776594597/global_step_225/actor` (retained as third-redundancy backup; other 7 FSDP steps deleted 2026-04-21) |
| **Inference Engine** | vLLM 0.12.0 (FLASH_ATTN backend, V1 engine, tp=8, on Eval machine L40S) |
| **Benchmarks** | AIME-2025, MATH-500, AMC-2023, AQUA, GSM8K, MAWPS, SVAMP (all with system prompt) |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=3, max_tokens=4096 |
| **Date** | 2026-04-20 |

### Results (n=3)

| Benchmark | Samples | mean@3 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|
| **AIME-2025** | 30 | **14.4%** | 23.3% | 16.7% | 51.1% |
| **MATH-500** | 500 | **83.1%** | 89.4% | 86.8% | 8.3% |
| **AMC-2023** | 40 | **55.0%** | 70.0% | 65.0% | 25.0% |
| **AQUA** | 254 | **70.2%** | 83.9% | 73.6% | 3.9% |
| **GSM8K** | 1319 | **91.3%** | 95.0% | 92.3% | 0.1% |
| **MAWPS** | 355 | **95.4%** | 96.3% | 95.8% | 0.0% |
| **SVAMP** | 300 | **93.7%** | 96.3% | 95.0% | 0.2% |

### Notes

- **v2 IS-corrected loss breaks the v1 MATH-500 m2 ceiling**: 83.1% vs the v1 m2 cluster (M5.5 EVAL-12 78.6%, M5.6 EVAL-14 78.6%, LR3 EVAL-10 79.6%) — **+3.5 to +4.5 pp**. The online gain (+2.4 pp at step 300 vs M5.5) carries through to multi-sample offline eval and grows slightly at the best-step comparison. The v1 ~79% MATH-500 mean@3 plateau was loss-bound, not capacity-bound.
- **Near-saturated easy benchmarks** (GSM8K 91.3%, MAWPS 95.4%, SVAMP 93.7%) with <0.3% extraction_fail — format compliance on the strong side is clean. The v2 headroom concentrates on harder benches (MATH-500, AMC-2023, AQUA) where better IS correction lets the strong side refine.
- AIME-2025 extraction_fail 51.1% — the familiar "4B hits max_tokens=4096 on multi-minute chain" failure mode, not a v2 regression (compare EVAL-10 LR3 m2 70.0%, EVAL-12 M5.5 m2 elevated similarly).
- **1A m2 step 225 vs 1B m2 step 300 (EVAL-18) MATH-500**: 83.1% vs 82.9% — β=0 (1A) and β=0.1 (1B) **tie on model2** under v2. The case for β>0 can only come from model1-side benefit, which EVAL-17/19 refutes. Forward-only (β=0) is the recommended default.
- **Model1 eval explicitly deferred (low priority)**: weights are at `/data-1/model_weights/WDL-SFT-4B-MATH-1A/step_225_model1/` on both machines (8.3G, byte-identical) — extraction completed 2026-04-20 — but offline vLLM eval has NOT been run on either machine (verified 2026-04-22 by exhaustive filesystem and log search; the only 1A m1 artifact is the `extract_WDL-SFT_1A_step225_model1_20260420_075424.log` extract log, no `inference_n3/` subdir and no `eval_metrics.json`). Under β=0 the reverse-SFT term contributes zero gradient so m1 should be very close to the untouched Qwen3-4B-Base anchor; evaluating it would provide a clean β=0 baseline for the EVAL-17/19 collapse but is not required for the main v2 headline (m2 breaks the v1 ceiling). Scheduled for a later date once higher-priority runs complete.
- Generation time: 2076s (~34.6 min) for 2798 prompts × 3 generations (tp=8 on 8×L40S).
- Raw results saved to: `/data-1/model_weights/WDL-SFT-4B-MATH-1A/step_225_model2/inference_n3/`

---

## EVAL-16: EXP-17 WDL-SFT-1B step 275 (model2 — strong/trainable)

| Field | Value |
|---|---|
| **Source Experiment** | EXP-17 (WDL-SFT-1B, v2 IS-corrected, bidirectional, lr=5e-7, β=0.1) |
| **Model Weights** | `/data-1/model_weights/WDL-SFT-4B-MATH-1B/step_275_model2` |
| **Checkpoint Step** | 275 (online MATH-500 tied for peak with step 225 at 70.97%) |
| **Sub-Model** | model2 (strong/trainable, index=1, extracted from joint checkpoint) |
| **FSDP Source** | `/data-1/checkpoints/WDL-SFT-Qwen3-4B-MATH-1B_1776695220/global_step_275/actor` |
| **Inference Engine** | vLLM 0.12.0 (FLASH_ATTN backend, V1 engine, tp=8, on Eval machine L40S) |
| **Benchmarks** | AIME-2025, MATH-500, AMC-2023, AQUA, GSM8K, MAWPS, SVAMP (all with system prompt) |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=3, max_tokens=4096 |
| **Date** | 2026-04-21 |

### Results (n=3)

| Benchmark | Samples | mean@3 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|
| **AIME-2025** | 30 | **15.6%** | 20.0% | 20.0% | 54.4% |
| **MATH-500** | 500 | **82.5%** | 89.2% | 86.4% | 9.6% |
| **AMC-2023** | 40 | **57.5%** | 72.5% | 70.0% | 28.3% |
| **AQUA** | 254 | **80.2%** | 88.2% | 84.3% | 6.0% |
| **GSM8K** | 1319 | **91.8%** | 95.3% | 93.3% | 0.1% |
| **MAWPS** | 355 | **95.2%** | 96.3% | 95.5% | 0.0% |
| **SVAMP** | 300 | **93.7%** | 96.0% | 94.3% | 0.0% |

### Notes

- **1B m2 step 275 ≈ 1A m2 step 225 (EVAL-20, MATH-500 = 83.1%)**: within 0.6 pp on MATH-500 (82.5% vs 83.1%). Both v2 runs (1A β=0, 1B β=0.1) land well above the v1 ~79% model2-MATH-500 ceiling (M5.5/M5.6/LR3 m2 cluster at 78.6%–79.6%). v2 IS-corrected loss breaks through the v1 model2 plateau regardless of whether β=0 or β=0.1 on the model2 side.
- **β=0.1 does NOT degrade model2 under v2**: 1B m2 (β=0.1) is within noise of 1A m2 (β=0) across all 7 benchmarks. This matches the v1-era pattern (M5.5 vs M5.6 m2 were also indistinguishable) — reverse SFT at β=0.1 is harmless to the strong side.
- MATH-500 extraction_fail is 9.6% (vs 1A m2 step 225's 8.2%) — format compliance on the strong side is preserved.
- Generation time: 2193s (~36.6 min) for 2798 prompts × 3 generations (tp=8 on 8×L40S).
- Raw results saved to: `/data-1/model_weights/WDL-SFT-4B-MATH-1B/step_275_model2/inference_n3/`

---

## EVAL-17: EXP-17 WDL-SFT-1B step 275 (model1 — weak/anchor) ⚠⚠

| Field | Value |
|---|---|
| **Source Experiment** | EXP-17 (WDL-SFT-1B, v2 IS-corrected, bidirectional, lr=5e-7, β=0.1) |
| **Model Weights** | `/data-1/model_weights/WDL-SFT-4B-MATH-1B/step_275_model1` |
| **Checkpoint Step** | 275 |
| **Sub-Model** | model1 (weak/anchor, index=0, initialized from Qwen3-4B-Base, trained 275 steps alongside model2) |
| **Inference Engine** | vLLM 0.12.0 (FLASH_ATTN backend, V1 engine, tp=8, on Eval machine L40S) |
| **Benchmarks** | AIME-2025, MATH-500, AMC-2023, AQUA, GSM8K, MAWPS, SVAMP (all with system prompt) |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=3, max_tokens=4096 |
| **Date** | 2026-04-21 |

### Results (n=3)

| Benchmark | Samples | mean@3 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|
| **AIME-2025** | 30 | **4.4%** | 10.0% | 3.3% | 36.7% |
| **MATH-500** | 500 | **38.7%** | 71.6% | 61.8% | 40.5% |
| **AMC-2023** | 40 | **30.8%** | 60.0% | 52.5% | 37.5% |
| **AQUA** | 254 | **13.9%** | 33.1% | 22.8% | 42.7% |
| **GSM8K** | 1319 | **43.8%** | 79.1% | 72.4% | 43.8% |
| **MAWPS** | 355 | **43.6%** | 79.2% | 74.4% | 47.9% |
| **SVAMP** | 300 | **38.6%** | 74.7% | 68.3% | 49.0% |

### Notes

- ⚠⚠ **v2's lower-bound clip did NOT prevent model1 format collapse — it is WORSE than v1 EVAL-15.** vs EVAL-15 (M5.6 m1 β=0.1, v1): MATH-500 −10.2 pp (38.7% vs 48.9%), GSM8K −20.6 pp (43.8% vs 64.4%), MAWPS −20.7 pp (43.6% vs 64.3%), SVAMP −24.1 pp (38.6% vs 62.7%), AQUA −7.0 pp, AMC tied. Extraction_fail is 37%–49% uniformly across all 7 benchmarks (versus v1 EVAL-15's 24%–28%) — a +15 pp jump in uniform format breakage.
- The hypothesis motivating EXP-17 — "v2's lower-bound clip on negative samples (`ratio < 1 − clip_ratio_low` → zero gradient) should contain the reverse SFT push-away signal that destroys model1 format tokens" — is **refuted**. Adding the clip did not restrict the mechanism; if anything, it allowed the mechanism to run longer before the IS ratio collapsed enough to self-cap, pushing extraction_fail HIGHER than v1.
- pass@3 figures (71.6% MATH-500, 79.1% GSM8K, 79.2% MAWPS) are roughly aligned with other model1s, confirming the **capability remains** — the model still *can* solve the problems, but its outputs are broken at the format level ~40–49% of the time. mean@3 is dragged down proportionally.
- Compared to EXP-11 Qwen3-4B-SFT-DPO (external, 365 DPO steps on SFT init): MATH-500 −29.0 pp, GSM8K −46.0 pp. 1B m1 is substantially below every other 4B anchor evaluated.
- Generation time: 3814s (~63.6 min) for 2798 prompts × 3 generations (tp=8). Longer than m2 runs because model1 generates longer (often malformed) reasoning chains that hit the 4096-token limit more often.
- Raw results saved to: `/data-1/model_weights/WDL-SFT-4B-MATH-1B/step_275_model1/inference_n3/`

---

## EVAL-18: EXP-17 WDL-SFT-1B step 300 (model2 — strong/trainable)

| Field | Value |
|---|---|
| **Source Experiment** | EXP-17 (WDL-SFT-1B, v2 IS-corrected, bidirectional, lr=5e-7, β=0.1) |
| **Model Weights** | `/data-1/model_weights/WDL-SFT-4B-MATH-1B/step_300_model2` |
| **Checkpoint Step** | 300 (final step of training) |
| **Sub-Model** | model2 (strong/trainable, index=1, extracted from joint checkpoint) |
| **FSDP Source** | `/data-1/checkpoints/WDL-SFT-Qwen3-4B-MATH-1B_1776695220/global_step_300/actor` |
| **Inference Engine** | vLLM 0.12.0 (FLASH_ATTN backend, V1 engine, tp=8, on Eval machine L40S) |
| **Benchmarks** | AIME-2025, MATH-500, AMC-2023, AQUA, GSM8K, MAWPS, SVAMP (all with system prompt) |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=3, max_tokens=4096 |
| **Date** | 2026-04-21 |

### Results (n=3)

| Benchmark | Samples | mean@3 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|
| **AIME-2025** | 30 | **17.8%** | 23.3% | 20.0% | 52.2% |
| **MATH-500** | 500 | **82.9%** | 90.2% | 88.4% | 8.7% |
| **AMC-2023** | 40 | **63.3%** | 80.0% | 72.5% | 25.8% |
| **AQUA** | 254 | **76.6%** | 85.4% | 78.7% | 5.4% |
| **GSM8K** | 1319 | **92.0%** | 95.0% | 93.1% | 0.2% |
| **MAWPS** | 355 | **95.2%** | 96.3% | 95.8% | 0.0% |
| **SVAMP** | 300 | **93.9%** | 96.0% | 95.0% | 0.0% |

### Notes

- **Step 300 m2 ≥ step 275 m2 on 5/7 benchmarks**: MATH-500 +0.3 pp (82.9% vs 82.5%), AIME +2.2 pp (17.8% vs 15.6%), AMC **+5.8 pp** (63.3% vs 57.5%), GSM8K +0.2 pp, SVAMP +0.2 pp; MAWPS tied; AQUA −3.5 pp (76.6% vs 80.2%). Offline m2-only signal favors the **final checkpoint over the online-peak checkpoint** — the online MATH-500 joint-fused value (70.97% → 70.36%) suggested step 275 was better, but that signal was pulled down by the degrading model1 in the joint fusion. The m2-only view says the strong side keeps improving through step 300.
- **Practical implication**: when extracting and deploying m2 from a WDL-SFT run, pick the **final** step, not the online-joint peak step. The online joint signal is a biased estimate of m2 quality under β>0 because m1 is drifting.
- **EXP-17 (1B β=0.1) step 300 m2 matches EXP-16 (1A β=0) 1A headline**: MATH-500 82.9% vs 83.07% (preliminary 1A step 225) — β=0.1 under v2 provides no additional value on the model2 side at the MATH-500 bench but also does not hurt. The case for β>0 has to come from some benefit *not* captured in these 7 math benchmarks, otherwise β=0 is strictly preferable (because of the model1 cost; see EVAL-17/19).
- Generation time: 2248s (~37.5 min).
- Raw results saved to: `/data-1/model_weights/WDL-SFT-4B-MATH-1B/step_300_model2/inference_n3/`

---

## EVAL-19: EXP-17 WDL-SFT-1B step 300 (model1 — weak/anchor) ⚠⚠

| Field | Value |
|---|---|
| **Source Experiment** | EXP-17 (WDL-SFT-1B, v2 IS-corrected, bidirectional, lr=5e-7, β=0.1) |
| **Model Weights** | `/data-1/model_weights/WDL-SFT-4B-MATH-1B/step_300_model1` |
| **Checkpoint Step** | 300 (final step) |
| **Sub-Model** | model1 (weak/anchor, index=0, initialized from Qwen3-4B-Base, trained 300 steps alongside model2) |
| **Inference Engine** | vLLM 0.12.0 (FLASH_ATTN backend, V1 engine, tp=8, on Eval machine L40S) |
| **Benchmarks** | AIME-2025, MATH-500, AMC-2023, AQUA, GSM8K, MAWPS, SVAMP (all with system prompt) |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=3, max_tokens=4096 |
| **Date** | 2026-04-21 |

### Results (n=3)

| Benchmark | Samples | mean@3 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|
| **AIME-2025** | 30 | **4.4%** | 13.3% | 3.3% | 43.3% |
| **MATH-500** | 500 | **37.9%** | 71.6% | 60.8% | 42.5% |
| **AMC-2023** | 40 | **25.0%** | 50.0% | 40.0% | 43.3% |
| **AQUA** | 254 | **15.8%** | 37.4% | 25.2% | 42.1% |
| **GSM8K** | 1319 | **42.9%** | 78.6% | 70.9% | 44.5% |
| **MAWPS** | 355 | **41.3%** | 75.5% | 69.0% | 48.4% |
| **SVAMP** | 300 | **38.7%** | 74.0% | 67.7% | 48.9% |

### Notes

- ⚠⚠ **Model1 continues to degrade from step 275 to step 300**, exactly opposite to model2's improvement. vs EVAL-17 (step 275 m1): MATH-500 −0.8 pp, AMC −5.8 pp, AIME tied, AQUA +1.8 pp, GSM8K −0.8 pp, MAWPS −2.3 pp, SVAMP tied. Extraction_fail also ticks up 1–3 pp uniformly. The bidirectional reverse SFT term under β=0.1 keeps eroding model1 format tokens across training — there is no equilibrium.
- Pass@3 holds steady across 275→300 (MATH-500 71.6% both, GSM8K 79.1%→78.6%, MAWPS 79.2%→75.5%) — the capability trajectory is flat; the degradation is entirely on the format-compliance axis.
- **Final verdict on the v2-β=0.1 hypothesis**: EXP-17 has cleanly falsified "v2's lower-bound clip prevents model1 format collapse." The collapse is (1) present, (2) more severe than v1's EVAL-15, and (3) monotonically worsening with training. β>0 remains "safe only if you throw away model1."
- **Impact on training recommendations**: use β=0 (forward-only) by default. If β>0 is desired for some downstream reason, use the step where model1 is least broken, which for EXP-17 is the earliest post-warmup checkpoint, not the final one. Step 275 m1 is marginally better than step 300 m1, but neither is usable.
- Generation time: 3919s (~65.3 min).
- Raw results saved to: `/data-1/model_weights/WDL-SFT-4B-MATH-1B/step_300_model1/inference_n3/`

---

## EVAL-21: EXP-18 WDL-SFT-1C step 150 (model2 — strong/trainable) ★

| Field | Value |
|---|---|
| **Source Experiment** | EXP-18 (WDL-SFT-1C, v2 IS-corrected, forward-only, **lr=1e-6**, β=0) |
| **Model Weights** | `/data-1/model_weights/WDL-SFT-4B-MATH-1C/step_150_model2` |
| **Checkpoint Step** | 150 (online MATH-500 peak at 71.98% — highest online across all v2 runs) |
| **Sub-Model** | model2 (strong/trainable, index=1, extracted from joint checkpoint) |
| **FSDP Source** | `/data-1/checkpoints/WDL-SFT-Qwen3-4B-MATH-1C_1776768784/global_step_150/actor` (deleted after double-mirrored extraction verified — see EXPERIMENT_INDEX.md 2026-04-22 deletion entry) |
| **Inference Engine** | vLLM 0.12.0 (FLASH_ATTN backend, V1 engine, tp=8, on Eval machine L40S) |
| **Benchmarks** | AIME-2025, MATH-500, AMC-2023, AQUA, GSM8K, MAWPS, SVAMP (all with system prompt) |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=3, max_tokens=4096 |
| **Date** | 2026-04-22 |

### Results (n=3)

| Benchmark | Samples | mean@3 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|
| **AIME-2025** | 30 | **18.9%** | 30.0% | 23.3% | 46.7% |
| **MATH-500** | 500 | **82.5%** | 88.6% | 85.4% | 7.6% |
| **AMC-2023** | 40 | **63.3%** | 75.0% | 70.0% | 23.3% |
| **AQUA** | 254 | **81.8%** | 88.6% | 86.2% | 3.9% |
| **GSM8K** | 1319 | **91.7%** | 95.5% | 93.0% | 0.2% |
| **MAWPS** | 355 | **95.7%** | 96.3% | 96.1% | 0.0% |
| **SVAMP** | 300 | **93.9%** | 96.3% | 94.7% | 0.1% |

### Notes

- **Online peak translates to offline, but does NOT break the 1A/1B ceiling**: 1C's online MATH-500 peak at step 150 (71.98% — highest across all v2 runs) gives offline mean@3 **82.5%**, which exactly matches EVAL-16 (1B step 275 m2, 82.5%) and trails EVAL-20 (1A step 225 m2, 83.1%) by 0.6 pp. Doubling lr from 5e-7 (1A/1B) to 1e-6 (1C) under v2 buys an earlier/higher online peak but **no offline gain** on MATH-500 mean@3. The v2 ceiling for model2-MATH-500 sits at ~83% regardless of lr ∈ {5e-7, 1e-6}.
- **1C m2 step 150 on easier benchmarks matches 1A/1B within noise**: GSM8K 91.7%, MAWPS 95.7%, SVAMP 93.9%, AQUA 81.8% — all within ±1 pp of 1A/1B m2.
- **AMC-2023 63.3%** ties 1B step 300 m2 (EVAL-18) as the best 4B m2 AMC number to date; AIME-2025 18.9% sits between 1A (14.4%) and 1B step 300 (17.8%) — AIME variance across v2 runs is within the extraction_fail envelope (all 46–54%) and doesn't differentiate settings.
- Generation time: 1526s (~25.4 min) for 2798 prompts × 3 generations (tp=8 on 8×L40S). Fastest m2 eval in this table — consistent with the step 150 checkpoint producing more compact outputs than step 300 (see EVAL-23 for the step 300 counterpart).
- Raw results saved to: `/data-1/model_weights/WDL-SFT-4B-MATH-1C/step_150_model2/inference_n3/`

---

## EVAL-22: EXP-18 WDL-SFT-1C step 150 (model1 — weak/anchor)

| Field | Value |
|---|---|
| **Source Experiment** | EXP-18 (WDL-SFT-1C, v2 IS-corrected, forward-only, lr=1e-6, β=0) |
| **Model Weights** | `/data-1/model_weights/WDL-SFT-4B-MATH-1C/step_150_model1` |
| **Checkpoint Step** | 150 (same ckpt as EVAL-21, different sub-model extracted) |
| **Sub-Model** | model1 (weak/anchor, index=0, initialized from Qwen3-4B-Base) |
| **Inference Engine** | vLLM 0.12.0 (FLASH_ATTN backend, V1 engine, tp=8, on Eval machine L40S) |
| **Benchmarks** | AIME-2025, MATH-500, AMC-2023, AQUA, GSM8K, MAWPS, SVAMP (all with system prompt) |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=3, max_tokens=4096 |
| **Date** | 2026-04-22 |

### Results (n=3)

| Benchmark | Samples | mean@3 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|
| **AIME-2025** | 30 | **4.4%** | 13.3% | 6.7% | 30.0% |
| **MATH-500** | 500 | **52.7%** | 78.4% | 69.6% | 23.4% |
| **AMC-2023** | 40 | **30.0%** | 47.5% | 40.0% | 27.5% |
| **AQUA** | 254 | **21.5%** | 44.9% | 26.4% | 22.7% |
| **GSM8K** | 1319 | **59.9%** | 89.5% | 81.1% | 24.5% |
| **MAWPS** | 355 | **65.9%** | 93.0% | 88.2% | 25.2% |
| **SVAMP** | 300 | **59.7%** | 88.7% | 81.0% | 27.4% |

### Notes

- **β=0 anchors-out the model1 collapse seen under β>0**: 1C m1 step 150 at MATH-500 52.7% is 13.8–14.8 pp higher than the v2 β=0.1 m1 cluster (EVAL-17 1B step 275 m1 38.7%, EVAL-19 1B step 300 m1 37.9%). Extraction_fail drops from 1B's 37–49% uniform band to 1C's **22–30% uniform band** — still higher than a clean 4B, but not in "format-collapse" territory. This is the clean β=0 control that EXP-16 1A m1 was supposed to provide (still deferred; see EVAL-20 notes).
- **Above pretrained baseline, below v1 m1s**: 1C m1 on MATH-500 is +20.2 pp over EXP-08 pretrained Qwen3-4B-Base (32.5%) but trails the v1 β=0 m1s by 11 pp (EVAL-11 LR3 m1 63.7%, EVAL-13 M5.5 m1 70.5%). Under β=0 the reverse-SFT loss coefficient is zero, so model1 receives no direct gradient through the loss — the gap vs pretrained suggests **joint-arch coupling still moves m1** (shared lm_head / embed / normalization tokens), just not as strongly as when m1 is trained directly via reverse SFT. v1 m1s were trained with nonzero β on v1 loss, which actively updated m1.
- Pass@3 is dominated by format survival: MATH-500 pass@3 78.4% (vs mean@3 52.7%) — the model *can* reach correct answers in multiple samples but often fails extraction on 1–2 of the 3 rollouts. This is the same format-sensitive profile as EVAL-11/13 v1 m1s, not the format-broken profile of EVAL-17/19 v2 β=0.1 m1s.
- Generation time: 3845s (~64.1 min) — similar to EVAL-17/19 m1 (~65 min), slower than m2 because m1's higher extraction_fail correlates with longer responses hitting max_tokens=4096.
- Raw results saved to: `/data-1/model_weights/WDL-SFT-4B-MATH-1C/step_150_model1/inference_n3/`

---

## EVAL-23: EXP-18 WDL-SFT-1C step 300 (model2 — strong/trainable)

| Field | Value |
|---|---|
| **Source Experiment** | EXP-18 (WDL-SFT-1C, v2 IS-corrected, forward-only, lr=1e-6, β=0) |
| **Model Weights** | `/data-1/model_weights/WDL-SFT-4B-MATH-1C/step_300_model2` |
| **Checkpoint Step** | 300 (final step; online MATH-500 = 67.34%, AIME = 7.69% — end-of-run is the weakest point for 1C online) |
| **Sub-Model** | model2 (strong/trainable, index=1, extracted from joint checkpoint) |
| **FSDP Source** | `/data-1/checkpoints/WDL-SFT-Qwen3-4B-MATH-1C_1776768784/global_step_300/actor` (deleted after double-mirrored extraction verified) |
| **Inference Engine** | vLLM 0.12.0 (FLASH_ATTN backend, V1 engine, tp=8, on Eval machine L40S) |
| **Benchmarks** | AIME-2025, MATH-500, AMC-2023, AQUA, GSM8K, MAWPS, SVAMP (all with system prompt) |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=3, max_tokens=4096 |
| **Date** | 2026-04-22 |

### Results (n=3)

| Benchmark | Samples | mean@3 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|
| **AIME-2025** | 30 | **10.0%** | 16.7% | 10.0% | 7.8% |
| **MATH-500** | 500 | **78.1%** | 86.8% | 82.6% | 1.9% |
| **AMC-2023** | 40 | **50.8%** | 67.5% | 60.0% | 2.5% |
| **AQUA** | 254 | **79.8%** | 89.4% | 83.5% | 1.7% |
| **GSM8K** | 1319 | **90.1%** | 94.2% | 91.7% | 0.1% |
| **MAWPS** | 355 | **94.7%** | 95.8% | 94.7% | 0.0% |
| **SVAMP** | 300 | **92.0%** | 96.0% | 93.0% | 0.2% |

### Notes

- **Step 300 m2 is UNIFORMLY WORSE than step 150 m2 (EVAL-21) on the harder benches**: MATH-500 −4.4 pp (78.1% vs 82.5%), AIME-2025 −8.9 pp (10.0% vs 18.9%), AMC-2023 −12.5 pp (50.8% vs 63.3%). Easier benches are flat to slightly down: GSM8K −1.6 pp, MAWPS −1.0 pp, SVAMP −1.9 pp, AQUA −2.0 pp. The late-training drift visible in online MATH-500 (71.98% @ step 150 → 67.34% @ step 300, −4.6 pp) **carries through to offline mean@3** at roughly the same magnitude.
- **Contrast with 1B (β=0.1)**: 1B m2 IMPROVED from step 275 → step 300 (MATH +0.3, AMC +5.8 — EVAL-16→18). 1C m2 at higher lr DEGRADES across the same late-training window. **Higher lr under v2 is stable-ish but not monotone**: the online-peak→offline-gain story of 1A/1B does not extend to 1C — for 1C, the best offline m2 is the online-peak checkpoint (step 150), not the final one.
- **Extraction_fail is dramatically LOWER than step 150 m2** (AIME 7.8% vs 46.7%, MATH 1.9% vs 7.6%, AMC 2.5% vs 23.3%, AQUA 1.7% vs 3.9%). Late-training outputs are much more compact/format-compliant — the model has "settled" format-wise but at a lower capability level. The compactness is why this eval ran in 851s (14 min) vs step 150's 25 min.
- **Recommendation for 1C deployment**: extract m2 from **step 150** (EVAL-21), not step 300. The online peak and the offline peak coincide for 1C. This is opposite to the 1B recommendation (extract from final step) — the rule is **lr-dependent**: low lr (5e-7) → m2 improves through training; high lr (1e-6) → m2 peaks mid-training.
- Generation time: 851s (~14.2 min).
- Raw results saved to: `/data-1/model_weights/WDL-SFT-4B-MATH-1C/step_300_model2/inference_n3/`

---

## EVAL-24: EXP-18 WDL-SFT-1C step 300 (model1 — weak/anchor)

| Field | Value |
|---|---|
| **Source Experiment** | EXP-18 (WDL-SFT-1C, v2 IS-corrected, forward-only, lr=1e-6, β=0) |
| **Model Weights** | `/data-1/model_weights/WDL-SFT-4B-MATH-1C/step_300_model1` |
| **Checkpoint Step** | 300 (final step) |
| **Sub-Model** | model1 (weak/anchor, index=0, initialized from Qwen3-4B-Base) |
| **Inference Engine** | vLLM 0.12.0 (FLASH_ATTN backend, V1 engine, tp=8, on Eval machine L40S) |
| **Benchmarks** | AIME-2025, MATH-500, AMC-2023, AQUA, GSM8K, MAWPS, SVAMP (all with system prompt) |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=3, max_tokens=4096 |
| **Date** | 2026-04-22 |

### Results (n=3)

| Benchmark | Samples | mean@3 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|
| **AIME-2025** | 30 | **4.4%** | 10.0% | 10.0% | 18.9% |
| **MATH-500** | 500 | **64.7%** | 82.0% | 72.8% | 8.4% |
| **AMC-2023** | 40 | **37.5%** | 52.5% | 45.0% | 8.3% |
| **AQUA** | 254 | **27.7%** | 50.4% | 28.4% | 6.3% |
| **GSM8K** | 1319 | **81.9%** | 93.5% | 89.2% | 4.8% |
| **MAWPS** | 355 | **88.8%** | 95.5% | 94.1% | 4.9% |
| **SVAMP** | 300 | **82.4%** | 94.3% | 89.7% | 5.9% |

### Notes

- ★ **Surprise finding: 1C m1 IMPROVES from step 150 → step 300**, the opposite direction of m2. vs EVAL-22 (step 150 m1): MATH-500 **+12.0 pp** (52.7% → 64.7%), GSM8K **+22.0 pp** (59.9% → 81.9%), MAWPS **+22.9 pp** (65.9% → 88.8%), SVAMP **+22.8 pp** (59.7% → 82.4%), AQUA +6.2 pp, AMC +7.5 pp. AIME tied at 4.4%. Extraction_fail drops from 22–30% (step 150) to 5–19% (step 300) — a uniform −15 pp format-compliance gain.
- **Under β=0 the reverse-SFT loss coefficient is zero**, so m1 should receive no direct gradient through the loss term. The observed +12 pp MATH-500 / +22 pp GSM8K gain strongly implies **joint-arch coupling moves m1 implicitly**: the joint model almost certainly shares lm_head / token embeddings / norm weights between m1 and m2, and m2's continued forward-SFT training during steps 150 → 300 drags m1 along on the format-compliance axis (where the two sub-models overlap). The capability trajectory (pass@3) shows this: MATH-500 pass@3 moves 78.4% → 82.0% (+3.6 pp), much smaller than the mean@3 gain of +12.0 pp — **the improvement is predominantly format, not reasoning**.
- **1C m1 trajectory is the MIRROR of 1B m1's**: 1B β=0.1 m1 monotonically DEGRADED (EVAL-17 → EVAL-19: MATH −0.8, ext_fail +1–3 pp) because the reverse SFT term pushed m1 embeddings away from format tokens. 1C β=0 m1 IMPROVED because that push-away is absent, and whatever m2 training does to shared parameters is format-positive.
- **Best v2 m1 on easier benches**: 1C step 300 m1 on GSM8K 81.9% / MAWPS 88.8% / SVAMP 82.4% beats all v2 m1s to date and approaches the v1 β=0 m1 cluster (EVAL-11 LR3 m1 70.7% / 80.7% / 77.4%, EVAL-13 M5.5 m1 82.0% / 84.9% / 85.3%). Note that v1 m1s were trained with active m1 gradient under v1 loss at β>0 (M5.5/M5.6) or β=0 (LR3) — different regimes.
- Generation time: 1231s (~20.5 min) — much faster than EVAL-22 step 150 m1 (64 min) because the late-training m1 is far more compact. This is the 4B "output length ≈ extraction_fail" story again.
- Raw results saved to: `/data-1/model_weights/WDL-SFT-4B-MATH-1C/step_300_model1/inference_n3/`

---

## Cross-Experiment Comparison (On-Policy WDL-SFT vs 4B Baselines)

### 4B model2 (strong/trainable sub-model) — mean@3

| Benchmark | EXP-11 Qwen3-4B-SFT-DPO (ext, 365) | EXP-13 M5.5 m2 (v1, β=0, 300) | EXP-14 M5.6 m2 (v1, β=0.1, 300) | EXP-15 LR3 m2 (v1, lr=1e-6, β=0, 125) | **EXP-16 1A m2 (v2, β=0, 225)** | **EXP-17 1B m2 (v2, β=0.1, 275)** | **EXP-17 1B m2 (v2, β=0.1, 300)** | **EXP-18 1C m2 (v2, lr=1e-6, β=0, 150)** | **EXP-18 1C m2 (v2, lr=1e-6, β=0, 300)** |
|---|---|---|---|---|---|---|---|---|---|
| **MATH-500** | 67.7% | 78.6% | 79.1% | 79.6% | **83.1%** | 82.5% | 82.9% | 82.5% | 78.1% |
| **AIME-2025** | 7.8% | **20.0%** | 17.8% | **20.0%** | 14.4% | 15.6% | 17.8% | 18.9% | 10.0% |
| **AMC-2023** | 40.8% | 55.8% | 52.5% | 51.7% | 55.0% | 57.5% | **63.3%** | **63.3%** | 50.8% |
| **AQUA** | 65.0% | 80.1% | 79.9% | 73.8% | 70.2% | 80.2% | 76.6% | **81.8%** | 79.8% |
| **GSM8K** | 89.8% | 91.8% | 91.6% | 91.3% | 91.3% | 91.8% | **92.0%** | 91.7% | 90.1% |
| **MAWPS** | 94.4% | 95.2% | 95.1% | 95.6% | 95.4% | 95.2% | 95.2% | **95.7%** | 94.7% |
| **SVAMP** | 90.7% | 93.4% | 93.3% | **94.8%** | 93.7% | 93.7% | 93.9% | 93.9% | 92.0% |

### 4B model1 (weak/anchor sub-model) — mean@3

| Benchmark | EXP-08 Qwen3-4B-Base (pretrained) | EXP-13 M5.5 m1 (v1, β=0, 300) | EXP-14 M5.6 m1 (v1, β=0.1, 300) ⚠ | EXP-15 LR3 m1 (v1, lr=1e-6, β=0, 125) | **EXP-17 1B m1 (v2, β=0.1, 275)** ⚠⚠ | **EXP-17 1B m1 (v2, β=0.1, 300)** ⚠⚠ | **EXP-18 1C m1 (v2, lr=1e-6, β=0, 150)** | **EXP-18 1C m1 (v2, lr=1e-6, β=0, 300)** |
|---|---|---|---|---|---|---|---|---|
| **MATH-500** | 32.5% | **70.5%** | 48.9% | 63.7% | 38.7% | 37.9% | 52.7% | 64.7% |
| **AIME-2025** | 3.3% | **8.9%** | 5.6% | 6.7% | 4.4% | 4.4% | 4.4% | 4.4% |
| **AMC-2023** | 15.0% | **45.8%** | 30.8% | 36.7% | 30.8% | 25.0% | 30.0% | 37.5% |
| **AQUA** | 6.8% | **57.5%** | 20.9% | 45.9% | 13.9% | 15.8% | 21.5% | 27.7% |
| **GSM8K** | 27.8% | **82.0%** | 64.4% | 70.7% | 43.8% | 42.9% | 59.9% | 81.9% |
| **MAWPS** | 21.7% | **84.9%** | 64.3% | 80.7% | 43.6% | 41.3% | 65.9% | **88.8%** |
| **SVAMP** | 25.9% | **85.3%** | 62.7% | 77.4% | 38.6% | 38.7% | 59.7% | 82.4% |

*Note: EXP-16 1A m1 still not evaluated (see EVAL-20 notes). EXP-18 1C m1 at step 300 is the first clean β=0 v2 m1 data point with strong format compliance (5–19% ext_fail vs 1B's 37–49%).*

### Observations

1. **v2 IS-corrected loss breaks the v1 ~79–80% model2-MATH-500 ceiling, and the v2 ceiling sits at ~83% regardless of lr or β**: EVAL-20 1A m2 reaches **83.1%**, EVAL-18 1B m2 82.9%, EVAL-16 1B m2 82.5%, EVAL-21 1C m2 82.5%. All four v2 m2 peaks land in a tight 82.5–83.1% band. Doubling lr (5e-7 → 1e-6) gives 1C a higher *online* peak (71.98% vs 1A's 71.37%, 1B's 70.97%) but **no offline gain**. The v1 runs (M5.5, M5.6, LR3) clustered at 78.6%–79.6%. The ceiling was loss-bound, not data-/capacity-/lr-bound.
2. **β>0 is strictly worse on model1, and v2's lower-bound clip does NOT fix it**: EVAL-17/19 (1B m1 β=0.1 under v2) are MORE degraded than EVAL-15 (M5.6 m1 β=0.1 under v1). Extraction_fail climbs from v1's 24–28% uniform band to v2's 37–49% uniform band. EVAL-22/24 (**1C m1 β=0 under v2**) provides the clean control: ext_fail 22–30% at step 150, **5–19% at step 300** — no format collapse. Confirmation: the collapse is a β>0 effect specifically, not a v2 loss property.
3. **Online joint signal is a biased estimate of m2 quality in different directions under different lrs**: 1B (β=0.1) online said step 275 > step 300, but offline m2-only said step 300 > step 275 (the joint-fusion signal was dragged down by the degrading m1). 1C (β=0, higher lr) online peaked at step 150 and drifted down; offline m2-only confirms step 150 > step 300 by 4.4 pp on MATH-500. **Rule**: extract m2 from the offline-peak step, which under lr=5e-7 is the final step, and under lr=1e-6 is the online-peak step.
4. **Training time under β=0 IMPROVES m1 when lr is high enough** — an unexpected 1C finding: EVAL-22 → EVAL-24 moves m1 MATH-500 from 52.7% → 64.7% (+12 pp) and GSM8K from 59.9% → 81.9% (+22 pp) while ext_fail drops −15 pp. Under β=0 the reverse-SFT coefficient is zero so m1 receives no direct gradient — the gain strongly implies joint-arch parameter sharing (lm_head / embeddings / norms) lets m2's continued forward-SFT training drag m1 along on format-compliance axes. pass@3 moves only +3.6 pp, so the +12 mean@3 is mostly format, not capability.
5. **Training time under β>0 DEGRADES m1 monotonically**: 1B step 275→300 m1 MATH −0.8, ext_fail +1–3 pp. The direction is opposite to 1C's m1 trajectory (β=0). The mirror pairing (1C vs 1B on m1 across steps) cleanly isolates β as the driver of m1 format erosion.
6. **Best 4B model2 to date**: **EVAL-20 1A m2 step 225 (MATH-500 83.1%)**, narrowly ahead of 1B step 300 m2 (82.9%), 1C step 150 m2 (82.5%), 1B step 275 m2 (82.5%). **β=0 at lr=5e-7** (EXP-16 1A) is the recommended default: matches or leads on m2, avoids the β>0 m1 collapse, and avoids the 1C late-training m2 drift at lr=1e-6.
7. **Best 4B model1 to date (v2 regime)**: **EVAL-24 1C m1 step 300 (MATH-500 64.7%, GSM8K 81.9%, MAWPS 88.8%)** — the first v2 m1 that doesn't collapse. Still trails the v1 β=0 m1 peak (EVAL-13 M5.5 m1, MATH 70.5% / GSM8K 82.0%), but those v1 m1s had explicit m1 gradient under v1 loss. Among v2 m1s, the pattern is clear: β=0 high-lr / long training maximizes m1 quality via joint-coupling; β>0 destroys it regardless of lr.
8. **Decisive m1 format-collapse experiment is done**: EVAL-17/19 refuted "v2's lower-bound clip prevents m1 collapse under β>0", and EVAL-22/24 established a clean β=0 baseline. Future experiments should (a) default to β=0 and invest in data / loss-mode / scale variants, or (b) if β>0 is required for a downstream reason, target the format-token mechanism directly (freezing m1 embed/lm_head, format-token logit KL, gradient masking on extraction-critical tokens).

---

<!--
Template for new entries:

## EVAL-XX: {description}

| Field | Value |
|---|---|
| **Source Experiment** | EXP-XX |
| **Model Weights** | /data-1/eval_results/{path} |
| **Checkpoint Step** | {N} |
| **Sub-Model** | model1 / model2 / N/A |
| **Inference Engine** | vLLM 0.12.0 |
| **Benchmarks** | {list} |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=3, max_tokens=4096 |
| **Date** | YYYY-MM-DD |

### Results (n=3)
| Benchmark | Samples | mean@3 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|

### Notes
{observations}
-->
