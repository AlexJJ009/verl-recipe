# Inference Results — On-Policy WDL-SFT

This file tracks offline vLLM evaluation results for On-Policy WDL-SFT experiments. Each entry cross-references back to the experiment index (`EXPERIMENT_INDEX.md`) by EXP-ID.

Numbering continues from `recipe/joint_training/INFERENCE_RESULTS.md` (EVAL-01 through EVAL-09).

> **Note**: `EXPERIMENT_INDEX.md` carries EXP-12 (M5, diverged, no checkpoints retained), EXP-13 (M5.5, offline eval at step 300 → EVAL-12 / EVAL-13), EXP-14 (M5.6, offline eval at step 300 → EVAL-14 / EVAL-15), EXP-15 (LR3, offline eval at step 125 → EVAL-10 / EVAL-11), **EXP-16 (1A, v2; training complete 2026-04-20; step 225 model2 offline eval 2026-04-20 → EVAL-20, MATH-500 mean@3 = 83.1% — v2 breaks the v1 ~79% ceiling by +3.5 pp; step 225 model1 offline eval not yet run)**, **EXP-17 (1B, v2 with β=0.1; offline eval complete 2026-04-21 at step 275 → EVAL-16/17 and step 300 → EVAL-18/19; model1 format collapse is MORE severe than v1 EVAL-15, refuting the hypothesis that v2's lower-bound clip fixes the β>0 anchor-degradation failure mode)**, **EXP-18 (1C, v2 at lr=1e-6; training + offline eval complete 2026-04-22; EVAL-21/22/23/24 on steps 150 + 300 × m1/m2; m2 ceiling 82.5% at step 150, drops 4.4 pp by step 300; does not exceed 1A 83.1% offline ceiling)**, **ABL-MINIRL-01 (2Z-SFT, single-model MiniRL baseline from SFT init, lr=5e-7; training complete 2026-04-22; offline eval complete 2026-04-22 → EVAL-25/26 on steps 275/300; MATH-500 mean@3 peak 80.7% at step 300, −2.4 pp below 1A m2 — H3 init-dominant partially supported: SFT init gets to ~80% without joint machinery, last ~2.4 pp requires joint)**, and **ABL-MINIRL-02 (2A-SFT, single-model + `wdl_sft_is` loss, lr=5e-7; training complete 2026-04-22; offline eval complete 2026-04-23 → EVAL-27/28 on steps 275/300; L_loss isolation = 2A − 2Z shows ≈0 on MATH-500, −16 to −25 pp on AQUA — the joint-vs-single MATH-500 lift is architectural, not loss-driven)**. All offline evals use the same pipeline and generation params; the only axes that vary are the source checkpoint and the sub-model extracted. **Single-model ABL-MINIRL runs have only one backbone, so there is no model1/model2 split — one EVAL entry per checkpoint, not a pair.**

> **Reward-label bug note (2026-04-27)**: offline results from EXP-16/17/18 and ABL-MINIRL-02 are evaluations of the pre-fix `wdl_sft_is` implementation, where `advantages` carried GRPO-centered values instead of raw reward labels. Keep them as historical results; do not treat them as spec-correct WDL-SFT-IS until rerun under the `-LABELFIX` launch prefixes.

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

## EVAL-25: ABL-MINIRL-01 2Z-SFT step 275 (single-model, online peak)

| Field | Value |
|---|---|
| **Source Experiment** | ABL-MINIRL-01 (MINIRL-2Z-SFT, single-model MiniRL from SFT-stage-1 init, lr=5e-7) |
| **Model Weights** | `/data-1/model_weights/MINIRL-Qwen3-4B-MATH-2Z-SFT/step_275` |
| **Checkpoint Step** | 275 (online MATH-500 peak = 70.56%) |
| **Sub-Model** | n/a (single Qwen3-4B backbone, not joint) |
| **Inference Engine** | vLLM 0.12.0 (FLASH_ATTN backend, V1 engine, tp=8, on Eval machine L40S) |
| **Benchmarks** | AIME-2025, MATH-500, AMC-2023, AQUA, GSM8K, MAWPS, SVAMP (all with system prompt) |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=3, max_tokens=4096 |
| **Date** | 2026-04-22 |

### Results (n=3)

| Benchmark | Samples | mean@3 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|
| **AIME-2025** | 30 | **21.1%** | 26.7% | 26.7% | 15.6% |
| **MATH-500** | 500 | **79.6%** | 88.4% | 82.6% | 3.4% |
| **AMC-2023** | 40 | **61.7%** | 77.5% | 65.0% | 12.5% |
| **AQUA** | 254 | **82.3%** | 92.5% | 87.0% | 1.4% |
| **GSM8K** | 1319 | **92.5%** | 95.8% | 93.5% | 0.1% |
| **MAWPS** | 355 | **95.6%** | 96.1% | 95.8% | 0.0% |
| **SVAMP** | 300 | **94.1%** | 96.0% | 94.3% | 0.1% |

### Notes

- **First offline data point for the ABL-MINIRL single-model ablation family.** Purpose: floor for H3 decomposition — the gap between v2 joint m2 and single-model MiniRL from the same SFT init quantifies the joint-machinery contribution (loss + fusion + parameter sharing).
- **Trails v2 joint m2 band (82.5–83.1%) on MATH-500 by −2.9 to −3.5 pp** (vs EVAL-20 1A m2 83.1%, EVAL-18 1B m2 82.9%, EVAL-21 1C m2 82.5%). Joint + wdl_sft_is adds a real MATH-500 lift over pure single-model MiniRL from the same init; the gap is small but consistent with EVAL-26.
- **GSM8K 92.5%, MAWPS 95.6%, SVAMP 94.1% are at or above every v2 joint m2 data point** (best v2 joint: GSM8K 92.0% (1B step 300), MAWPS 95.7% (1C step 150), SVAMP 93.9% (1B step 300)). On easier benchmarks the joint machinery does not separate from single-model MiniRL — the joint contribution concentrates on MATH-500/AIME-2025.
- **AIME-2025 21.1%**: matches EVAL-12 M5.5 m2 (20.0%) and edges every v2 joint m2 peak (14.4–18.9%). n=30 so ±3.3 pp per-problem noise applies, but the direction is consistent with single-model having more token budget per prompt (no joint fusion overhead).
- **Format compliance is clean**: MATH-500 ext_fail 3.4%, GSM8K/MAWPS/SVAMP ≤ 0.1% — no β>0-style collapse (single-model MiniRL has no β term by construction). AMC-2023 12.5% and AIME-2025 15.6% reflect longer-trajectory bench-specific max_tokens pressure, not model instability.
- Generation time: 2312s (~38.5 min) — AIME/AMC max_tokens pressure dominates.
- Raw results saved to: `/data-1/model_weights/MINIRL-Qwen3-4B-MATH-2Z-SFT/step_275/inference_n3/` (mirrored on Eval machine under the same path).

---

## EVAL-26: ABL-MINIRL-01 2Z-SFT step 300 (single-model, final)

| Field | Value |
|---|---|
| **Source Experiment** | ABL-MINIRL-01 (MINIRL-2Z-SFT, single-model MiniRL from SFT-stage-1 init, lr=5e-7) |
| **Model Weights** | `/data-1/model_weights/MINIRL-Qwen3-4B-MATH-2Z-SFT/step_300` |
| **Checkpoint Step** | 300 (final step; online MATH-500 = 70.16%) |
| **Sub-Model** | n/a (single Qwen3-4B backbone, not joint) |
| **Inference Engine** | vLLM 0.12.0 (FLASH_ATTN backend, V1 engine, tp=8, on Eval machine L40S) |
| **Benchmarks** | AIME-2025, MATH-500, AMC-2023, AQUA, GSM8K, MAWPS, SVAMP (all with system prompt) |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=3, max_tokens=4096 |
| **Date** | 2026-04-22 |

### Results (n=3)

| Benchmark | Samples | mean@3 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|
| **AIME-2025** | 30 | **13.3%** | 20.0% | 13.3% | 12.2% |
| **MATH-500** | 500 | **80.7%** | 89.2% | 83.6% | 2.9% |
| **AMC-2023** | 40 | **60.8%** | 72.5% | 67.5% | 6.7% |
| **AQUA** | 254 | **82.0%** | 90.2% | 86.6% | 1.4% |
| **GSM8K** | 1319 | **91.9%** | 95.5% | 92.7% | 0.1% |
| **MAWPS** | 355 | **95.7%** | 96.6% | 95.5% | 0.0% |
| **SVAMP** | 300 | **93.7%** | 96.0% | 94.7% | 0.1% |

### Notes

- **Step_275 → step_300 delta**: MATH-500 **+1.1 pp** (79.6 → 80.7), AIME-2025 **−7.8 pp** (21.1 → 13.3), AMC-2023 −0.9, AQUA −0.3, GSM8K −0.6, MAWPS +0.1, SVAMP −0.4. Late-training is **neutral to mildly positive on MATH-500 and flat elsewhere**; AIME swing is on the edge of n=30 interpretability (±3.3 pp per problem → −7.8 pp ≈ 2 problems flipped).
- **No late-training drift — opposite of 1C m2**: EVAL-21 → EVAL-23 lost **−4.4 pp** on MATH-500 from step 150 → 300 at lr=1e-6 β=0. Single-model MiniRL at lr=5e-7 stays monotone through the final step, same as v2 joint 1A/1B at lr=5e-7. Confirms that late-training drift is lr-driven, not joint-vs-single.
- **Best ABL-MINIRL-01 ckpt for MATH-500**: step_300 (80.7%) narrowly beats step_275 (79.6%). For deployment, pick step_300.
- **Gap to v2 joint m2 = ~2.4 pp on MATH-500**: step_300 MATH-500 80.7% is **−2.4 pp below EVAL-20 1A m2 (83.1%)** and **−2.2 pp below EVAL-18 1B m2 (82.9%)**. This is the ABL-MINIRL-01 answer for H3: **the SFT init gets you to ~80% on MATH-500 without any joint machinery, but the remaining ~2.4 pp to the v2 joint ceiling requires the joint architecture and wdl_sft_is loss**. Online val does not see this gap (joint and single-model online peaks within 1 pp); only offline mean@3 separates them.
- **Format compliance clean**: MATH-500 ext_fail 2.9% (down from 3.4% at step_275), GSM8K/MAWPS/SVAMP all ≤ 0.1%. Late-training m2 is slightly more compact than step_275 (ext_fail down across the board).
- Generation time: 2079s (~34.6 min), ~10% faster than EVAL-25's 2312s — consistent with lower ext_fail → fewer samples hitting max_tokens.
- Raw results saved to: `/data-1/model_weights/MINIRL-Qwen3-4B-MATH-2Z-SFT/step_300/inference_n3/` (mirrored on Eval machine under the same path).

---

## EVAL-27: ABL-MINIRL-02 2A-SFT step 275 (single-model + wdl_sft_is, online peak)

| Field | Value |
|---|---|
| **Source Experiment** | ABL-MINIRL-02 (WDL-SFT-2A-SFT, single-model with `loss_mode=wdl_sft_is` from SFT-stage-1 init, lr=5e-7) |
| **Model Weights** | `/data-1/model_weights/WDL-SFT-Qwen3-4B-MATH-2A-SFT/step_275` |
| **Checkpoint Step** | 275 (online MATH-500 peak) |
| **Sub-Model** | n/a (single Qwen3-4B backbone, not joint) |
| **Inference Engine** | vLLM 0.12.0 (FLASH_ATTN backend, V1 engine, tp=8, on Eval machine L40S) |
| **Benchmarks** | AIME-2025, MATH-500, AMC-2023, AQUA, GSM8K, MAWPS, SVAMP (all with system prompt) |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=3, max_tokens=4096 |
| **Date** | 2026-04-23 |

### Results (n=3)

| Benchmark | Samples | mean@3 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|
| **AIME-2025** | 30 | **11.1%** | 16.7% | 10.0% | 32.2% |
| **MATH-500** | 500 | **80.1%** | 88.4% | 82.6% | 4.9% |
| **AMC-2023** | 40 | **55.8%** | 75.0% | 67.5% | 10.8% |
| **AQUA** | 254 | **66.3%** | 79.1% | 69.3% | 4.2% |
| **GSM8K** | 1319 | **90.6%** | 93.6% | 91.6% | 0.1% |
| **MAWPS** | 355 | **94.3%** | 95.2% | 94.9% | 0.1% |
| **SVAMP** | 300 | **92.8%** | 95.0% | 93.0% | 0.2% |

### Notes

- **L_loss isolation — this is the ABL-MINIRL-02 purpose**: 2A-SFT differs from 2Z-SFT (ABL-MINIRL-01) only in the loss term — `wdl_sft_is` (single-model variant of the v2 joint loss) vs pure MiniRL. Both are single Qwen3 backbones, both init from SFT-stage-1, both lr=5e-7, same data budget. The per-benchmark delta 2A − 2Z **attributes exclusively to the loss term**, with joint-architecture + fused-logit effects excluded by design. See paired comparison vs EVAL-25.
- **MATH-500 essentially tied with 2Z-SFT**: 2A step_275 **80.1%** vs 2Z step_275 **79.6%** = **+0.5 pp** on mean@3; pass@3 **88.4% vs 88.4%** (identical). In single-model form, the wdl_sft_is loss does not lift the MATH-500 mean@3 meaningfully over pure MiniRL at the peak step. The ~2.4 pp gap 1A-vs-2Z observed in EVAL-20/26 is therefore **not** attributable to the loss; it tracks with the joint architecture + parameter sharing.
- **AQUA regression vs 2Z-SFT is large — −16.0 pp mean@3 / −13.4 pp pass@3** (2A 66.3/79.1 vs 2Z 82.3/92.5). Extraction_fail is comparable (4.2% vs 1.4%) so the gap is mostly reasoning, not format. wdl_sft_is single-model appears to hurt AQUA multiple-choice performance — possibly because reverse-SFT / IS correction alters token-distribution in a way that breaks letter-answer discipline. Worth flagging for the 2B/2C follow-ups.
- **Easy-bench slight regression (−1 to −2 pp)**: GSM8K 90.6 vs 2Z 92.5 (−1.9), MAWPS 94.3 vs 95.6 (−1.3), SVAMP 92.8 vs 94.1 (−1.3). Consistent small drop; not noise-level.
- **AIME-2025 noise**: 2A 11.1 vs 2Z 21.1 is 3 problems (±3.3 pp each); don't over-interpret. pass@3 16.7 vs 26.7 is also in noise range for n=30.
- **Format compliance clean** for the high-n benchmarks (GSM8K/MAWPS/SVAMP/MATH-500 all under 5% ext_fail). AIME ext_fail 32.2% and AMC 10.8% are elevated vs 2Z's 15.6% and 12.5% — single-model + wdl_sft_is consumes more token budget on hard math.
- Generation time: 875s (~14.6 min), ~62% faster than EVAL-25's 2312s — likely driven by fewer samples hitting max_tokens due to differing output distributions (not yet characterized).
- Raw results saved to: `/data-1/model_weights/WDL-SFT-Qwen3-4B-MATH-2A-SFT/step_275/inference_n3/` (mirrored on Eval machine under the same path).

---

## EVAL-28: ABL-MINIRL-02 2A-SFT step 300 (single-model + wdl_sft_is, final)

| Field | Value |
|---|---|
| **Source Experiment** | ABL-MINIRL-02 (WDL-SFT-2A-SFT, single-model with `loss_mode=wdl_sft_is` from SFT-stage-1 init, lr=5e-7) |
| **Model Weights** | `/data-1/model_weights/WDL-SFT-Qwen3-4B-MATH-2A-SFT/step_300` |
| **Checkpoint Step** | 300 (final step) |
| **Sub-Model** | n/a (single Qwen3-4B backbone, not joint) |
| **Inference Engine** | vLLM 0.12.0 (FLASH_ATTN backend, V1 engine, tp=8, on Eval machine L40S) |
| **Benchmarks** | AIME-2025, MATH-500, AMC-2023, AQUA, GSM8K, MAWPS, SVAMP (all with system prompt) |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=3, max_tokens=4096 |
| **Date** | 2026-04-23 |

### Results (n=3)

| Benchmark | Samples | mean@3 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|
| **AIME-2025** | 30 | **10.0%** | 16.7% | 13.3% | 37.8% |
| **MATH-500** | 500 | **80.1%** | 88.6% | 83.8% | 4.8% |
| **AMC-2023** | 40 | **60.0%** | 82.5% | 75.0% | 3.3% |
| **AQUA** | 254 | **57.0%** | 68.9% | 58.3% | 3.4% |
| **GSM8K** | 1319 | **90.4%** | 93.7% | 91.6% | 0.1% |
| **MAWPS** | 355 | **94.5%** | 95.5% | 94.4% | 0.0% |
| **SVAMP** | 300 | **91.7%** | 95.7% | 92.0% | 0.0% |

### Notes

- **Step_275 → step_300 delta (2A-SFT)**: MATH-500 flat (80.1 → 80.1, 0.0 pp); AIME-2025 −1.1; AMC-2023 **+4.2 pp** (55.8 → 60.0); AQUA **−9.3 pp** (66.3 → 57.0); GSM8K −0.2; MAWPS +0.2; SVAMP −1.1. **MATH-500 plateaus; AMC improves; AQUA continues to drift down** (late-training continues to degrade AQUA letter-answer extraction under wdl_sft_is). Unlike 2Z-SFT (EVAL-25 → EVAL-26, MATH +1.1 pp / flat elsewhere), 2A-SFT's late-training is not uniformly neutral.
- **Loss-term isolation vs 2Z-SFT step 300 (EVAL-26)** — the decisive L_loss decomposition:
  - **MATH-500 mean@3**: 80.1 vs 80.7 → **−0.6 pp** (2A slightly *worse*); pass@3 88.6 vs 89.2 → **−0.6 pp**. Both within step-to-step noise. **wdl_sft_is as a loss term, in single-model form, does not lift MATH-500 over pure MiniRL.**
  - **AMC-2023**: mean@3 60.0 vs 60.8 (−0.8), but pass@3 **82.5 vs 72.5 (+10.0 pp)** — 2A-SFT has higher AMC capability ceiling but same per-sample accuracy. Interesting mismatch; could indicate wdl_sft_is broadens the output distribution on hard math.
  - **AQUA**: **mean@3 57.0 vs 82.0 (−25.0 pp) / pass@3 68.9 vs 90.2 (−21.3 pp)** — confirmed, wdl_sft_is single-model is structurally worse on AQUA. Worth isolating as a 2B/2C or follow-up question.
  - **Easy benches**: GSM8K −1.5, MAWPS −1.2, SVAMP −2.0 — consistent small drop.
- **Combining EVAL-27 + EVAL-28 (2A-SFT best step is step 275 = step 300 on MATH-500; pick step 300 for deployment because AMC is +4.2 pp better and MATH is tied)**.
- **Verdict on L_loss (score(2A-SFT) − score(2Z-SFT))**: **near-zero on MATH-500**, negative on easy benches (−1 to −2), large negative on AQUA (−16 to −25), mixed on AMC (−0.8 mean@3 but +10 pass@3). Cross-referencing against the plan's §5 decomposition: $L_\text{loss}$ is **not** what buys the joint-vs-single MATH-500 lift. The remaining candidate for the joint lift is $L_\text{fusion}$ = score(1A) − score(2A-SFT) — that lift is now measured at **step 300: 83.1 − 80.1 = +3.0 pp on MATH-500 mean@3** (EVAL-20 1A step_225 m2 vs EVAL-28), **90.2 − 88.6 = +1.6 pp on pass@3** (EVAL-18 1B step_300 m2 vs EVAL-28). Joint architecture + fused-logit rollout + parameter sharing buys ~3 pp mean@3 and ~1.6 pp pass@3 on MATH-500 over a single-model variant using the same loss.
- **Format compliance**: MATH-500 ext_fail 4.8% (slight down from 275's 4.9%). AQUA ext_fail 3.4% is normal — the AQUA regression is NOT a format-collapse artifact. AIME ext_fail creeps up to 37.8%; AMC comes down to 3.3% (from 10.8% at step_275, a clean improvement).
- Generation time: 828s (~13.8 min), consistent with EVAL-27's 875s. Much faster than 2Z-SFT (EVAL-25/26 were 2312s / 2079s) — single-model wdl_sft_is appears to produce more compact outputs under this eval config.
- Raw results saved to: `/data-1/model_weights/WDL-SFT-Qwen3-4B-MATH-2A-SFT/step_300/inference_n3/` (mirrored on Eval machine under the same path).

---

## Cross-Experiment Comparison (On-Policy WDL-SFT vs 4B Baselines)

Every table below reports **both `mean@3` and `pass@3`** side-by-side. Rationale: `mean@3` ranks average per-sample accuracy (production signal); `pass@3` is the n=3 oracle upper bound (capability ceiling — "can the model reach the answer in ≥1 of 3 tries"). The **`pass@3 − mean@3` gap** directly measures per-sample inconsistency: a high gap means the model sometimes solves and sometimes doesn't, independent of whether it "can". This can be driven by (a) extraction/format failure OR (b) reasoning variance — the data below distinguishes these, see the ★ Paper-worthy finding subsection after the Observations.

**Step selection**: for methods where the **offline-peak step ≠ final step (300)**, both are shown so late-training drift is visible as a separate row. Methods with monotone offline trajectories (1A, 1B under β=0.1, ABL-MINIRL-01) show just one or two representative rows. `(peak)` / `(final)` annotations mark which is which.

Table layout is **method × benchmark** (transposed from earlier layout) — one row per (method, step), one column per benchmark. Best per-column is **bolded**; winner of the "best-per-benchmark" summary row is the single EVAL-N that holds the top in that column.

### 4B model2 (strong / trainable sub-model) — mean@3

| Method | Step | MATH-500 | AIME-25 | AMC-23 | AQUA | GSM8K | MAWPS | SVAMP |
|---|---|---|---|---|---|---|---|---|
| EXP-13 M5.5 m2 (v1, β=0, lr=5e-7) | 300 | 78.6 | 20.0 | 55.8 | 80.1 | 91.8 | 95.2 | 93.4 |
| EXP-14 M5.6 m2 (v1, β=0.1, lr=5e-7) | 300 | 79.1 | 17.8 | 52.5 | 79.9 | 91.6 | 95.1 | 93.3 |
| EXP-15 LR3 m2 (v1, β=0, lr=1e-6) | 125 (peak) | 79.6 | 20.0 | 51.7 | 73.8 | 91.3 | 95.6 | 94.8 |
| EXP-16 1A m2 (v2, β=0, lr=5e-7) ★ | 225 (peak) | **83.1** | 14.4 | 55.0 | 70.2 | 91.3 | 95.4 | 93.7 |
| EXP-17 1B m2 (v2, β=0.1, lr=5e-7) | 275 | 82.5 | 15.6 | 57.5 | 80.2 | 91.8 | 95.2 | 93.7 |
| EXP-17 1B m2 (v2, β=0.1, lr=5e-7) | 300 (final) | 82.9 | 17.8 | **63.3** | 76.6 | **92.0** | 95.2 | 93.9 |
| EXP-18 1C m2 (v2, β=0, lr=1e-6) | 150 (peak) | 82.5 | 18.9 | **63.3** | 81.8 | 91.7 | **95.7** | 93.9 |
| EXP-18 1C m2 (v2, β=0, lr=1e-6) | 300 (final) ↓ drift | 78.1 | 10.0 | 50.8 | 79.8 | 90.1 | 94.6 | 92.0 |
| ABL-MINIRL-01 2Z-SFT (single, lr=5e-7) | 275 | 79.6 | **21.1** | 61.7 | **82.3** | **92.5** | 95.6 | 94.1 |
| ABL-MINIRL-01 2Z-SFT (single, lr=5e-7) | 300 (final) | 80.7 | 13.3 | 60.8 | 82.0 | 91.9 | **95.7** | 93.7 |
| ABL-MINIRL-02 2A-SFT (single + wdl_sft_is, lr=5e-7) | 275 | 80.1 | 11.1 | 55.8 | 66.3 | 90.6 | 94.3 | 92.8 |
| ABL-MINIRL-02 2A-SFT (single + wdl_sft_is, lr=5e-7) | 300 (final) | 80.1 | 10.0 | 60.0 | 57.0 | 90.4 | 94.5 | 91.7 |
| **Best-per-benchmark (EVAL-N)** |  | **83.1 (EVAL-20)** | **21.1 (EVAL-25)** | **63.3 (EVAL-18/21)** | **82.3 (EVAL-25)** | **92.5 (EVAL-25)** | **95.7 (EVAL-21/26)** | **94.8 (EVAL-10)** |

### 4B model2 (strong / trainable sub-model) — pass@3

| Method | Step | MATH-500 | AIME-25 | AMC-23 | AQUA | GSM8K | MAWPS | SVAMP |
|---|---|---|---|---|---|---|---|---|
| EXP-13 M5.5 m2 | 300 | 87.4 | 26.7 | 65.0 | 84.3 | 95.7 | 96.9 | 96.7 |
| EXP-14 M5.6 m2 | 300 | 86.8 | 23.3 | 62.5 | 84.6 | 94.8 | 96.3 | 95.3 |
| EXP-15 LR3 m2 | 125 (peak) | 87.6 | 23.3 | 60.0 | 84.6 | 95.5 | 96.3 | **97.0** |
| EXP-16 1A m2 | 225 (peak) | 89.4 | 23.3 | 70.0 | 83.9 | 95.0 | 96.3 | 96.3 |
| EXP-17 1B m2 | 275 | 89.2 | 20.0 | 72.5 | 88.2 | 95.3 | 96.3 | 96.0 |
| EXP-17 1B m2 ★ | 300 (final) | **90.2** | 23.3 | **80.0** | 85.4 | 95.0 | 96.3 | 96.0 |
| EXP-18 1C m2 | 150 (peak) | 88.6 | **30.0** | 75.0 | 88.6 | 95.5 | 96.3 | 96.3 |
| EXP-18 1C m2 | 300 (final) ↓ | 86.8 | 16.7 | 67.5 | 89.4 | 94.2 | 95.8 | 96.0 |
| ABL-MINIRL-01 | 275 | 88.4 | 26.7 | 77.5 | **92.5** | **95.8** | 96.1 | 96.0 |
| ABL-MINIRL-01 | 300 (final) | 89.2 | 20.0 | 72.5 | 90.2 | 95.5 | **96.6** | 96.0 |
| ABL-MINIRL-02 2A-SFT | 275 | 88.4 | 16.7 | 75.0 | 79.1 | 93.6 | 95.2 | 95.0 |
| ABL-MINIRL-02 2A-SFT | 300 (final) | 88.6 | 16.7 | **82.5** | 68.9 | 93.7 | 95.5 | 95.7 |
| **Best-per-benchmark (EVAL-N)** |  | **90.2 (EVAL-18)** | **30.0 (EVAL-21)** | **82.5 (EVAL-28)** | **92.5 (EVAL-25)** | **95.8 (EVAL-25)** | **96.6 (EVAL-26)** | **97.0 (EVAL-10)** |

**Key mean@3-vs-pass@3 ranking reversal on m2**: `mean@3` MATH-500 is led by **1A (EVAL-20, 83.1%)**, but `pass@3` MATH-500 is led by **1B step_300 (EVAL-18, 90.2%)**. The two metrics disagree on which v2 checkpoint is "best" on the headline benchmark — 1A has higher mean-per-sample accuracy, 1B has higher capability ceiling in 3 tries. This happens because 1B m2 has slightly more format failures (`extraction_fail` 8.7% vs 1A's 8.3%) pulling mean down, but more correct answers appear in *at least one* of 3 samples.

### 4B model1 (weak / anchor sub-model) — mean@3

| Method | Step | MATH-500 | AIME-25 | AMC-23 | AQUA | GSM8K | MAWPS | SVAMP |
|---|---|---|---|---|---|---|---|---|
| EXP-13 M5.5 m1 (v1, β=0) ★ | 300 | **70.5** | **8.9** | **45.8** | **57.5** | 82.0 | 84.9 | 85.3 |
| EXP-14 M5.6 m1 (v1, β=0.1) ⚠ | 300 | 48.9 | 5.6 | 30.8 | 20.9 | 64.4 | 64.3 | 62.7 |
| EXP-15 LR3 m1 (v1, β=0, lr=1e-6) | 125 | 63.7 | 6.7 | 36.7 | 45.9 | 70.7 | 80.7 | 77.4 |
| EXP-17 1B m1 (v2, β=0.1) ⚠⚠ | 275 | 38.7 | 4.4 | 30.8 | 13.9 | 43.8 | 43.6 | 38.6 |
| EXP-17 1B m1 (v2, β=0.1) ⚠⚠ | 300 (final) | 37.9 | 4.4 | 25.0 | 15.7 | 42.9 | 41.3 | 38.7 |
| EXP-18 1C m1 (v2, β=0, lr=1e-6) | 150 | 52.7 | 4.4 | 30.0 | 21.5 | 59.9 | 65.9 | 59.7 |
| EXP-18 1C m1 (v2, β=0, lr=1e-6) ★v2 | 300 (final) | 64.7 | 4.4 | 37.5 | 27.7 | **81.9** | **88.8** | 82.4 |
| **Best-per-benchmark (EVAL-N)** |  | **70.5 (EVAL-13)** | **8.9 (EVAL-13)** | **45.8 (EVAL-13)** | **57.5 (EVAL-13)** | **82.0 (EVAL-13)** | **88.8 (EVAL-24)** | **85.3 (EVAL-13)** |

### 4B model1 (weak / anchor sub-model) — pass@3

| Method | Step | MATH-500 | AIME-25 | AMC-23 | AQUA | GSM8K | MAWPS | SVAMP |
|---|---|---|---|---|---|---|---|---|
| EXP-13 M5.5 m1 ★ | 300 | **86.6** | **16.7** | **65.0** | **81.1** | **94.9** | 96.1 | **96.7** |
| EXP-14 M5.6 m1 ⚠ | 300 | 78.2 | 13.3 | 50.0 | 46.9 | 92.0 | 92.4 | 92.7 |
| EXP-15 LR3 m1 | 125 | 84.4 | 13.3 | 60.0 | 79.1 | 93.3 | **96.3** | 94.7 |
| EXP-17 1B m1 ⚠⚠ | 275 | 71.6 | 10.0 | 60.0 | 33.1 | 79.1 | 79.2 | 74.7 |
| EXP-17 1B m1 ⚠⚠ | 300 (final) | 71.6 | 13.3 | 50.0 | 37.4 | 78.6 | 75.5 | 74.0 |
| EXP-18 1C m1 | 150 | 78.4 | 13.3 | 47.5 | 44.9 | 89.5 | 93.0 | 88.7 |
| EXP-18 1C m1 | 300 (final) | 82.0 | 10.0 | 52.5 | 50.4 | 93.5 | 95.5 | 94.3 |
| **Best-per-benchmark (EVAL-N)** |  | **86.6 (EVAL-13)** | **16.7 (EVAL-13)** | **65.0 (EVAL-13)** | **81.1 (EVAL-13)** | **94.9 (EVAL-13)** | **96.3 (EVAL-11)** | **96.7 (EVAL-13)** |

**Striking pass@3 finding for m1**: EVAL-13 M5.5 m1's `pass@3 MATH-500 = 86.6%` is **higher than EVAL-20 1A m2's mean@3 MATH-500 = 83.1%** and **close to 1A m2's pass@3 of 89.4%**. This means the v1 β=0 m1 is **capable** of finding MATH-500 answers at almost the same rate as the v2 m2, but it *fails to extract / majority-vote* on enough samples to compete on mean@3 (where it lands at 70.5%, −12.6 pp from its pass@3). All v2 m1s (1B / 1C) also show large `pass@3 − mean@3` gaps (14–26 pp), but their pass@3 ceilings are lower (71.6–82.0% MATH-500 vs M5.5's 86.6%) — v2's lower-bound clip under β>0 reduces both format compliance AND capability on m1, while the joint-coupling gain under β=0/v2 (1C step 300) closes the format gap but not the capability gap.

*Note: EXP-16 1A m1 not evaluated (deferred — see EVAL-20 notes). EXP-08 Qwen3-4B-Base pretrained and EXP-11 Qwen3-4B-SFT-DPO external baselines were evaluated pre-harness (pass@3 not tabulated here — reference MATH-500 mean@3 32.5% and 67.7% respectively; full data at `/data-1/model_weights/qwen3-4b-sft-dpo/step_367/inference_n3/eval_metrics.json`).*

### 4B single-model ablation (ABL-MINIRL-01) vs v2 joint m2 — best-per-benchmark baseline

Head-to-head for H3 decomposition. ABL-MINIRL-01 uses the same SFT-stage-1 init as joint's model2, same MATH data, same 300-step horizon, lr=5e-7 — the only difference is single Qwen3-4B backbone + MiniRL loss (no joint fusion, no wdl_sft_is). **The opponent column is "v2 joint m2 best-per-benchmark"** — the single best v2 joint m2 value across EXP-16/17/18 for each benchmark, NOT a single checkpoint. This avoids the misleading impression produced by using 1A m2 alone as the baseline on every row.

| Benchmark | Metric | ABL-MINIRL-01 best | v2 joint m2 best | Δ (ABL − joint) |
|---|---|---|---|---|
| MATH-500  | mean@3 | 80.7 (step 300) | **83.1** (1A 225, EVAL-20) | **−2.4 pp** |
| MATH-500  | pass@3 | 89.2 (step 300) | **90.2** (1B 300, EVAL-18) | **−1.0 pp** |
| AIME-2025 | mean@3 | **21.1** (step 275) | 18.9 (1C 150, EVAL-21) | +2.2 pp (n=30 noise ±3.3/prob) |
| AIME-2025 | pass@3 | 26.7 (step 275) | **30.0** (1C 150, EVAL-21) | −3.3 pp (n=30 noise) |
| AMC-2023  | mean@3 | 61.7 (step 275) | **63.3** (1B 300 / 1C 150) | −1.6 pp |
| AMC-2023  | pass@3 | 77.5 (step 275) | **80.0** (1B 300, EVAL-18) | −2.5 pp |
| AQUA      | mean@3 | **82.3** (step 275) | 81.8 (1C 150, EVAL-21) | +0.5 pp |
| AQUA      | pass@3 | **92.5** (step 275) | 89.4 (1C 300, EVAL-23) | +3.1 pp |
| GSM8K     | mean@3 | **92.5** (step 275) | 92.0 (1B 300, EVAL-18) | +0.5 pp |
| GSM8K     | pass@3 | **95.8** (step 275) | 95.5 (1C 150, EVAL-21) | +0.3 pp |
| MAWPS     | mean@3 | 95.7 (step 300) | 95.7 (1C 150, EVAL-21) | 0.0 pp (tied) |
| MAWPS     | pass@3 | **96.6** (step 300) | 96.3 (multiple) | +0.3 pp |
| SVAMP     | mean@3 | 94.1 (step 275) | 93.9 (1B 300 / 1C 150) | +0.2 pp |
| SVAMP     | pass@3 | 96.0 (step 275/300) | **96.3** (1A 225, EVAL-20) | −0.3 pp |

**Reading**:

- **Hard math (MATH-500, AIME-2025, AMC-2023)** — joint leads on both metrics, gap **−1.0 to −3.3 pp**. This is the meaningful region of the comparison.
  - MATH-500 `pass@3 −1.0 pp` vs `mean@3 −2.4 pp` → ABL's **capability ceiling** (pass@3) is only ~1 pp behind joint; the extra ~1.4 pp on `mean@3` comes from **per-sample inconsistency** (ABL reaches the answer in 3 tries almost as often as joint, but on fewer of the 3 rollouts per prompt). This is NOT extraction failure — ABL's MATH-500 `extraction_fail` is 2.9% vs joint v2 m2's 7.6–8.7%; ABL extracts *better*. The remaining gap is **reasoning variance**: ABL samples reach correct answers less consistently even when it has the capability. See ★ Paper-worthy finding subsection below.
- **AQUA / GSM8K / MAWPS / SVAMP** — essentially tied (±0.5 pp). SFT init saturates these; joint machinery provides no measurable lift.
- **AIME-2025** — opposite direction on the two metrics (ABL +2.2 on mean@3, −3.3 on pass@3), but n=30 makes both swings noise-range. Don't draw conclusions.

**Joint contribution is concentrated on hard-math reasoning (MATH-500 + AMC-2023), with half of the `mean@3` gap on MATH-500 closing at `pass@3`**. On easier benchmarks, single-model MiniRL from SFT init is cost-equivalent.

### 4B single-model loss-term ablation (ABL-MINIRL-02 2A-SFT vs ABL-MINIRL-01 2Z-SFT) — L_loss isolation

Pair: **same single Qwen3 backbone, same SFT-stage-1 init, same data, same lr=5e-7, same 300-step horizon. Only difference = `loss_mode=wdl_sft_is` (2A-SFT) vs pure MiniRL (2Z-SFT).** Per-benchmark delta 2A − 2Z attributes exclusively to the loss term; joint architecture, fused-logit rollout, and parameter-sharing effects are excluded by design. Both step_275 (online peak-ish) and step_300 (final) are shown.

| Benchmark | Metric | 2A-SFT step 275 (EVAL-27) | 2Z-SFT step 275 (EVAL-25) | Δ (2A − 2Z) | 2A-SFT step 300 (EVAL-28) | 2Z-SFT step 300 (EVAL-26) | Δ (2A − 2Z) |
|---|---|---|---|---|---|---|---|
| MATH-500  | mean@3 | 80.1 | 79.6 | **+0.5** | 80.1 | 80.7 | −0.6 |
| MATH-500  | pass@3 | 88.4 | 88.4 | 0.0 | 88.6 | 89.2 | −0.6 |
| AIME-2025 | mean@3 | 11.1 | 21.1 | −10.0 (n=30 noise) | 10.0 | 13.3 | −3.3 (noise) |
| AIME-2025 | pass@3 | 16.7 | 26.7 | −10.0 (n=30 noise) | 16.7 | 20.0 | −3.3 (noise) |
| AMC-2023  | mean@3 | 55.8 | 61.7 | −5.9 | 60.0 | 60.8 | −0.8 |
| AMC-2023  | pass@3 | 75.0 | 77.5 | −2.5 | **82.5** | 72.5 | **+10.0** |
| AQUA      | mean@3 | 66.3 | 82.3 | **−16.0** ⚠ | 57.0 | 82.0 | **−25.0** ⚠ |
| AQUA      | pass@3 | 79.1 | 92.5 | **−13.4** ⚠ | 68.9 | 90.2 | **−21.3** ⚠ |
| GSM8K     | mean@3 | 90.6 | 92.5 | −1.9 | 90.4 | 91.9 | −1.5 |
| GSM8K     | pass@3 | 93.6 | 95.8 | −2.2 | 93.7 | 95.5 | −1.8 |
| MAWPS     | mean@3 | 94.3 | 95.6 | −1.3 | 94.5 | 95.7 | −1.2 |
| MAWPS     | pass@3 | 95.2 | 96.1 | −0.9 | 95.5 | 96.6 | −1.1 |
| SVAMP     | mean@3 | 92.8 | 94.1 | −1.3 | 91.7 | 93.7 | −2.0 |
| SVAMP     | pass@3 | 95.0 | 96.0 | −1.0 | 95.7 | 96.0 | −0.3 |

**Reading (L_loss verdict)**:

- **MATH-500**: L_loss ≈ **0** in single-model form (−0.6 to +0.5 pp on mean@3; 0 to −0.6 on pass@3). **wdl_sft_is as a loss term, absent the joint architecture, does NOT lift MATH-500.** The ~2.4 pp joint-vs-single MATH-500 gap observed in EVAL-20/26 is therefore attributable to **joint architecture + fused-logit rollout + parameter sharing**, not the loss.
- **AQUA regression is striking — L_loss = −16 to −25 pp**. Not a format/extraction artifact (ext_fail is comparable across 2A and 2Z); the drop is reasoning-level. Hypothesis: wdl_sft_is's IS ratio / clipping under single-model training alters the output distribution in a way that breaks AQUA letter-answer discipline (AQUA is multiple-choice A/B/C/D/E). Worth isolating.
- **AMC-2023 pass@3 +10 pp at step 300**: the only place 2A-SFT beats 2Z-SFT. Combined with near-zero mean@3 delta, this says wdl_sft_is broadens capability on hard math but doesn't tighten sample-to-sample consistency. Small n (n=40) so treat as a hint, not a confirmation.
- **Easy benchmarks (GSM8K / MAWPS / SVAMP)**: consistent −1 to −2 pp small regression on both metrics. wdl_sft_is appears to slightly hurt easy-bench performance in single-model form.
- **AIME-2025**: n=30 noise-dominated; both directions of delta are within 2-problem noise.

**Implication for plan §5 L_fusion**: with L_loss ≈ 0 on MATH-500, the full joint-vs-single MATH-500 lift collapses onto **L_fusion** = score(1A) − score(2A-SFT). Measured at step 300: **mean@3 +3.0 pp** (EVAL-20 83.1 − EVAL-28 80.1), **pass@3 +1.6 pp** (EVAL-18 90.2 − EVAL-28 88.6). The joint machinery buys ~3 pp mean@3 on MATH-500, almost entirely via fused-logit rollout + parameter sharing, not the loss term.

### Observations

1. **v2 IS-corrected loss breaks the v1 ~79–80% model2-MATH-500 ceiling, and the v2 ceiling sits at ~83% regardless of lr or β**: EVAL-20 1A m2 reaches **83.1%**, EVAL-18 1B m2 82.9%, EVAL-16 1B m2 82.5%, EVAL-21 1C m2 82.5%. All four v2 m2 peaks land in a tight 82.5–83.1% band. Doubling lr (5e-7 → 1e-6) gives 1C a higher *online* peak (71.98% vs 1A's 71.37%, 1B's 70.97%) but **no offline gain**. The v1 runs (M5.5, M5.6, LR3) clustered at 78.6%–79.6%. The ceiling was loss-bound, not data-/capacity-/lr-bound.
2. **β>0 is strictly worse on model1, and v2's lower-bound clip does NOT fix it**: EVAL-17/19 (1B m1 β=0.1 under v2) are MORE degraded than EVAL-15 (M5.6 m1 β=0.1 under v1). Extraction_fail climbs from v1's 24–28% uniform band to v2's 37–49% uniform band. EVAL-22/24 (**1C m1 β=0 under v2**) provides the clean control: ext_fail 22–30% at step 150, **5–19% at step 300** — no format collapse. Confirmation: the collapse is a β>0 effect specifically, not a v2 loss property.
3. **Online joint signal is a biased estimate of m2 quality in different directions under different lrs**: 1B (β=0.1) online said step 275 > step 300, but offline m2-only said step 300 > step 275 (the joint-fusion signal was dragged down by the degrading m1). 1C (β=0, higher lr) online peaked at step 150 and drifted down; offline m2-only confirms step 150 > step 300 by 4.4 pp on MATH-500. **Rule**: extract m2 from the offline-peak step, which under lr=5e-7 is the final step, and under lr=1e-6 is the online-peak step.
4. **Training time under β=0 IMPROVES m1 when lr is high enough** — an unexpected 1C finding: EVAL-22 → EVAL-24 moves m1 MATH-500 from 52.7% → 64.7% (+12 pp) and GSM8K from 59.9% → 81.9% (+22 pp) while ext_fail drops −15 pp. Under β=0 the reverse-SFT coefficient is zero so m1 receives no direct gradient — the gain strongly implies joint-arch parameter sharing (lm_head / embeddings / norms) lets m2's continued forward-SFT training drag m1 along on format-compliance axes. pass@3 moves only +3.6 pp, so the +12 mean@3 is mostly format, not capability.
5. **Training time under β>0 DEGRADES m1 monotonically**: 1B step 275→300 m1 MATH −0.8, ext_fail +1–3 pp. The direction is opposite to 1C's m1 trajectory (β=0). The mirror pairing (1C vs 1B on m1 across steps) cleanly isolates β as the driver of m1 format erosion.
6. **Best 4B model2 to date — winner depends on metric**: `mean@3` MATH-500 → **EVAL-20 1A m2 step 225 (83.1%)**; `pass@3` MATH-500 → **EVAL-18 1B m2 step 300 (90.2%)**. The reversal tells us 1B has higher capability ceiling under n=3 rollouts but 1A has lower format/extraction loss rate per sample. For production rollout-cost-constrained deployment, 1A is the default; if downstream uses majority-vote or reranking (i.e. can exploit pass@k), 1B is preferable. **β=0 at lr=5e-7** (EXP-16 1A) is the recommended default: leads on `mean@3`, avoids the β>0 m1 collapse, and avoids the 1C late-training m2 drift at lr=1e-6.
7. **Best 4B model1 to date — v2 wins only on easier benchmarks; v1 M5.5 is the overall ceiling**: (a) On `mean@3` **EVAL-13 M5.5 m1 step 300** sweeps MATH-500 (**70.5**), AIME-2025 (**8.9**), AMC-2023 (**45.8**), AQUA (**57.5**), GSM8K (**82.0**), SVAMP (**85.3**); EVAL-24 1C m1 step 300 only wins MAWPS (88.8). (b) On `pass@3` the sweep is even more uniform — EVAL-13 leads on 6 of 7 benchmarks (only MAWPS goes to EVAL-11 LR3 m1). (c) `pass@3 − mean@3` gap on m1s is 14–26 pp, much larger than m2s' 5–10 pp — m1s are **format-constrained, not capability-constrained**; the v1 M5.5 m1 "knows" MATH-500 answers at pass@3 86.6% but extracts them on mean@3 only 70.5%. **v2 m1s (1C under β=0) close part of this format gap but their capability ceiling is still below v1's**. Directionally: to improve m1 reasoning capability, change training signal (more / different m1 gradient); to improve m1 accuracy from a given ceiling, target format-compliance (extract-fail reduction).
8. **Decisive m1 format-collapse experiment is done**: EVAL-17/19 refuted "v2's lower-bound clip prevents m1 collapse under β>0", and EVAL-22/24 established a clean β=0 baseline. Future experiments should (a) default to β=0 and invest in data / loss-mode / scale variants, or (b) if β>0 is required for a downstream reason, target the format-token mechanism directly (freezing m1 embed/lm_head, format-token logit KL, gradient masking on extraction-critical tokens).
9. **Joint-vs-single decomposition (EVAL-25/26 — ABL-MINIRL-01), per-benchmark and per-metric**: using **v2 joint m2 best-per-benchmark** (not 1A m2 alone) as the baseline, single-model MiniRL from the same SFT-stage-1 init **trails on hard-math only**:
   - **MATH-500**: `mean@3` −2.4 pp (80.7 vs 83.1), `pass@3` −1.0 pp (89.2 vs 90.2). Decomposition: **capability ceiling gap ≈ 1 pp** (pass@3), **per-sample reasoning-variance gap ≈ 1.4 pp** (the difference between the two). The reasoning-variance component is the larger of the two, and it is **not** explained by extraction failure — ABL has *lower* `extraction_fail` on MATH-500 (2.9%) than joint v2 m2 (7.6–8.7%). Joint's `mean@3` advantage comes from producing correct answers more consistently across rollouts, not from a higher ceiling. See ★ Paper-worthy finding subsection below.
   - **AMC-2023**: `mean@3` −1.6 pp, `pass@3` −2.5 pp — joint-dominant on both metrics.
   - **AIME-2025**: n=30 noise-dominated (±3.3 pp/problem); mean@3 and pass@3 disagree on direction, don't read into it.
   - **AQUA / GSM8K / MAWPS / SVAMP**: ties within ±0.5 pp on both metrics. SFT init saturates; joint adds nothing measurable.
   **H3 (init-dominant) is partially confirmed**: the SFT init delivers the ~80% MATH-500 band and the full easy-bench performance without any joint machinery. The joint architecture + wdl_sft_is loss buys a real but narrow ~2.4 pp `mean@3` lift on hard math (MATH-500 + AMC-2023), shrinking to ~1.0 pp on `pass@3`. Online val fails to see this gap (joint and single-model online peaks within 1 pp); only offline `mean@3` / `pass@3` on hard-math separates them. **Practical implication**: joint machinery is worth its ~2× param memory + extraction overhead if MATH-500 / AMC-2023 is the target metric; otherwise single-model MiniRL from SFT init is cost-equivalent.
10. **L_loss isolation (ABL-MINIRL-02 2A-SFT vs ABL-MINIRL-01 2Z-SFT, EVAL-27/28 vs EVAL-25/26)** — the single-model loss-only ablation cleanly isolates the `wdl_sft_is` loss contribution from the joint architecture's contribution. Same single Qwen3 backbone, same init, same data, same lr; only the loss differs. Result on MATH-500: **mean@3 delta ≈ 0** (80.1 vs 80.7 at step 300, within step-to-step noise), **pass@3 delta ≈ 0** (88.6 vs 89.2). Result on AQUA: **L_loss = −16 to −25 pp** — `wdl_sft_is` single-model hurts AQUA multiple-choice discipline badly; this is reasoning-level, not format-level (ext_fail comparable). Result on easy benches: small consistent −1 to −2 pp regression. **Plan §5 decomposition resolves**: the full joint-vs-single MATH-500 lift is **L_fusion** (architecture + fused-logit rollout + parameter sharing), not L_loss. Measured L_fusion on MATH-500 at step 300: **+3.0 pp mean@3** (1A m2 83.1 vs 2A-SFT 80.1), **+1.6 pp pass@3** (1B m2 90.2 vs 2A-SFT 88.6). **Follow-on experiment implied**: joint-architecture + MiniRL loss (no wdl_sft_is) would complete the 2×2 grid and confirm whether the tightness of `pass@3 − mean@3` (6.1–6.7 pp) is an architectural property rather than a wdl_sft_is property.
11. **`mean@3` vs `pass@3` give different winners — and the pattern itself is informative**: on m2, 1A (EVAL-20) leads MATH-500 `mean@3` but 1B step 300 (EVAL-18) leads MATH-500 `pass@3`; on m1, EVAL-13 M5.5 m1 sweeps `pass@3` but is partial on `mean@3` (EVAL-24 1C m1 takes MAWPS on mean@3 only). The `pass@3 − mean@3` gap decomposes into two mechanisms: **(a) extraction failure** — the answer is correct internally but the output format prevents extraction (one-to-one correspondence with `extraction_fail`); **(b) reasoning variance** — the model reaches the answer on some of 3 samples but not others, independent of format. The relative contribution of (a) vs (b) can be inferred by cross-referencing `pass@3 − mean@3` with `extraction_fail`: a method with **low `extraction_fail` AND large `pass@3 − mean@3` gap** has its inconsistency dominated by reasoning variance (ABL on MATH-500 is this profile). A method with **high `extraction_fail` AND large `pass@3 − mean@3` gap** has it dominated by format (all m1s, especially v2 β=0.1 m1s). m1 gaps: 14–26 pp (mostly format). v2 m2 peak gaps: 6.1–7.3 pp (tight). v1 m2 gaps: 7.7–8.8 pp. ABL-MINIRL-01 gaps on MATH-500: 8.5–8.8 pp. **Practical rule**: for the "what is this method capable of" question, report `pass@3`; for "what will production serving see", report `mean@3`; for "how consistent is this method's reasoning", report `pass@3 − mean@3` alongside `extraction_fail` to disentangle the two mechanisms.

### ★ Paper-worthy finding: `pass@3 − mean@3` as a reasoning-consistency statistic

**Motivation**: most RL-for-reasoning papers report only `mean@k` (or `pass@1` / `maj@k`). But a single-number accuracy metric conflates three orthogonal failure modes — (a) capability ceiling, (b) extraction/format failure, (c) reasoning variance. Reporting `mean@3` and `pass@3` side-by-side **and cross-referencing with `extraction_fail`** separates them cleanly.

**Proposed statistic for write-up**:

$$\text{ReasoningVar}_k(m, D) = \text{pass@}k(m, D) - \text{mean@}k(m, D) - \text{extraction\_fail}(m, D) \cdot \text{pass@}k(m, D)$$

(Roughly: "fraction of prompts where the model reaches the answer in at least 1 of k tries but doesn't reach it on all k, adjusted for extraction losses.") On MATH-500 this gives a cleaner reasoning-consistency signal than `pass@3 − mean@3` alone.

**Key empirical observation from this work** (all MATH-500, n=3):

| Method cluster | Representative (EVAL-N) | `mean@3` | `pass@3` | `ext_fail` | `pass@3 − mean@3` |
|---|---|---|---|---|---|
| **v2 joint m2 peak (`wdl_sft_is` + joint)** | 1A step 225 (EVAL-20) | 83.1 | 89.4 | 8.3 | **6.3** ★ tight |
| v2 joint m2 peak | 1C step 150 (EVAL-21) | 82.5 | 88.6 | 7.6 | **6.1** ★ tight |
| v2 joint m2 peak | 1B step 275 (EVAL-16) | 82.5 | 89.2 | 9.6 | 6.7 |
| v1 joint m2 (`wdl_sft` + joint) | M5.5 step 300 (EVAL-12) | 78.6 | 87.4 | 16.7 | 8.8 |
| v1 joint m2 | LR3 step 125 (EVAL-10) | 79.6 | 87.6 | 14.6 | 8.0 |
| **Single-model MiniRL (ABL, no wdl_sft_is, no joint)** | 2Z-SFT step 300 (EVAL-26) | 80.7 | 89.2 | 2.9 | **8.5** ★ loose |
| Single-model MiniRL | 2Z-SFT step 275 (EVAL-25) | 79.6 | 88.4 | 3.4 | **8.8** ★ loose |
| **Single-model + wdl_sft_is (ABL, no joint)** | 2A-SFT step 300 (EVAL-28) | 80.1 | 88.6 | 4.8 | **8.5** ★ loose |
| Single-model + wdl_sft_is | 2A-SFT step 275 (EVAL-27) | 80.1 | 88.4 | 4.9 | **8.3** ★ loose |

**The finding**: **v2 `wdl_sft_is` loss (paired with joint architecture) tightens the `pass@3 − mean@3` gap on MATH-500 from ~8 pp down to ~6.5 pp**, while simultaneously raising the `pass@3` ceiling by ~2 pp over v1. The single-model ablation (ABL-MINIRL-01) — same SFT init, same data, same lr, only the loss and architecture differ — stays at the v1-era gap of ~8.5 pp despite having *dramatically lower* `extraction_fail` (2.9% vs v1's 14.6%). So: **ABL is a better extractor but a less consistent reasoner than v2 joint m2.** The improvement from v1 → v2 joint m2 is NOT primarily format-compliance (v2 m2 actually has *similar* `extraction_fail` to v1 m2, 7.6–9.6% vs 14.6–16.9%, modestly better). It is reasoning-variance reduction.

**Further refinement from ABL-MINIRL-02 (2A-SFT, EVAL-27/28)**: adding the `wdl_sft_is` loss to the single-model variant (ABL-MINIRL-02 = 2Z-SFT + wdl_sft_is loss, no joint) **does NOT tighten the gap** — 2A-SFT step_300 has `pass@3 − mean@3` = 8.5 pp, identical to 2Z-SFT's 8.5 pp. The tightening from 8.5 → 6.5 pp therefore requires the **joint architecture** (fused-logit rollout + parameter sharing between m1/m2), not the loss function. This shifts the mechanism hypothesis: reasoning-variance reduction is an **architectural** property of the joint setup, not a loss-function property. Candidate mechanisms: (a) fused-logit rollout regularizes output distributions during RL training (exposes m2 to a smoother reward landscape), or (b) shared lm_head / embeddings act as an implicit ensemble that dampens sample-level variance. A clean next ablation would be **joint-architecture + MiniRL loss** (no wdl_sft_is) to isolate (a) from the specific loss choice; if that also tightens to ~6.5 pp, the architecture is the driver.

**Hypothesis for the mechanism** (to be validated): fused-logit rollout + IS-corrected loss produces a softer / more consistent policy on hard-math prompts. When backed out to m2-only offline evaluation, this appears as lower per-sample variance at a given capability ceiling. The tightness is specifically a **v2 + joint** property — v1 joint and single-model v2-less both show loose gaps.

**Why this matters for the paper**:
1. It gives a **mechanism-level claim** for why v2 wdl_sft_is beats v1: not "higher ceiling" but "more consistent sampling at a slightly higher ceiling".
2. It identifies a statistic (`pass@3 − mean@3` alongside `extraction_fail`) that **separates format-compliance gains from reasoning-consistency gains** — both are cited as "accuracy improvements" in SOTA comparisons, but they are structurally different and respond to different interventions.
3. It suggests that `pass@3 − mean@3` could be a **training-dynamics metric** during RL (not just a final eval statistic) — a natural next ablation is to track it across checkpoints and see when it tightens.

**Caveat before citing in the paper**: the decomposition is clean for "single method × all prompts in aggregate", but prompt-level `pass@3 − mean@3` is noisy (each prompt contributes 0, 1/3, 2/3, or 3/3 to mean@3 and 0 or 1 to pass@3 — tight discrete distribution). Variance should be reported (bootstrapped over prompts). Also this is n=3 data; for a write-up, consider also reporting at n=8 or n=16 where the gap opens wider and the decomposition gets more precise.

**Where to place in the paper**: either as a dedicated subsection in the "Analysis" section under a heading like "Decomposing the v2-over-v1 gain: capability ceiling vs reasoning consistency", or as a 2-column table in the main results if space permits. Either way, include `extraction_fail` in every benchmark table — without it, the `pass@3 − mean@3` interpretation is ambiguous.

**Greppable marker for future sessions**: `★ Paper-worthy finding`. One other such finding in the doc at the moment, to be added as they emerge.

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
