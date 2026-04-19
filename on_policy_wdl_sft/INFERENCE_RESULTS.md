# Inference Results — On-Policy WDL-SFT

This file tracks offline vLLM evaluation results for On-Policy WDL-SFT experiments. Each entry cross-references back to the experiment index (`EXPERIMENT_INDEX.md`) by EXP-ID.

Numbering continues from `recipe/joint_training/INFERENCE_RESULTS.md` (EVAL-01 through EVAL-09).

> **Note**: `EXPERIMENT_INDEX.md` carries EXP-12 (M5, diverged, no checkpoints retained), EXP-13 (M5.5, offline eval at step 300 → EVAL-12 / EVAL-13), EXP-14 (M5.6, offline eval at step 300 → EVAL-14 / EVAL-15), EXP-15 (LR3, offline eval at step 125 → EVAL-10 / EVAL-11). All offline evals use the same pipeline and generation params; the only axes that vary are the source checkpoint and the sub-model extracted.

---

## EVAL-10: EXP-15 WDL-SFT-LR3 step 125 (model2 — strong/trainable)

| Field | Value |
|---|---|
| **Source Experiment** | EXP-15 (WDL-SFT-LR3, On-Policy WDL-SFT lr=1e-6) |
| **Model Weights** | `/data-1/eval_results/wdl-sft-lr3-step125_model2` |
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

- model2 is the strong/trainable sub-model (initialized from Qwen3-4B-Base-SFT-stage-1, then trained by On-Policy WDL-SFT for 125 steps at lr=1e-6). It was extracted from the merged joint checkpoint at `/data-1/eval_results/wdl-sft-lr3-step125_merged_joint/`.
- Online training validation at step 125 reported MATH-500 acc/mean@1 = **68.15%** (joint fused inference). Offline model2-only mean@3 = **79.6%** — the large gap (+11.5%) indicates model2 alone is substantially stronger than the fused joint model on MATH-500. This is expected: model2 benefits from the SFT-stage-1 initialization and WDL-SFT training, while the fused model blends in the weaker Qwen3-4B-Base (model1).
- Compared to best previous 4B baseline **EXP-11 (Qwen3-4B-SFT-DPO, mean@3)** from `recipe/joint_training/INFERENCE_RESULTS.md`: model2 outperforms on all 7 benchmarks — MATH-500 **+11.9%** (79.6% vs 67.7%), AIME-2025 **+12.2%** (20.0% vs 7.8%), AMC-2023 **+10.9%** (51.7% vs 40.8%), AQUA **+8.8%** (73.8% vs 65.0%), GSM8K **+1.5%** (91.3% vs 89.8%), MAWPS **+1.2%** (95.6% vs 94.4%), SVAMP **+4.1%** (94.8% vs 90.7%).
- **High extraction failure on AIME-2025 (70%) and AMC-2023 (36.7%)**: the model often generates long reasoning chains without a clearly formatted `\boxed{}` final answer on hard competition problems. The AIME mean@3 of 20.0% is achieved entirely among the ~30% prompts where extraction succeeded; effective accuracy among those is ~67%.
- Generation time: 1449s (~24 min) for 2798 prompts × 3 = 8394 generations (tp=8).
- Raw results saved to: `/data-1/eval_results/wdl-sft-lr3-step125_n3_sysprompt/`

---

## EVAL-11: EXP-15 WDL-SFT-LR3 step 125 (model1 — weak/anchor)

| Field | Value |
|---|---|
| **Source Experiment** | EXP-15 (WDL-SFT-LR3, On-Policy WDL-SFT lr=1e-6) |
| **Model Weights** | `/data-1/eval_results/wdl-sft-lr3-step125_model1` |
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
- Raw results saved to: `/data-1/eval_results/wdl-sft-lr3-step125_model1_n3_sysprompt/`

---

## EVAL-12: EXP-13 WDL-SFT-M5.5 step 300 (model2 — strong/trainable)

| Field | Value |
|---|---|
| **Source Experiment** | EXP-13 (WDL-SFT-M5.5, forward-only On-Policy WDL-SFT, lr=5e-7, β=0, baseline) |
| **Model Weights** | `/data-1/eval_results/wdl-sft-m5_5-step300_model2` |
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
- Raw results saved to: `/data-1/eval_results/wdl-sft-m5_5-step300_n3_sysprompt/`

---

## EVAL-13: EXP-13 WDL-SFT-M5.5 step 300 (model1 — weak/anchor)

| Field | Value |
|---|---|
| **Source Experiment** | EXP-13 (WDL-SFT-M5.5, forward-only, lr=5e-7, β=0) |
| **Model Weights** | `/data-1/eval_results/wdl-sft-m5_5-step300_model1` |
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
- Raw results saved to: `/data-1/eval_results/wdl-sft-m5_5-step300_model1_n3_sysprompt/`

---

## EVAL-14: EXP-14 WDL-SFT-M5.6 step 300 (model2 — strong/trainable)

| Field | Value |
|---|---|
| **Source Experiment** | EXP-14 (WDL-SFT-M5.6, bidirectional On-Policy WDL-SFT, lr=5e-7, β=0.1) |
| **Model Weights** | `/data-1/eval_results/wdl-sft-m5_6-step300_model2` |
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
- Raw results saved to: `/data-1/eval_results/wdl-sft-m5_6-step300_n3_sysprompt/`

---

## EVAL-15: EXP-14 WDL-SFT-M5.6 step 300 (model1 — weak/anchor) ⚠

| Field | Value |
|---|---|
| **Source Experiment** | EXP-14 (WDL-SFT-M5.6, bidirectional, lr=5e-7, β=0.1) |
| **Model Weights** | `/data-1/eval_results/wdl-sft-m5_6-step300_model1` |
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
- Raw results saved to: `/data-1/eval_results/wdl-sft-m5_6-step300_model1_n3_sysprompt/`

---

## Cross-Experiment Comparison (On-Policy WDL-SFT vs 4B Baselines)

### 4B model2 (strong/trainable sub-model) — mean@3

| Benchmark | EXP-11 Qwen3-4B-SFT-DPO (ext, 365 steps) | **EXP-13 M5.5 m2** (lr=5e-7, β=0, step 300) | **EXP-14 M5.6 m2** (lr=5e-7, β=0.1, step 300) | **EXP-15 LR3 m2** (lr=1e-6, β=0, step 125) |
|---|---|---|---|---|
| **MATH-500** | 67.7% | 78.6% | 79.1% | **79.6%** |
| **AIME-2025** | 7.8% | **20.0%** | 17.8% | **20.0%** |
| **AMC-2023** | 40.8% | **55.8%** | 52.5% | 51.7% |
| **AQUA** | 65.0% | **80.1%** | 79.9% | 73.8% |
| **GSM8K** | 89.8% | **91.8%** | 91.6% | 91.3% |
| **MAWPS** | 94.4% | 95.2% | 95.1% | **95.6%** |
| **SVAMP** | 90.7% | 93.4% | 93.3% | **94.8%** |

### 4B model1 (weak/anchor sub-model) — mean@3

| Benchmark | EXP-08 Qwen3-4B-Base (pretrained) | **EXP-13 M5.5 m1** (lr=5e-7, β=0, step 300) | **EXP-14 M5.6 m1** (lr=5e-7, β=0.1, step 300) ⚠ | **EXP-15 LR3 m1** (lr=1e-6, β=0, step 125) |
|---|---|---|---|---|
| **MATH-500** | 32.5% | **70.5%** | 48.9% | 63.7% |
| **AIME-2025** | 3.3% | **8.9%** | 5.6% | 6.7% |
| **AMC-2023** | 15.0% | **45.8%** | 30.8% | 36.7% |
| **AQUA** | 6.8% | **57.5%** | 20.9% | 45.9% |
| **GSM8K** | 27.8% | **82.0%** | 64.4% | 70.7% |
| **MAWPS** | 21.7% | **84.9%** | 64.3% | 80.7% |
| **SVAMP** | 25.9% | **85.3%** | 62.7% | 77.4% |

### Observations

1. **model2 ceiling is ~79–80% on MATH-500**: All three WDL-SFT runs (M5.5 at step 300, M5.6 at step 300, LR3 at step 125) land within 1% of each other. Doubling lr (LR3) accelerates reaching this ceiling but does not raise it — and loses stability after step 125. **M5.5 (forward-only, lr=5e-7) is the safest way to get there**; M5.6 (β=0.1, lr=5e-7) is equally good on model2 but has a catastrophic side-effect on model1.
2. **β>0 destroys model1**: M5.6 model1 is a full tier below M5.5 model1 on every benchmark (~20% gaps across the board), driven by a uniform 24–28% extraction_fail rate. The reverse SFT term at β=0.1 erodes format compliance in the anchor, which starts from a weaker pretrained init and has no SFT-style prior to resist the push-away signal.
3. **Training time matters for model1**: M5.5 m1 (300 steps) > LR3 m1 (125 steps) by ~7% MATH-500 and ~10%+ on AMC/AQUA/GSM8K/MAWPS/SVAMP. Longer training at the safer lr=5e-7 is strictly better for the anchor.
4. **Best 4B model to date**: EXP-15 LR3 m2 step 125 remains the headline number (79.6% MATH-500, 95.6% MAWPS, 94.8% SVAMP), but EXP-13 M5.5 m2 step 300 is effectively tied and comes from a fully-stable training trajectory, making it the more reliable choice if reproducibility matters.

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
