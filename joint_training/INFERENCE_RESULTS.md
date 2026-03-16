# Inference Results

This file tracks offline vLLM inference and evaluation results for merged model weights. Each entry cross-references back to the experiment index (`EXPERIMENT_INDEX.md`) by EXP-ID.

**Maintenance rules**: See `docs/joint_training/constraints/experiment_tracking/experiment_index_spec.md`

---

## EVAL-01: EXP-04 Joint-MiniRL-1.7B step 100 (model2 only)

| Field | Value |
|---|---|
| **Source Experiment** | EXP-04 (Joint-MiniRL-Qwen3-1.7B-MATH) |
| **Model Weights** | `/data-1/model_weights/EXP-04_Joint-MiniRL-1.7B-MATH/step_100_model2` |
| **Checkpoint Step** | 100 (final) |
| **Sub-Model** | model2 (trainable, extracted from joint model) |
| **Inference Engine** | vLLM 0.8.5 (FLASH_ATTN backend, V1 engine, tp=4) |
| **Benchmarks** | MATH-500, AIME-2025, AMC-2023, MinervaMAth, OlympiadBench |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=3, max_tokens=4096 |
| **Date** | 2026-03-16 |

### Results

| Benchmark | Samples | mean@3 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|
| **MATH-500** | 500 | **64.2%** | 76.8% | 67.8% | 1.4% |
| **AIME-2025** | 30 | **4.4%** | 10.0% | 3.3% | 0.0% |
| **AMC-2023** | 40 | **36.7%** | 57.5% | 42.5% | 4.2% |
| **MinervaMAth** | 272 | **24.4%** | 35.3% | 26.1% | 1.1% |
| **OlympiadBench** | 674 | **28.3%** | 42.3% | 32.8% | 3.7% |

### Notes

- Weights were extracted from the joint model using `extract_sub_model.py --sub_model_index 1` (model2 only, standard Qwen3ForCausalLM format).
- Training validation reported MATH-500 acc@1 = 61.8% at step 100. Offline mean@3 = 64.2% is consistent (n=3 sampling variance).
- All verification used `latex_semantic` method (100%).
- Total: 1516 prompts × 3 = 4548 generations, completed in 271s.
- OlympiadBench uses text-only, open-ended, English math competition subset (`OE_TO_maths_en_COMP`, 674 problems).
- Detailed per-response results saved to: `/data-1/model_weights/EXP-04_Joint-MiniRL-1.7B-MATH/step_100_model2/inference/`
- Results updated on 2026-03-16 (re-evaluation, replaces previous EVAL-01 run).

---

## EVAL-02: EXP-05 Baseline-MiniRL-1.7B step 200

| Field | Value |
|---|---|
| **Source Experiment** | EXP-05 (Baseline-MiniRL-Qwen3-1.7B-MATH) |
| **Model Weights** | `/data-1/model_weights/EXP-05_Baseline-MiniRL-1.7B-MATH/step_200` |
| **Checkpoint Step** | 200 (final) |
| **Sub-Model** | N/A (single model, no joint training) |
| **Inference Engine** | vLLM 0.8.5 (FLASH_ATTN backend, V1 engine, tp=4) |
| **Benchmarks** | MATH-500, AIME-2025, AMC-2023, MinervaMAth, OlympiadBench |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=3, max_tokens=4096 |
| **Date** | 2026-03-16 |

### Results

| Benchmark | Samples | mean@3 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|
| **MATH-500** | 500 | **61.4%** | 74.2% | 65.2% | 0.8% |
| **AIME-2025** | 30 | **5.6%** | 10.0% | 3.3% | 5.6% |
| **AMC-2023** | 40 | **41.7%** | 62.5% | 45.0% | 0.8% |
| **MinervaMAth** | 272 | **27.6%** | 36.0% | 29.8% | 0.2% |
| **OlympiadBench** | 674 | **28.1%** | 41.8% | 30.1% | 3.0% |

### Notes

- This is the baseline (no joint training) — same MiniRL algorithm on single Qwen3-1.7B-Base.
- Training used `grad_clip=1.0` which was too aggressive (see EXP-05 known issue in EXPERIMENT_INDEX.md).
- Detailed per-response results saved to: `/data-1/model_weights/EXP-05_Baseline-MiniRL-1.7B-MATH/step_200/inference/`

---

## Cross-Experiment Comparison (EVAL-01 vs EVAL-02)

| Benchmark | EXP-04 Joint (mean@3) | EXP-05 Baseline (mean@3) | Delta |
|---|---|---|---|
| **MATH-500** | **64.2%** | 61.4% | +2.8% |
| **AIME-2025** | 4.4% | **5.6%** | -1.2% |
| **AMC-2023** | 36.7% | **41.7%** | -5.0% |
| **MinervaMAth** | 24.4% | **27.6%** | -3.2% |
| **OlympiadBench** | **28.3%** | 28.1% | +0.2% |

**Observations**: Joint training (EXP-04, 100 steps) vs baseline (EXP-05, 200 steps). Joint model is better on MATH-500 (+2.8%) — its training dataset. Baseline outperforms on out-of-distribution benchmarks: AMC-2023 (-5.0%), MinervaMAth (-3.2%), AIME (-1.2%). Note that the baseline ran 2x more steps but with severely clipped gradients (effective lr ~2.8e-9 vs intended 1e-6).

---

<!--
Template for new entries:

## EVAL-XX: {description}

| Field | Value |
|---|---|
| **Source Experiment** | EXP-XX |
| **Model Weights** | /data-1/model_weights/{path} |
| **Checkpoint Step** | {N} |
| **Inference Engine** | vLLM |
| **Benchmarks** | {list} |
| **Generation Params** | temperature={T}, top_p={P}, max_tokens={N} |
| **Date** | YYYY-MM-DD |

### Results
| Benchmark | Metric | Value |
|---|---|---|
| ... | ... | ... |

### Notes
{any observations}
-->
