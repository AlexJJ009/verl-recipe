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

## EVAL-03: EXP-06 Baseline-MiniRL-1.7B-GC500 step 200

| Field | Value |
|---|---|
| **Source Experiment** | EXP-06 (Baseline-MiniRL-Qwen3-1.7B-MATH-GC500) |
| **Model Weights** | `/data-1/model_weights/EXP-06_Baseline-MiniRL-1.7B-MATH-GC500/step_200` |
| **Checkpoint Step** | 200 (final) |
| **Sub-Model** | N/A (single model, no joint training) |
| **Inference Engine** | vLLM 0.8.5 (FLASH_ATTN backend, V1 engine, tp=4) |
| **Benchmarks** | MATH-500, AIME-2025, AMC-2023, MinervaMAth, OlympiadBench |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=3, max_tokens=4096 |
| **Date** | 2026-03-16 |

### Results

| Benchmark | Samples | mean@3 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|
| **MATH-500** | 500 | **64.1%** | 76.2% | 69.2% | 0.1% |
| **AIME-2025** | 30 | **2.2%** | 3.3% | 3.3% | 0.0% |
| **AMC-2023** | 40 | **35.0%** | 52.5% | 40.0% | 0.0% |
| **MinervaMAth** | 272 | **24.8%** | 34.6% | 25.0% | 0.0% |
| **OlympiadBench** | 674 | **28.1%** | 42.6% | 30.6% | 0.5% |

### Notes

- This is the grad_clip=500.0 fix for EXP-05 (which used grad_clip=1.0). Same algorithm, same model, same steps.
- Training validation reported MATH-500 acc@1 = 60.0% at step 200, best = 64.2% at step 160. Offline mean@3 = 64.1% at step 200 is consistent.
- Detailed per-response results saved to: `/data-1/model_weights/EXP-06_Baseline-MiniRL-1.7B-MATH-GC500/step_200/inference/`

---

## EVAL-04: EXP-06 Baseline-MiniRL-1.7B-GC500 step 680 (resumed)

| Field | Value |
|---|---|
| **Source Experiment** | EXP-06 (Baseline-MiniRL-Qwen3-1.7B-MATH-GC500, resumed to 700 steps) |
| **Model Weights** | `/data-1/model_weights/EXP-06_Baseline-MiniRL-1.7B-MATH-GC500/step_680` |
| **Checkpoint Step** | 680 (last checkpoint of resumed run, ~696/700 steps completed) |
| **Sub-Model** | N/A (single model, no joint training) |
| **Inference Engine** | vLLM 0.8.5 (FLASH_ATTN backend, V1 engine, tp=4) |
| **Benchmarks** | MATH-500, AIME-2025, AMC-2023, MinervaMAth, OlympiadBench |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, max_tokens=4096 |
| **Date** | 2026-03-17 |

### Results (n=3)

| Benchmark | Samples | mean@3 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|
| **MATH-500** | 500 | **66.4%** | 77.8% | 70.2% | 0.1% |
| **AIME-2025** | 30 | **3.3%** | 6.7% | 3.3% | 0.0% |
| **AMC-2023** | 40 | **37.5%** | 55.0% | 47.5% | 0.0% |
| **MinervaMAth** | 272 | **26.1%** | 34.6% | 27.6% | 0.2% |
| **OlympiadBench** | 674 | **30.1%** | 42.0% | 33.5% | 1.0% |

### Results (n=8, multi-k)

| Benchmark | Samples | mean@8 | pass@1 | maj@1 | pass@2 | maj@2 | pass@4 | maj@4 | pass@8 | maj@8 | extraction_fail |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **MATH-500** | 500 | **66.9%** | 66.9% | 66.8% | 74.6% | 66.8% | 80.1% | 71.4% | 84.6% | 73.7% | 0.1% |
| **AIME-2025** | 30 | **3.3%** | 3.3% | 3.2% | 5.8% | 3.2% | 8.8% | 5.1% | 10.0% | 10.0% | 0.4% |
| **AMC-2023** | 40 | **43.1%** | 43.1% | 43.1% | 54.1% | 43.5% | 64.9% | 49.1% | 75.0% | 51.8% | 0.6% |
| **MinervaMAth** | 272 | **25.7%** | 25.7% | 25.8% | 31.8% | 25.9% | 37.4% | 27.9% | 43.0% | 29.6% | 0.3% |
| **OlympiadBench** | 674 | **30.3%** | 30.3% | 30.2% | 38.6% | 30.4% | 46.3% | 34.5% | 52.8% | 37.4% | 1.2% |

### Notes

- This is the same EXP-06 experiment resumed from step 200 to 700 (stopped at ~696 due to wandb teardown error). Last checkpoint at step 680.
- Training validation at step 680: MATH-500 acc@1 = 63.8%, AIME-2025 acc@1 = 3.8%. Offline mean@8 = 66.9% (n=8 run) is higher, likely due to sampling variance and different random seeds.
- Compared to EVAL-03 (same experiment, step 200): MATH-500 improved from 64.1% to **66.4%** (+2.3%), OlympiadBench from 28.1% to **30.1%** (+2.0%), AMC from 35.0% to **37.5%** (+2.5%). AIME-2025 improved from 2.2% to 3.3%.
- Best training validation MATH-500 was **67.2%** at step 460 (peak), suggesting step 680 is slightly past the MATH-500 optimum.
- n=3 results saved to: `/data-1/model_weights/EXP-06_Baseline-MiniRL-1.7B-MATH-GC500/step_680/inference/`
- n=8 results saved to: `/data-1/model_weights/EXP-06_Baseline-MiniRL-1.7B-MATH-GC500/step_680/inference_n8/`

---

## EVAL-05: EXP-07 Joint-MiniRL-1.7B-GC500-Dual step 200 (model2 only)

| Field | Value |
|---|---|
| **Source Experiment** | EXP-07 (Joint-MiniRL-Qwen3-1.7B-MATH-GC500-Dual-Step680) |
| **Model Weights** | `/data-1/model_weights/EXP-07_Joint-MiniRL-1.7B-MATH-GC500-Dual/step_200_model2` |
| **Checkpoint Step** | 200 (final) |
| **Sub-Model** | model2 (trainable, extracted from joint model) |
| **Inference Engine** | vLLM 0.8.5 (FLASH_ATTN backend, V1 engine, tp=4) |
| **Benchmarks** | MATH-500, AIME-2025, AMC-2023, MinervaMAth, OlympiadBench |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=8, max_tokens=4096 |
| **Date** | 2026-03-18 |

### Results (n=8, multi-k)

| Benchmark | Samples | mean@8 | pass@1 | maj@1 | pass@2 | maj@2 | pass@4 | maj@4 | pass@8 | maj@8 | extraction_fail |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **MATH-500** | 500 | **66.1%** | 66.1% | 66.1% | 73.7% | 69.0% | 79.6% | 73.4% | 84.2% | 75.8% | 16.2% |
| **AIME-2025** | 30 | **3.8%** | 3.8% | 3.7% | 5.4% | 4.3% | 6.4% | 5.4% | 6.7% | 6.7% | 61.3% |
| **AMC-2023** | 40 | **36.2%** | 36.2% | 36.3% | 43.5% | 39.8% | 50.3% | 44.1% | 57.5% | 45.0% | 34.1% |
| **MinervaMAth** | 272 | **24.6%** | 24.6% | 24.7% | 31.0% | 25.8% | 37.7% | 27.1% | 44.9% | 29.4% | 11.4% |
| **OlympiadBench** | 674 | **29.1%** | 29.1% | 29.2% | 36.4% | 32.1% | 43.4% | 35.9% | 50.1% | 38.9% | 38.1% |

### Notes

- Weights were extracted from the EXP-07 joint checkpoint and evaluated on model2 only.
- Training validation at step 200 was most likely already `model2-only` rather than fused joint logits: in the current code path, `_validate()` switches joint checkpoints to `eval_only=True`, and that logic predates the 2026-03-17 EXP-07 run. The archived run does not record an exact git commit / working-tree snapshot, so this cannot be proven with absolute certainty from the log alone. This offline run still differs because it evaluates extracted HF `step_200_model2` weights through vLLM rather than the trainer-integrated validation path.
- Compared to EVAL-04 (EXP-06 step 680, n=8 mean@8), EXP-07 final model2 is lower on MATH-500 (66.1% vs 66.9%), AMC-2023 (36.2% vs 43.1%), MinervaMAth (24.6% vs 25.7%), and OlympiadBench (29.1% vs 30.3%), with only a small AIME-2025 gain (3.8% vs 3.3%).
- Extraction failures are substantially higher than prior runs, especially on AIME-2025 (61.3%), AMC-2023 (34.1%), and OlympiadBench (38.1%), which likely suppressed pass/maj metrics.
- Total: 1516 prompts × 8 = 12128 generations, completed in 1631s.
- Raw results saved to: `/data-1/model_weights/EXP-07_Joint-MiniRL-1.7B-MATH-GC500-Dual/step_200_model2/inference_n8/`

---

## EVAL-06: EXP-08 Qwen3-4B-Base (Pretrained Baseline)

| Field | Value |
|---|---|
| **Source Experiment** | EXP-08 (Qwen3-4B-Base, Pretrained Baseline) |
| **Model Weights** | `/data-1/.cache/huggingface/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539` |
| **Checkpoint Step** | N/A (pretrained) |
| **Sub-Model** | N/A (single model) |
| **Inference Engine** | vLLM 0.12.0 (FLASH_ATTN backend, V1 engine, tp=8) |
| **Benchmarks** | MATH-500, AIME-2025, AMC-2023, AQUA, GSM8K, MAWPS, SVAMP (all with system prompt) |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=3, max_tokens=4096 |
| **Date** | 2026-04-05 |

### Results (n=3)

| Benchmark | Samples | mean@3 | pass@1 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|---|
| **MATH-500** | 500 | **32.5%** | 32.5% | 60.8% | 50.6% | 39.8% |
| **AIME-2025** | 30 | **3.3%** | 3.3% | 10.0% | 3.3% | 40.0% |
| **AMC-2023** | 40 | **15.0%** | 15.0% | 30.0% | 22.5% | 37.5% |
| **AQUA** | 254 | **6.8%** | 6.8% | 18.1% | 13.4% | 69.0% |
| **GSM8K** | 1319 | **27.8%** | 27.8% | 60.0% | 52.8% | 59.3% |
| **MAWPS** | 355 | **21.7%** | 21.7% | 51.0% | 46.8% | 70.1% |
| **SVAMP** | 300 | **25.9%** | 25.9% | 58.0% | 53.0% | 66.0% |

### Notes

- This is the raw pretrained Qwen3-4B-Base with no training at all — serves as the zero-training baseline.
- All datasets use `_with_system_prompt` variants with unified `\boxed{}` instruction in user prompt.
- Extraction failure is high (38-70%) because the pretrained base model has not been trained to follow the `\boxed{}` format consistently.
- Total: 2798 prompts × 3 = 8394 generations, completed in 1414s.
- Raw results saved to: `/data-1/eval_results/qwen3-4b-base_n3_sysprompt/`

---

## EVAL-07: EXP-09 Qwen3-4B-DPO (Base-Sourced Preference Pairs)

| Field | Value |
|---|---|
| **Source Experiment** | EXP-09 (Qwen3-4B-DPO, External DPO Training) |
| **Model Weights** | `/data-1/checkpoints/qwen3-4b-dpo` |
| **Checkpoint Step** | 376 (final) |
| **Sub-Model** | N/A (single model) |
| **Inference Engine** | vLLM 0.12.0 (FLASH_ATTN backend, V1 engine, tp=8) |
| **Benchmarks** | MATH-500, AIME-2025, AMC-2023, AQUA, GSM8K, MAWPS, SVAMP (all with system prompt) |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=3, max_tokens=4096 |
| **Date** | 2026-04-05 |

### Results (n=3)

| Benchmark | Samples | mean@3 | pass@1 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|---|
| **MATH-500** | 500 | **34.1%** | 34.1% | 66.0% | 51.6% | 35.5% |
| **AIME-2025** | 30 | **2.2%** | 2.2% | 6.7% | 6.7% | 41.1% |
| **AMC-2023** | 40 | **20.0%** | 20.0% | 47.5% | 27.5% | 27.5% |
| **AQUA** | 254 | **7.3%** | 7.3% | 20.5% | 13.0% | 59.1% |
| **GSM8K** | 1319 | **35.0%** | 35.0% | 69.1% | 60.7% | 50.1% |
| **MAWPS** | 355 | **24.9%** | 24.9% | 56.1% | 50.1% | 66.0% |
| **SVAMP** | 300 | **27.9%** | 27.9% | 61.0% | 54.3% | 59.9% |

### Notes

- DPO trained from Qwen3-4B-Base using preference pairs generated from the Base model itself (not SFT).
- Compared to EVAL-06 (Base): DPO shows modest gains on most benchmarks (MATH-500 +1.6%, GSM8K +7.2%, AMC +5.0%, MAWPS +3.2%, SVAMP +2.0%), but AIME-2025 dropped from 3.3% to 2.2%.
- Extraction failure improved slightly vs Base (35.5% vs 39.8% on MATH-500, 50.1% vs 59.3% on GSM8K), suggesting DPO partially learned to follow `\boxed{}` format.
- Both extraction failure rates remain much higher than verl-trained 1.7B models (EVAL-01 through EVAL-05 had <5% on MATH-500), indicating 4B base→DPO pipeline has significant format compliance issues.
- DPO training summary: final_loss=0.138, final_margins=3.99, 376 steps, 1 epoch over 6013 pairs.
- Total: 2798 prompts × 3 = 8394 generations, completed in 1218s.
- Raw results saved to: `/data-1/eval_results/qwen3-4b-dpo_n3_sysprompt_v2/`

---

## EVAL-08: EXP-10 Qwen3-8B-DPO (External DPO Training)

| Field | Value |
|---|---|
| **Source Experiment** | EXP-10 (Qwen3-8B-DPO, External DPO Training) |
| **Model Weights** | `/data-1/checkpoints/qwen3-8b-dpo` |
| **Checkpoint Step** | 496 (final) |
| **Sub-Model** | N/A (single model) |
| **Inference Engine** | vLLM 0.12.0 (FLASH_ATTN backend, V1 engine, tp=8) |
| **Benchmarks** | MATH-500, AIME-2025, AMC-2023, AQUA, GSM8K, MAWPS, SVAMP (all with system prompt) |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=3, max_tokens=4096 |
| **Date** | 2026-04-05 |

### Results (n=3)

| Benchmark | Samples | mean@3 | pass@1 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|---|
| **MATH-500** | 500 | **51.1%** | 51.1% | 76.6% | 63.0% | 13.5% |
| **AIME-2025** | 30 | **8.9%** | 8.9% | 13.3% | 10.0% | 17.8% |
| **AMC-2023** | 40 | **30.0%** | 30.0% | 52.5% | 35.0% | 12.5% |
| **AQUA** | 254 | **5.1%** | 5.1% | 14.2% | 6.7% | 31.9% |
| **GSM8K** | 1319 | **57.2%** | 57.2% | 88.3% | 75.6% | 20.7% |
| **MAWPS** | 355 | **72.8%** | 72.8% | 93.8% | 87.6% | 16.1% |
| **SVAMP** | 300 | **70.2%** | 70.2% | 92.7% | 87.3% | 16.2% |

### Notes

- DPO trained from Qwen3-8B-Base using TRL (beta=0.1), 7,934 preference pairs from `/data-1/dataset/dpo/dpo-8b/dpo-8b-pairs.jsonl`.
- Training: lr=5e-7, 1 epoch (~496 steps), effective batch=16, max_length=2048, cosine scheduler.
- Final metrics: loss=0.130, margins=4.51, rewards_chosen=3.11, rewards_rejected=-1.41.
- Compared to EVAL-07 (4B DPO): 8B DPO is substantially stronger across all benchmarks — MATH-500 51.1% vs 34.1% (+17.0%), GSM8K 57.2% vs 35.0% (+22.2%), MAWPS 72.8% vs 24.9% (+47.9%), SVAMP 70.2% vs 27.9% (+42.3%).
- Extraction failure is lower than 4B DPO (13-32% vs 28-66%), indicating better format compliance from the larger model.
- Total: 2798 prompts × 3 = 8394 generations, completed in 488s.
- Raw results saved to: `/data-1/checkpoints/qwen3-8b-dpo/inference_n3/`

---

## EVAL-09: EXP-11 Qwen3-4B-SFT-DPO (SFT→DPO Pipeline)

| Field | Value |
|---|---|
| **Source Experiment** | EXP-11 (Qwen3-4B-SFT-DPO, External DPO on SFT Checkpoint) |
| **Model Weights** | `/data-1/checkpoints/qwen3-4b-sft-dpo` |
| **Checkpoint Step** | 365 (final) |
| **Sub-Model** | N/A (single model) |
| **Inference Engine** | vLLM 0.12.0 (FLASH_ATTN backend, V1 engine, tp=8) |
| **Benchmarks** | MATH-500, AIME-2025, AMC-2023, AQUA, GSM8K, MAWPS, SVAMP (all with system prompt) |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, n=3, max_tokens=4096 |
| **Date** | 2026-04-06 |

### Results (n=3)

| Benchmark | Samples | mean@3 | pass@1 | pass@3 | maj@3 | extraction_fail |
|---|---|---|---|---|---|---|
| **MATH-500** | 500 | **67.7%** | 67.7% | 80.2% | 78.4% | 29.9% |
| **AIME-2025** | 30 | **7.8%** | 7.8% | 13.3% | 13.3% | 87.8% |
| **AMC-2023** | 40 | **40.8%** | 40.8% | 52.5% | 52.5% | 57.5% |
| **AQUA** | 254 | **65.0%** | 65.0% | 80.7% | 76.4% | 23.2% |
| **GSM8K** | 1319 | **89.8%** | 89.8% | 94.8% | 92.5% | 3.7% |
| **MAWPS** | 355 | **94.4%** | 94.4% | 96.3% | 96.3% | 1.6% |
| **SVAMP** | 300 | **90.7%** | 90.7% | 95.0% | 93.3% | 4.0% |

### Notes

- DPO trained from Qwen3-4B-Base-SFT-stage-1 (SFT checkpoint, not original base) using TRL (beta=0.1), 5,860 preference pairs from `/data-1/dataset/dpo/dpo-4b-sft/dpo-4b-sft-pairs.jsonl`.
- Training: lr=5e-7, 1 epoch (~365 steps), effective batch=16, max_length=2048, cosine scheduler.
- Final metrics: loss=0.460, margins=0.57, rewards_chosen=-0.014, rewards_rejected=-0.582.
- Compared to EVAL-07 (4B Base→DPO): SFT→DPO dramatically improves across all benchmarks — MATH-500 67.7% vs 34.1% (+33.6%), GSM8K 89.8% vs 35.0% (+54.8%), MAWPS 94.4% vs 24.9% (+69.5%), SVAMP 90.7% vs 27.9% (+62.8%), AQUA 65.0% vs 7.3% (+57.7%), AMC 40.8% vs 20.0% (+20.8%), AIME 7.8% vs 2.2% (+5.6%).
- Compared to EVAL-08 (8B Base→DPO): SFT→DPO 4B surpasses 8B DPO on MATH-500 (67.7% vs 51.1%), GSM8K (89.8% vs 57.2%), MAWPS (94.4% vs 72.8%), SVAMP (90.7% vs 70.2%), AQUA (65.0% vs 5.1%), AMC (40.8% vs 30.0%). Only trails on AIME (7.8% vs 8.9%).
- Extraction failure is mixed: low on GSM8K/MAWPS/SVAMP (1.6-4.0%), moderate on MATH-500/AQUA (23-30%), high on AMC/AIME (58-88%). The SFT base improves format compliance vs Base→DPO on easy benchmarks but not on competition-level ones.
- Total: 2798 prompts × 3 = 8394 generations, completed in 1286s.
- Raw results saved to: `/data-1/checkpoints/qwen3-4b-sft-dpo/inference_n3/`

---

## EVAL-10: EXP-12 Qwen3-1.7B Post-Trained/Instruct Step-Zero Format Screen

| Field | Value |
|---|---|
| **Source Experiment** | EXP-12 (Qwen3-1.7B Post-Trained/Instruct Step-Zero Baseline) |
| **Model Weights** | `/data-1/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| **Checkpoint Step** | 0 (official raw post-trained model) |
| **Sub-Model** | N/A (single model) |
| **Inference Engine** | vLLM 0.12.0 (FLASHINFER backend, V1 engine, tp=8) |
| **Benchmarks** | MATH-500, AIME-2025, AMC-2023, AQUA, GSM8K, MAWPS, SVAMP (all with system prompt) |
| **Generation Params** | temperature=0.2, top_p=0.95, top_k=-1, n=1, max_tokens=4096, seed=20260719 |
| **Date** | 2026-07-19 |

### Results (n=1)

| Benchmark | Samples | mean@1 | think complete | answer complete | boxed extraction | EOS | truncation |
|---|---:|---:|---:|---:|---:|---:|---:|
| **MATH-500** | 500 | **52.6%** | 74.0% | 66.6% | 57.4% | 74.2% | 25.8% |
| **AIME-2025** | 30 | **6.7%** | 13.3% | 3.3% | 13.3% | 6.7% | 93.3% |
| **AMC-2023** | 40 | **37.5%** | 47.5% | 40.0% | 40.0% | 47.5% | 52.5% |
| **AQUA** | 254 | **33.1%** | 81.5% | 76.4% | 34.3% | 80.3% | 19.7% |
| **GSM8K** | 1319 | **49.3%** | 95.8% | 91.0% | 55.7% | 96.7% | 3.3% |
| **MAWPS** | 355 | **39.2%** | 99.7% | 96.9% | 41.7% | 100.0% | 0.0% |
| **SVAMP** | 300 | **39.3%** | 99.3% | 97.3% | 42.7% | 99.7% | 0.3% |
| **Math-7 macro** | 7 datasets | **36.8%** | 73.0% | 67.4% | 40.7% | 72.1% | 27.9% |

### Format Decision

- Response-level micro rates across all 2,798 prompts: accuracy 45.4%, think complete 89.9%, answer complete 85.1%, boxed extraction 50.2%, reward-grader success 100.0%, EOS 90.2%, truncation 9.8%.
- Only 1,241/2,798 responses (44.4%) simultaneously passed both tag checks, boxed extraction, reward-grader execution, and EOS.
- 1,140/2,798 responses were non-truncated but still lacked a boxed answer. Typical failure: `<answer> 81 </answer>` instead of `<answer> \\boxed{81} </answer>`.
- Competition-level prompts also exhibit severe length failure: AIME-2025 truncation 93.3%, AMC-2023 52.5%, MATH-500 25.8%.
- The reward grader executed successfully for every response; the dominant failures are model output format and excessive response length, not grader crashes.
- Decision: the raw model fails the frozen format gate by a wide margin. Supervised format cold start is required before Stage1 Model1 selection.
- Generation completed in 799.8 seconds. Raw results: `/data-2/model_weights/math_task/qwen3_1p7b_cold_start_v1/validation/step_0_n1/`.

---

## Cross-Experiment Comparison (EVAL-01 through EVAL-10)

### 1.7B Models (verl-trained, mean@3 or mean@8)

| Benchmark | EXP-04 Joint 100s (mean@3) | EXP-05 Baseline gc=1.0 200s (mean@3) | EXP-06 Baseline gc=500 200s (mean@3) | EXP-06 Baseline gc=500 680s (mean@8) | EXP-07 Joint gc=500 dual 200s (mean@8) |
|---|---|---|---|---|---|
| **MATH-500** | 64.2% | 61.4% | 64.1% | **66.9%** | 66.1% |
| **AIME-2025** | 4.4% | **5.6%** | 2.2% | 3.3% | 3.8% |
| **AMC-2023** | 36.7% | 41.7% | 35.0% | **43.1%** | 36.2% |
| **MinervaMAth** | 24.4% | **27.6%** | 24.8% | 25.7% | 24.6% |
| **OlympiadBench** | 28.3% | 28.1% | 28.1% | **30.3%** | 29.1% |

### 4B/8B Models: Base vs DPO vs SFT→DPO (mean@3, n=3, system prompt + boxed instruction)

| Benchmark | EXP-08 Qwen3-4B-Base | EXP-09 Qwen3-4B-DPO | EXP-10 Qwen3-8B-DPO | EXP-11 Qwen3-4B-SFT-DPO |
|---|---|---|---|---|
| **MATH-500** | 32.5% | 34.1% | 51.1% | **67.7%** |
| **AIME-2025** | 3.3% | 2.2% | **8.9%** | 7.8% |
| **AMC-2023** | 15.0% | 20.0% | 30.0% | **40.8%** |
| **AQUA** | 6.8% | 7.3% | 5.1% | **65.0%** |
| **GSM8K** | 27.8% | 35.0% | 57.2% | **89.8%** |
| **MAWPS** | 21.7% | 24.9% | 72.8% | **94.4%** |
| **SVAMP** | 25.9% | 27.9% | 70.2% | **90.7%** |

**Observations**:

*1.7B verl-trained models*: EVAL-04 (EXP-06 step 680) remains the strongest 1.7B checkpoint overall, leading on MATH-500 (66.9%), AMC-2023 (43.1%), and OlympiadBench (30.3%).

*Qwen3-1.7B step-zero format screen*: EVAL-10 is not directly comparable to the
trained-model `mean@3/mean@8` table because it uses `n=1` as a format-screening
run. It demonstrates that the official post-trained model is not an admissible
format init: only 44.4% of responses pass the complete format contract and
boxed extraction succeeds on only 50.2% of responses.

*4B/8B DPO pipeline comparison*: The SFT→DPO pipeline (EXP-11) is dramatically stronger than both Base→DPO approaches. The 4B SFT→DPO model surpasses even the 8B Base→DPO on 6 of 7 benchmarks: MATH-500 67.7% vs 51.1% (+16.6%), GSM8K 89.8% vs 57.2% (+32.6%), MAWPS 94.4% vs 72.8% (+21.6%), SVAMP 90.7% vs 70.2% (+20.5%), AQUA 65.0% vs 5.1% (+59.9%), AMC 40.8% vs 30.0% (+10.8%). Only AIME-2025 (7.8% vs 8.9%) slightly trails the 8B DPO. This confirms the critical importance of SFT pre-training before DPO: the SFT checkpoint provides format compliance and reasoning capability that DPO alone cannot instill. Notably, the 4B SFT→DPO now matches verl-trained 1.7B models on MATH-500 (67.7% vs 66.9%).

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
