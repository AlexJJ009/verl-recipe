# Experiment Index

This table tracks all training experiments, their logs, checkpoints, and merged model weights.

**Maintenance rules**: See `docs/joint_training/constraints/experiment_tracking/experiment_index_spec.md`

---

## Active Experiments

### EXP-01: Joint-GRPO-Qwen3-1.7B-GSM8K

| Field | Value |
|---|---|
| **Script** | `run_joint_grpo_qwen3_1.7b.sh` |
| **Goal** | Validate joint-training runtime on simple math (GSM8K) with vanilla GRPO |
| **Algorithm** | GRPO (vanilla, token-mean, no KL) |
| **Model** | QwenJoint-1.7B (lambda=0.5) |
| **Dataset** | GSM8K (train) / GSM8K (test) |
| **Key Params** | lr=1e-6, warmup=5, batch=32, n_resp=4, max_resp_len=1024, clip=[0.2,0.28], steps=100, 8 GPUs |
| **Logs** | `Joint-GRPO-Qwen3-1.7B-GSM8K_1773026491.log` (completed: 0% acc) |
| | `Joint-GRPO-Qwen3-1.7B-GSM8K_1773032262.log` (completed: 0% acc) |
| | `Joint-GRPO-Qwen3-1.7B-GSM8K_1773041769.log` (stalled at vLLM init) |
| **Checkpoint** | `/data-1/checkpoints/Joint-GRPO-Qwen3-1.7B-GSM8K_1773500863/` (137 GB, steps 20-101) |
| **Note** | Checkpoint dir was created by the earlier `_1773498644` run which resumed into it. All GSM8K runs achieved 0% accuracy -- joint model never learned GSM8K with vanilla GRPO. |
| **Model Weights** | (not yet merged) |
| **W&B** | Project: `JointTraining`, Run: [`6nxypu0j`](https://wandb.ai/gongxunli-beihang-universally/JointTraining/runs/6nxypu0j) (synced 2026-03-16) |
| **Status** | Concluded (abandoned due to 0% accuracy) |

---

### EXP-02: Joint-GRPO-Qwen3-4B-RolloutCorr-MATH

| Field | Value |
|---|---|
| **Script** | `run_joint_grpo_qwen3_4b_rollout_corr.sh` |
| **Goal** | Test 4B joint model on MATH with sequence-level importance sampling to address train-inference mismatch |
| **Algorithm** | GRPO + sequence-level IS (threshold=2.0) |
| **Model** | QwenJoint-4B (lambda=0.5) |
| **Dataset** | MATH (train) / AIME-2025 (test) |
| **Key Params** | lr=1e-6, warmup=5, batch=8 (original), n_resp=8, max_resp_len=4096, clip=[0.2,0.28], steps=200, offload=True, gpu_mem=0.50, 8 GPUs |
| **Logs** | `Joint-GRPO-Qwen3-4B-RolloutCorr-MATH_1773505000.log` (completed: 200/200 steps) |
| **Checkpoint** | `/data-1/checkpoints/Joint-GRPO-Qwen3-4B-RolloutCorr-MATH_1773505000/` (198 GB, steps 20-200) |
| **Best Metric** | AIME-2025 acc@1: **23.1%** (peak), **7.7%** (final step 200) |
| **Model Weights** | (not yet merged) |
| **W&B** | Project: `JointTraining`, Run: [`qg8ezoj6`](https://wandb.ai/gongxunli-beihang-universally/JointTraining/runs/qg8ezoj6) (synced 2026-03-16) |
| **Status** | Completed |

---

### EXP-03: Joint-GRPO-Qwen3-4B-RolloutCorr-MATH-bsz16

| Field | Value |
|---|---|
| **Script** | `run_joint_grpo_qwen3_4b_rollout_corr.sh` (with batch_size override to 16) |
| **Goal** | Rerun EXP-02 with larger batch size (16 vs 8) to reduce zero-advantage batches |
| **Algorithm** | GRPO + sequence-level IS (threshold=2.0) |
| **Model** | QwenJoint-4B (lambda=0.5) |
| **Dataset** | MATH (train) / AIME-2025 (test) |
| **Key Params** | lr=1e-6, warmup=5, **batch=16**, n_resp=8, max_resp_len=4096, clip=[0.2,0.28], steps=200, offload=True, gpu_mem=0.50, 8 GPUs |
| **Logs** | `Joint-GRPO-Qwen3-4B-RolloutCorr-MATH-bsz16_1773544834.log` (killed by SIGTERM at step 19) |
| | `Joint-GRPO-Qwen3-4B-RolloutCorr-MATH-bsz16_1773548609.log` (interrupted at step 145/200) |
| **Checkpoint** | `/data-1/checkpoints/Joint-GRPO-Qwen3-4B-RolloutCorr-MATH-bsz16_1773548609/` (198 GB, steps 20-140) |
| **Best Metric** | AIME-2025 acc@1: **11.5%** (step 145) |
| **Model Weights** | (not yet merged) |
| **W&B** | Project: `JointTraining`, Run: [`nydl9cen`](https://wandb.ai/gongxunli-beihang-universally/JointTraining/runs/nydl9cen) (synced 2026-03-16) |
| **Status** | Interrupted at step 145 |

---

### EXP-04: Joint-MiniRL-Qwen3-1.7B-MATH

| Field | Value |
|---|---|
| **Script** | `run_joint_minirl_qwen3_1.7b_math.sh` |
| **Goal** | Test MiniRL loss (binary clip + REINFORCE gradient) with token-level IS on joint 1.7B model |
| **Algorithm** | MiniRL + Dr.GRPO advantage + token-level IS (threshold=5.0), seq-mean-token-sum aggregation |
| **Model** | QwenJoint-1.7B (lambda=0.55) |
| **Dataset** | MATH (train) / MATH-500 + AIME-2025 (test) |
| **Key Params** | lr=1e-6, warmup=5, batch=32, n_resp=8, max_resp_len=4096, clip=[0.2,0.27], steps=100, gpu_mem=0.60, 8 GPUs |
| **Logs** | `Joint-MiniRL-Qwen3-1.7B-MATH_1773581076.log` (completed: 100/100 steps) |
| **Checkpoint** | `/data-1/checkpoints/Joint-MiniRL-Qwen3-1.7B-MATH_1773581076/` (91 GB, steps 20-100) |
| **Best Metric** | MATH-500 acc@1: **63.0%** (step 80), final **61.8%** (step 100); AIME-2025 acc@1: 3.8% |
| **MATH-500 Progression** | 44.9% → 55.3% → 57.1% → 58.8% → 59.6% → 62.2% → **63.0%** → 61.7% → 61.8% |
| **Model Weights** | `/data-1/model_weights/EXP-04_Joint-MiniRL-1.7B-MATH/step_100` (joint), `step_100_model2` (model2 extracted) |
| **Inference** | EVAL-01 in `INFERENCE_RESULTS.md` (5 benchmarks: MATH-500, AIME-2025, AMC-2023, MinervaMAth, OlympiadBench) |
| **W&B** | Project: `JointTraining`, Run: [`2opyranp`](https://wandb.ai/gongxunli-beihang-universally/JointTraining/runs/2opyranp) (synced 2026-03-16) |
| **Status** | Completed |

---

### EXP-05: Baseline-MiniRL-Qwen3-1.7B-MATH

| Field | Value |
|---|---|
| **Script** | `run_baseline_minirl_qwen3_1.7b_math.sh` |
| **Goal** | Baseline comparison: same MiniRL algorithm but on a single Qwen3-1.7B-Base (no joint training) |
| **Algorithm** | MiniRL + Dr.GRPO advantage + token-level IS (threshold=5.0), seq-mean-token-sum aggregation |
| **Model** | Qwen3-1.7B-Base (NO joint training) |
| **Dataset** | MATH (train) / MATH-500 + AIME-2025 (test) |
| **Key Params** | lr=1e-6, warmup=5, batch=32, n_resp=8, max_resp_len=4096, clip=[0.2,0.27], **grad_clip=1.0**, steps=200, gpu_mem=0.55, **4 GPUs** |
| **Logs** | `Baseline-MiniRL-Qwen3-1.7B-MATH_1773625595.log` (completed: 200/200 steps, ~3h50m) |
| **Checkpoint** | `/data-1/checkpoints/Baseline-MiniRL-Qwen3-1.7B-MATH_1773625595/` (41 GB, steps 20-200 every 20) |
| **Best Metric** | MATH-500 acc@1: **64.0%** (steps 145, 150, 200); AIME-2025 acc@1: **11.5%** (steps 100, 140) |
| **MATH-500 Progression** | 42.5% → 52.1% → 54.9% → 55.9% → 58.6% → 57.1% → 57.3% → 61.2% → 61.2% → 58.1% → 59.0% → 61.0% → ... → **64.0%** (step 200) |
| **Model Weights** | `/data-1/model_weights/EXP-05_Baseline-MiniRL-1.7B-MATH/step_200` (3.8 GB) |
| **Inference** | EVAL-02 in `INFERENCE_RESULTS.md` (5 benchmarks) |
| **Known Issue** | `grad_clip=1.0` is too aggressive for MiniRL's `seq-mean-token-sum` aggregation — pre-clip grad_norms of 231-515 are clipped to 1.0 every step, reducing effective lr from 1e-6 to ~2.8e-9. See `docs/joint_training/reports/baseline_minirl_qwen3_1.7b_math_analysis.md` Section 6. |
| **W&B** | Project: `JointTraining`, Run: [`43fk0git`](https://wandb.ai/gongxunli-beihang-universally/JointTraining/runs/43fk0git) (synced 2026-03-16) |
| **Status** | Completed |

---

### EXP-06: Baseline-MiniRL-Qwen3-1.7B-MATH-GC500

| Field | Value |
|---|---|
| **Script** | `run_baseline_minirl_qwen3_1.7b_math.sh` (with `RUN_PREFIX` override) |
| **Goal** | Rerun EXP-05 with `grad_clip=500.0` to fix the gradient clipping bottleneck identified in EXP-05 analysis |
| **Algorithm** | MiniRL + Dr.GRPO advantage + token-level IS (threshold=5.0), seq-mean-token-sum aggregation |
| **Model** | Qwen3-1.7B-Base (NO joint training) |
| **Dataset** | MATH (train) / MATH-500 + AIME-2025 (test) |
| **Key Params** | lr=1e-6, warmup=5, batch=32, n_resp=8, max_resp_len=4096, clip=[0.2,0.27], **grad_clip=500.0**, steps=200, gpu_mem=0.55, **4 GPUs** |
| **Logs** | (pending launch) |
| **Checkpoint** | (pending launch) |
| **Best Metric** | (pending) |
| **Model Weights** | (not yet merged) |
| **W&B** | (pending launch) |
| **Status** | Planned

---

### EXP-00: GRPO-Example-Script (Reference Only)

| Field | Value |
|---|---|
| **Script** | `GRPO-Example-Script.sh` |
| **Goal** | Stage-3 GRPO training template from TSPO project (Qwen3-4B-Base SFT) |
| **Algorithm** | GRPO (vanilla, token-mean) with KL loss (coef=0.001) |
| **Model** | Qwen3-4B-Base SFT (`/data-1/huggingface_cache/hub/stage1_m1`) |
| **Dataset** | DAPO-Math-17k / AIME-2024 + AIME-2025 + MATH-500 (test) |
| **Key Params** | lr=1e-6, warmup=10, batch=128, n_resp=16, max_resp_len=6144, clip=[0.2,0.28], steps=500, tp=2 |
| **Checkpoint Dir** | `/data-2/checkpoints/experimental/TSPO_Refined_Experiments/` |
| **Status** | Reference script (not a joint-training experiment) |

---

## Parameter Comparison Matrix

| Parameter | EXP-01 (GRPO 1.7B GSM8K) | EXP-02 (GRPO 4B MATH) | EXP-03 (GRPO 4B bsz16) | EXP-04 (MiniRL 1.7B) | EXP-05 (Baseline) | EXP-06 (Baseline GC500) |
|---|---|---|---|---|---|---|
| Model Size | 1.7B | 4B | 4B | 1.7B | 1.7B | 1.7B |
| Joint Training | Yes (λ=0.5) | Yes (λ=0.5) | Yes (λ=0.5) | Yes (λ=0.55) | **No** | **No** |
| Loss Mode | vanilla | vanilla | vanilla | minirl | minirl | minirl |
| Loss Agg | token-mean | token-mean | token-mean | seq-mean-token-sum | seq-mean-token-sum | seq-mean-token-sum |
| Dataset | GSM8K | MATH | MATH | MATH | MATH | MATH |
| Batch Size | 32 | 8 | **16** | 32 | 32 | 32 |
| Responses/Prompt | 4 | 8 | 8 | 8 | 8 | 8 |
| Max Resp Len | 1024 | 4096 | 4096 | 4096 | 4096 | 4096 |
| Clip Ratio | [0.2, 0.28] | [0.2, 0.28] | [0.2, 0.28] | [0.2, 0.27] | [0.2, 0.27] | [0.2, 0.27] |
| Grad Clip | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | **500.0** |
| Total Steps | 100 | 200 | 200 | 100 | 200 | 200 |
| Rollout IS | None | sequence (2.0) | sequence (2.0) | token (5.0) | token (5.0) | token (5.0) |
| Dr.GRPO | N/A | N/A | N/A | Yes | Yes | Yes |
| Offload | No | Yes | Yes | No | No | No |
| GPU Mem | 0.75 | 0.50 | 0.50 | 0.60 | 0.55 | 0.55 |
| GPUs | 8 | 8 | 8 | 8 | **4** | **4** |

---

## Checkpoint Inventory

| Checkpoint Path | Experiment | Size | Steps Saved | Status |
|---|---|---|---|---|
| `/data-1/checkpoints/Joint-GRPO-Qwen3-1.7B-GSM8K_1773500863/` | EXP-01 | 137 GB | 20,40,60,80,100,101 | Concluded (0% acc) |
| `/data-1/checkpoints/Joint-GRPO-Qwen3-4B-RolloutCorr-MATH_1773505000/` | EXP-02 | 198 GB | 20-200 (every 20) | Completed |
| `/data-1/checkpoints/Joint-GRPO-Qwen3-4B-RolloutCorr-MATH-bsz16_1773548609/` | EXP-03 | 198 GB | 20-140 (every 20) | Interrupted |
| `/data-1/checkpoints/Joint-MiniRL-Qwen3-1.7B-MATH_1773581076/` | EXP-04 | 91 GB | 20-100 (every 20) | Completed |
| `/data-1/checkpoints/Baseline-MiniRL-Qwen3-1.7B-MATH_1773625595/` | EXP-05 | 41 GB | 20-200 (every 20) | Completed |

**Total checkpoint disk usage: ~665 GB**

---

## Model Weights Inventory

Target directory: `/data-1/model_weights/`

Recommended folder structure:
```
/data-1/model_weights/
├── EXP-02_Joint-GRPO-4B-MATH/
│   ├── step_200/        # final
│   └── step_130/        # best AIME acc (~23%)
├── EXP-03_Joint-GRPO-4B-MATH-bsz16/
│   └── step_140/        # last available
├── EXP-04_Joint-MiniRL-1.7B-MATH/
│   ├── step_100/        # final
│   └── step_80/         # best MATH-500 acc (63%)
├── EXP-05_Baseline-MiniRL-1.7B-MATH/
│   └── (pending merge)
└── EXP-06_Baseline-MiniRL-1.7B-MATH-GC500/
    └── (pending completion)
```

| Weight Path | Source Experiment | Checkpoint Step | Merge Status |
|---|---|---|---|
| `/data-1/model_weights/EXP-04_Joint-MiniRL-1.7B-MATH/step_100` | EXP-04 | 100 (final) | Merged (joint, 7.6 GB) |
| `/data-1/model_weights/EXP-04_Joint-MiniRL-1.7B-MATH/step_100_model2` | EXP-04 | 100 (final) | Extracted model2 (3.8 GB) |
| `/data-1/model_weights/EXP-05_Baseline-MiniRL-1.7B-MATH/step_200` | EXP-05 | 200 (final) | Merged (single model, 3.8 GB) |

---

## Deletion Log

| Date | Item Deleted | Experiment | Reason |
|---|---|---|---|
| 2026-03-16 | `Joint-GRPO-Qwen3-1.7B-GSM8K_1772760550.log` | EXP-01 | Crash log (disk full on /data-2), no useful data |
| 2026-03-16 | `Joint-GRPO-Qwen3-1.7B-GSM8K_1773025400.log` | EXP-01 | Crash log (/tmp full), no useful data |
| 2026-03-16 | `Joint-GRPO-Qwen3-1.7B-GSM8K_1773498644.log` | EXP-01 | Killed at step 21, 0% acc, superseded by later runs |
| 2026-03-16 | `Joint-GRPO-Qwen3-1.7B-GSM8K_1773500863.log` | EXP-01 | Crash at startup (Hydra parse error), no useful data |
