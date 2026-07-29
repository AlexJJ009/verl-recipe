# Rebuttal MATH RLVR

This family compares ordinary SFT and paper **offline WDL-SFT** initialization
under one frozen standard-GRPO recipe. The scientific configuration is
`frozen_grpo_v2.env`: vanilla PPO loss, symmetric `0.2` clip, dual-clip C
`3.0`, token-mean aggregation, reference KL `0.001`, and no rollout IS/RS. H20 may
select only the four values admitted by a validated common profile: GPU memory
utilization, generation micro-batch, log-prob micro-batch, and actor packing
budget. The remaining runtime knobs are fixed.

G1b v1 reconstructed the historical Project-2G launcher. Human review rejected
its nonstandard token IS, token-sum aggregation, C `10`, and gradient clip
`500`; v2 uses current verl GRPO surfaces while retaining the project-backed
MATH batch, learning rate, response length, and 115-step budget. G1b v2 was
human-approved on 2026-07-28 at frozen SHA `8dafbac...e54177`.

The initialization and RLVR datasets are separate stages. R01 and R02 are
registered as AM-1.4M SFT initializations; after loading either checkpoint,
GRPO uses the same 7,500-row `hendrycks_math` file (7,405 eligible prompts) for
both arms.

Formal runs fail closed until paired-init/checkpoint receipts and the live
train, Math-7, strict grader, and calibrated H20 receipts validate against
the files actually consumed. Dataset-derived hashes are recomputed from live
parquets; formal H20 admission additionally requires an independent detached
signature over both arm terminal receipts and staged AFO resource bytes. The
same signature binds the exact NVIDIA-driver/CUDA/PyTorch/vLLM/FlashInfer
runtime projection. Both calibration worker receipts are strict JSON, and each
formal worker re-probes that projection before training.

Formal Hope manifests use `root` as the storage-security boundary and bind
three strict child roots: `dataset_root`, `model_root`, and `state_root`.
Repository checkout remains at `$ROOT/$REPO_SUBPATH`; training and Math-7 data
come from `$DATASET_ROOT/data`, initialization comes from `$MODEL_ROOT`, and
persistent run state derives from `$STATE_ROOT`. The formal adapter discards
inherited host path overrides. Local/direct entry remains backward compatible:
when the three variables are unset, `DATASET_ROOT` and `STATE_ROOT` default to
`ROOT`, while `MODEL_ROOT` keeps its existing ROOT-derived default.

## Unified experiment entry

All local and platform launches resolve wrappers relative to this directory:

```bash
bash recipe/on_policy_wdl_sft/rebuttal_rlvr/run_experiment.sh R02
```

## Direct colleague entry

When both model directories have been placed on the Meituan worker, the
colleague can launch one full 115-step cell directly without constructing G0
receipts:

```bash
ROOT=/mnt/dolphinfs/.../lgx \
  bash platform/hope_rebuttal_rlvr/run_colleague.sh R01 20260727

ROOT=/mnt/dolphinfs/.../lgx \
  bash platform/hope_rebuttal_rlvr/run_colleague.sh R02 20260727
```

Default model locations are:

```text
$ROOT/models/rebuttal_rlvr/init/R01_ORDINARY_SFT_4B_AM1P4M
$ROOT/models/rebuttal_rlvr/init/R02_WDL_SFT_4B_AM1P4M
```

Repeat with seeds `20260728` and `20260729` for the six-cell comparison. The
script verifies the directory, `config.json`, and at least one weight file,
then records `external_provenance_assumption.env` beside the run. This route
does not claim provenance matching: conclusions are conditional on the exact
two supplied checkpoints. All GRPO/data/eval/retention/release settings remain
the frozen standard-GRPO v2 values.

If the colleague has one persistent eight-H20 worker and wants all six cells
sequentially with no further interaction:

```bash
ROOT=/mnt/dolphinfs/.../lgx \
  bash platform/hope_rebuttal_rlvr/run_colleague_matrix.sh
```

This intentionally runs one eight-GPU cell at a time. Parallel multi-worker
submission continues to use the manifest/Hope entry because one worker cannot
safely host multiple eight-GPU cells.

| ID | Initialization | Status |
|---|---|---|
| `R01` | ordinary-SFT 4B, AM-1.4M | expected at `R01_ORDINARY_SFT_4B_AM1P4M`; unavailable provenance is an accepted conditional-comparison assumption |
| `R02` | published WDL-SFT 4B | defaults to the pinned `chhao/Weak-Driven-Learning` snapshot |
| `R03` | published WDL-SFT 8B | fail-closed until a real public model ID and revision are supplied |

`model_paths.env` derives the local Hugging Face snapshot from
`HF_MODEL_CACHE_ROOT`, whose default is `${ROOT}/.cache/huggingface`. Override
only the parent when moving between machines:

```bash
ROOT=/data-1 HF_MODEL_CACHE_ROOT=/data-1/.cache/huggingface \
  bash recipe/on_policy_wdl_sft/rebuttal_rlvr/run_experiment.sh R02
```

Download the pinned model through the configured large-traffic proxy in tmux:

```bash
tmux new-session -s hf-wdl-r02
bash recipe/on_policy_wdl_sft/rebuttal_rlvr/download_model.sh R02
```

`DOWNLOAD_PROXY_URL` defaults to `http://127.0.0.1:7890`; set it to an empty
string on a machine with direct Hugging Face access. The downloader uses the
host `hf` CLI when present and otherwise runs the project `verl-harness` image.
The verified local R02 download receipt is
`/data-1/model_weights/manifests/chhao-weak-driven-learning-4b-download-20260728.json`;
it records file hashes and explicitly remains `g0_status=not_admitted`.

Hope batches remain manifest-driven. The colleague-facing entry is one command:

```bash
bash platform/hope_rebuttal_rlvr/run_handoff.sh
```

It reads the checked-in contract plus one local `handoff.env`, expands all
submitter arguments, and reports only unresolved slots. G3/G4 are expected to
remain pending until this command runs inside the Meituan/Hope environment;
they are never self-certified from the physical host.

After a formal worker reaches step 115 and the training process exits zero,
`release_after_success.sh` runs automatically. It records and checks
`success_complete`, imports the local experiment-registry row, verifies the
offline W&B directory, and writes a SHA-256 file manifest for handoff. Meituan
workers have no external network, so this path requires `WANDB_MODE=offline`
and never invokes `wandb sync`. The colleague returns the complete offline W&B
run, its manifest, logs, and checkpoints to the experiment owner for joint
analysis or later sync from a networked machine. A local release failure writes
`release_status=failed` beside the attempt while preserving
`training_status=success_complete`; failed or incomplete training never enters
this path. Offline Math-7 result analysis remains a later, separate workflow
after the colleague returns logs/checkpoints.

For path/image plumbing before the authoritative weights arrive, an explicit
Base placeholder is available:

```bash
RUN_MODE=smoke ALLOW_BASE_PLACEHOLDER=1 CONFIG_ONLY=1 \
  bash recipe/on_policy_wdl_sft/rebuttal_rlvr/run_experiment.sh R02
```

This mode is labelled `placeholder_base_smoke`; `_common_math_rlvr.sh` rejects
it whenever `RUN_MODE=formal`. A placeholder result is never experiment
evidence.
