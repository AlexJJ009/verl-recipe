# Job 130 scheduler-boundary audit

This directory freezes the source-backed part of Slurm Job 130 and the one-time
GON-37 comparison with the planned A800 Pueue path. It is not a universal
launcher schema.

The common scientific boundary is exactly
`run_math_stage1_grpo.sh` at blob
`ab6471bc96297317707289daea1dfb4ab2f58615`. Both scheduler paths must reach it
once. The scheduler adapter may not assign model, data, scorer, learning rate,
step/epoch, seed, batch, context, sampling, loss, optimizer, validation, or
checkpoint-cadence variables.

The historical outer `sbatch` command, raw Slurm log, and exact external
checkpoint path are not mounted on this A800 development instance. The root
repository records that Job 130 exited normally at local P160/effective P200
with a terminal checkpoint and complete metrics/validation. The manifest keeps
that source-backed claim separate from direct artifact inspection.

The audit approves only the adapter contract and dry-run work. It does not
authorize a GPU submission. GON-35 must separately qualify `gpu8`, the A800
`verl-dev-run` runtime, image digest, P0 input hashes, P1 runtime values, and
repository-external artifact roots.

The external admission receipt must carry the exact Job 130 `model_sha256`,
`data_sha256`, and `scorer_sha256` values, plus 64-hex
`p0_config_evidence_sha256` and `p1_review_evidence_sha256` digests. Boolean
P0/P1 completion flags alone never unlock real submission.

Run the checked audit with:

```bash
python3 on_policy_wdl_sft/standard_grpo/scheduler/verify_job_130_baseline.py
python3 on_policy_wdl_sft/standard_grpo/scheduler/verify_scheduler_audit.py
```
