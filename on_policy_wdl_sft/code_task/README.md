# Code Task On-Policy WDL-SFT Scripts

Status: code-task data, reward, Stage1 launch, queue, monitor, and Meituan/AFO
support are implemented. Do not launch non-dry-run training unless the user
explicitly approves that queue.

Code prompts do not use boxed answers. The contract is:

```text
<think>...</think>
<answer>
```python
...
```
</answer>
```

## G1-build-smoke

Run inside `verl-harness`:

```bash
export PROJECT_CACHE_ROOT=/data-1/.cache
export HF_HOME=/data-1/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME
export XDG_CACHE_HOME=/data-1/.cache
export CODE_OFFICIAL_SOURCE_ROOT=/data-1/dataset/code/official_sources
export BIGCODEBENCH_OVERRIDE_PATH=$CODE_OFFICIAL_SOURCE_ROOT/bigcodebench/BigCodeBench-v0.1.4.jsonl
export HTTP_PROXY=${HTTP_PROXY:-http://127.0.0.1:7890}
export HTTPS_PROXY=${HTTPS_PROXY:-http://127.0.0.1:7890}
bash recipe/on_policy_wdl_sft/code_task/install_code_eval_deps.sh
python3 recipe/on_policy_wdl_sft/code_task/verify_code_eval_deps.py
python3 recipe/on_policy_wdl_sft/code_task/prepare_project_official_cache.py
python3 recipe/on_policy_wdl_sft/code_task/prepare_code_rl_dataset.py
python3 recipe/on_policy_wdl_sft/code_task/prepare_kodcode_light_rl_dataset.py --download
PYTHONPATH=/data-1/code_eval_envs/official_site:/data-1/code_eval_envs/LiveCodeBench:$PYTHONPATH \
  python3 recipe/on_policy_wdl_sft/code_task/prepare_official_only_validation.py
python3 recipe/on_policy_wdl_sft/code_task/create_code_stage2_nonoverlap_shard.py
python3 recipe/on_policy_wdl_sft/code_task/verify_code_dataset.py --verify-only
python3 recipe/on_policy_wdl_sft/code_task/verify_code_reward_env.py
cd /data-1/code_eval_envs/LiveCodeBench
PYTHONPATH=/root/buaa/local_data1/verl07/verl:/data-1/code_eval_envs/official_site:/data-1/code_eval_envs/LiveCodeBench \
  python3 /root/buaa/local_data1/verl07/verl/recipe/on_policy_wdl_sft/code_task/verify_official_only_reward.py
cd /root/buaa/local_data1/verl07/verl
python3 recipe/on_policy_wdl_sft/code_task/verify_code_reward_metadata_dump.py
PYTHONPATH=/data-1/code_eval_envs/official_site:/data-1/code_eval_envs/LiveCodeBench:$PYTHONPATH \
  python3 recipe/on_policy_wdl_sft/code_task/eval_code_official.py --benchmark humaneval --samples /path/to/humaneval_samples.jsonl --output-dir /data-1/eval_outputs/code_task/humaneval --summary /data-1/eval_outputs/code_task/humaneval_official_summary.json
PYTHONPATH=/data-1/code_eval_envs/official_site:/data-1/code_eval_envs/LiveCodeBench:$PYTHONPATH \
  python3 recipe/on_policy_wdl_sft/code_task/eval_code_official.py --benchmark bigcodebench --samples /path/to/bigcodebench_samples.jsonl --output-dir /data-1/eval_outputs/code_task/bigcodebench --summary /data-1/eval_outputs/code_task/bigcodebench_official_summary.json
python3 recipe/on_policy_wdl_sft/code_task/eval_code_official.py --benchmark livecodebench --custom-output /path/to/livecodebench_custom_output.json --output-dir /data-1/eval_outputs/code_task/livecodebench --summary /data-1/eval_outputs/code_task/livecodebench_official_summary.json
DRY_RUN=1 bash recipe/on_policy_wdl_sft/code_task/run_s1_code_smoke_beta_0.sh
DRY_RUN=1 ALLOW_EXTERNAL_MODEL2_FOR_DRY_RUN=1 MODEL2_PATH=/data-1/.cache/Qwen3-4B-Base-SFT-stage-1 STAGE2_HANDOFF_STEP=5 bash recipe/on_policy_wdl_sft/code_task/run_s2_code_smoke_beta0_beta0.sh
for exp in s1-code-smoke-beta-0 s1-code-pilot-beta-0 s2-code-smoke-beta0-beta0 s2-code-pilot-beta0-beta0; do
  DRY_RUN=1 EXPERIMENT=$exp ALLOW_EXTERNAL_MODEL2_FOR_DRY_RUN=1 MODEL2_PATH=/data-1/.cache/Qwen3-4B-Base-SFT-stage-1 STAGE2_HANDOFF_STEP=5 bash platform/hope_code_task/jupyter.sh
done
DRY_RUN=1 bash recipe/on_policy_wdl_sft/code_task/run_code_task_smoke_queue.sh
DRY_RUN=1 bash recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh
```

`verify_code_eval_deps.py` records EvalPlus, BigCodeBench, and LiveCodeBench
harness status. Local `test_plus.jsonl` files are never official EvalPlus+
scores.

Official benchmark data must be resolved from project-owned paths:

- HF cache: `/data-1/.cache/huggingface`
- EvalPlus cache via `XDG_CACHE_HOME`: `/data-1/.cache/evalplus`
- official raw source root:
  `/data-1/dataset/code/official_sources`
- BigCodeBench official JSONL:
  `/data-1/dataset/code/official_sources/bigcodebench/BigCodeBench-v0.1.4.jsonl`
- LiveCodeBench `release_v1` HF dataset cache:
  `/data-1/.cache/huggingface/datasets/livecodebench___code_generation_lite/.../code_generation_lite-test.arrow`
- Manifest:
  `/data-1/dataset/code/official_sources/official_cache_manifest.json`

Do not use `/root/.cache` as a runtime dependency. It is allowed only as a
one-time migration source for `prepare_project_official_cache.py`. Official
offline eval and validation preparation export `HF_HUB_OFFLINE=1` and
`HF_DATASETS_OFFLINE=1`; missing project cache should fail instead of fetching
from the network or another user's cache.

`official_aligned_reward.py` is the default code-task reward for new runs.
HumanEval+/MBPP+ use EvalPlus official `check_correctness`; BigCodeBench uses
the official `bigcodebench` evaluator; LiveCodeBench uses the official
`lcb_runner` evaluator. Official datasets have no local-runner fallback: missing
official packages are hard failures. KodCode-Light-RL-10K rows use the
`kodcode_exec` pytest-style runner under `firejail`, following KodCode/code-r1's
official sandbox requirement. Missing `firejail` is a hard dependency error for
formal training; `KODCODE_ALLOW_UNSANDBOXED=1` is diagnostic-only. Other explicit
non-official smoke/train rows use `local_exec`.

## KodCode-Light-RL-10K

`prepare_kodcode_light_rl_dataset.py` downloads and converts
`KodCode/KodCode-Light-RL-10K` into the current code-task RL format:

```bash
python3 recipe/on_policy_wdl_sft/code_task/prepare_kodcode_light_rl_dataset.py --download
```

Outputs:

- Raw HF parquet: `/data-1/dataset/KodCode-Light-RL-10K/data/train-00000-of-00001.parquet`
- Training parquet: `/data-1/dataset/code/verl_rl/kodcode_light_rl_10k_train_rl_format.parquet`
- Audit report: `/data-1/dataset/KodCode-Light-RL-10K/reports/kodcode_light_rl_10k_audit.json`
- Validation probe: `/data-1/dataset/KodCode-Light-RL-10K/reports/kodcode_light_rl_10k_validation.json`

## User-approved G2-training-smoke

After G1 passes, the next training-smoke queue command is:

```bash
tmux new-session -d -s code_task_smoke_queue \
  "cd /data-1/verl07/verl && ALLOW_G2_TRAINING_SMOKE=1 bash recipe/on_policy_wdl_sft/code_task/run_code_task_smoke_queue.sh"
```

Run the monitor separately:

```bash
tmux new-session -d -s code_task_smoke_monitor \
  "cd /data-1/verl07/verl && bash recipe/on_policy_wdl_sft/code_task/monitor_code_task_queue_notify.sh"
```

Queue and monitor notifications use WxPusher by default for non-dry-run tmux
jobs. Set `WXPUSHER_NOTIFY=0` for dry-run verification or interactive debugging.
Operational decisions after failures belong to
`docs/joint_training/guides/code_task_monitor_agent_runbook.md`, not to the
shell monitor itself. Formal queue launch/repair history is recorded in
`docs/joint_training/guides/code_task_training_queue_runlog.md`.

## Formal KodCode Stage1

Formal Stage1 now uses KodCode-Light-RL-10K by default:

```bash
DRY_RUN=1 bash recipe/on_policy_wdl_sft/code_task/run_s1_code_onpolicy_sft_beta_0.sh
DRY_RUN=1 bash recipe/on_policy_wdl_sft/code_task/run_s1_code_onpolicy_sft_beta_01.sh
DRY_RUN=1 QUEUE_DRY_RUN_VALIDATE_WRAPPERS=1 bash recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh
QUEUE_MODE=full WXPUSHER_NOTIFY=0 timeout 2 bash recipe/on_policy_wdl_sft/code_task/monitor_code_task_queue_notify.sh || true
```

The train file is
`/data-1/dataset/code/verl_rl/kodcode_light_rl_10k_train_rl_format.parquet`.
Online validation now defaults to the full EvalPlus pair only:

- HumanEval+:
  `/data-1/dataset/code/verl_rl/online_full_humaneval_plus/official_humaneval_plus_val.parquet`
- MBPP+:
  `/data-1/dataset/code/verl_rl/online_full_mbpp_plus/official_mbpp_plus_val.parquet`

The Stage1 plateau-finding curve uses `VAL_N=1`, `VAL_TEMPERATURE=0.2`,
`VAL_TOP_P=0.95`, and reports core code accuracy as `pass@1`. BigCodeBench and
LiveCodeBench are intentionally excluded from every online validation step; run
them on candidate checkpoints and final reports through the offline official
eval queue. The earlier four-file official validation set under
`/data-1/dataset/code/verl_rl/official_*_val.parquet` was too small (`2/2/1/1`
prompts) and must not be used to pick best checkpoints or compare training
effects.

The formal queue runs two independent Stage1 experiments sequentially:

- `beta=0.0`: `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V2`
- `beta=0.1`: `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V2`

Both use the same train data, validation data, seed, batch geometry, response
length, save/test cadence, and latest+best checkpoint retention. The training
data seed is fixed by default with `DATA_SEED=20260604` and `DATA_SHUFFLE=True`.
The retained best checkpoint is HE+-primary (`val-core/HumanEval+/acc/pass@1`);
candidate selection must also report MBPP+-best and latest checkpoints before
running offline BigCodeBench/LiveCodeBench.

Formal queue launch after explicit approval:

```bash
tmux new-session -d -s code_task_full_queue \
  "cd /data-1/verl07/verl && MIN_FREE_GB=100 QUEUE_CONTINUE_ON_FAILURE=0 ALLOW_CODE_FULL_TRAINING=1 bash recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh"
tmux new-session -d -s code_task_full_monitor \
  "cd /data-1/verl07/verl && QUEUE_MODE=full POLL_SEC=1800 bash recipe/on_policy_wdl_sft/code_task/monitor_code_task_queue_notify.sh"
```

`run_code_task_full_queue.sh` performs a pre-launch checkpoint disk check,
blocks checkpoint collisions unless `ALLOW_RESUME=1`, and sends queue/run
start, completion, failure, skip, and queue-complete notifications. Formal
queues default to `QUEUE_CONTINUE_ON_FAILURE=0`: a failed item blocks the queue
so the Monitor Agent can diagnose, apply a recorded repair, and resume/relaunch
the same item before later experiments run. If disk is below `MIN_FREE_GB`, the
current independent item is skipped only by the pre-launch disk gate and the
Monitor Agent should follow the runbook before any cleanup or relaunch.

### Stage1 retention rerun

The original V2 formal queue kept latest+best only, so early plateau handoff
checkpoints are unavailable for Stage2. The retention rerun keeps the Stage1
comparison boundary clean by still training each beta to step 150 with the same
train file, `DATA_SEED=20260604`, `DATA_SHUFFLE=True`, batch geometry, online
validation settings, and reward function. It additionally protects the handoff
candidate checkpoints `70,80,90,100,110,120`.

Protected handoff checkpoints default to model weights only
(`PROTECTED_CKPT_STRIP_OPTIMIZER=True`) to control disk use. The latest step 150
checkpoint remains a normal latest checkpoint for the same-budget Stage1
baseline. The HE+-best checkpoint is still retained by the normal best rule.

### DeepCoder Stage1 retention

DeepCoder-Preview is handled as a separate data-domain Stage1 batch. The design
document is `docs/joint_training/reports/deepcoder_preview_code_task_transfer_design.md`;
the executable Stage1-only `/goal` plan is
`docs/joint_training/plans/active/deepcoder_stage1_training_execution_plan.md`.
The formal train file is prefiltered in `verl-harness` with the Qwen3 chat
template to `prompt_tokens <= 1024`:
`/data-1/dataset/code/verl_rl/deepcoder_preview_train_prompt1024_rl_format.parquet`.
The manifest is
`/data-1/dataset/code/verl_rl/deepcoder_preview_train_prompt1024_manifest.json`
and records `22,063` kept rows from `23,287`.

Dry-run:

```bash
DRY_RUN=1 bash recipe/on_policy_wdl_sft/code_task/run_s1_code_deepcoder_beta_0_retention.sh
DRY_RUN=1 bash recipe/on_policy_wdl_sft/code_task/run_s1_code_deepcoder_beta_01_retention.sh
DRY_RUN=1 QUEUE_DRY_RUN_VALIDATE_WRAPPERS=1 bash recipe/on_policy_wdl_sft/code_task/run_code_task_deepcoder_stage1_queue.sh
```

Formal launch after explicit approval:

```bash
tmux new-session -d -s code_task_deepcoder_stage1_queue \
  "cd /data-1/verl07/verl && MIN_FREE_GB=600 QUEUE_CONTINUE_ON_FAILURE=0 ALLOW_DEEPCODER_STAGE1_TRAINING=1 bash recipe/on_policy_wdl_sft/code_task/run_code_task_deepcoder_stage1_queue.sh"
tmux new-session -d -s code_task_deepcoder_stage1_monitor \
  "cd /data-1/verl07/verl && QUEUE_MODE=deepcoder_stage1 POLL_SEC=1800 bash recipe/on_policy_wdl_sft/code_task/monitor_code_task_queue_notify.sh"
```

This queue runs only Stage1 beta `0.0` and beta `0.1`; Stage2 handoff selection
and baseline launch are explicitly out of scope.

Dry-run:

```bash
DRY_RUN=1 bash recipe/on_policy_wdl_sft/code_task/run_s1_code_onpolicy_sft_beta_0_retention.sh
DRY_RUN=1 bash recipe/on_policy_wdl_sft/code_task/run_s1_code_onpolicy_sft_beta_01_retention.sh
DRY_RUN=1 QUEUE_DRY_RUN_VALIDATE_WRAPPERS=1 bash recipe/on_policy_wdl_sft/code_task/run_code_task_retention_queue.sh
```

Formal launch after explicit approval:

```bash
tmux new-session -d -s code_task_retention_queue \
  "cd /data-1/verl07/verl && MIN_FREE_GB=400 QUEUE_CONTINUE_ON_FAILURE=0 ALLOW_CODE_RETENTION_TRAINING=1 bash recipe/on_policy_wdl_sft/code_task/run_code_task_retention_queue.sh"
tmux new-session -d -s code_task_retention_monitor \
  "cd /data-1/verl07/verl && QUEUE_MODE=retention POLL_SEC=1800 bash recipe/on_policy_wdl_sft/code_task/monitor_code_task_queue_notify.sh"
```

Prompt alignment boundary: KodCode train rows, the online official validation
parquet, and the offline vLLM generation script are generated through the same
`build_prompt` contract. Offline official scorers consume generated samples or
custom output files after `convert_official_outputs.py` applies the shared code
extractor.

Full official offline eval for Stage1 candidates:

```bash
tmux new-session -d -s code_offline_beta0_heplus \
  "cd /root/buaa/local_data1/verl07/verl && ALLOW_CODE_OFFLINE_EVAL=1 START_INDEX=0 END_INDEX=0 BENCHMARK=humaneval N_SAMPLES=3 bash recipe/on_policy_wdl_sft/code_task/run_code_offline_eval_queue.sh"
tmux new-session -d -s code_offline_beta01_heplus \
  "cd /root/buaa/local_data1/verl07/verl && ALLOW_CODE_OFFLINE_EVAL=1 START_INDEX=1 END_INDEX=1 BENCHMARK=humaneval N_SAMPLES=3 bash recipe/on_policy_wdl_sft/code_task/run_code_offline_eval_queue.sh"
```

Run HumanEval+, MBPP+, BigCodeBench, and LiveCodeBench on selected candidate
checkpoints: HE+-best, MBPP+-best if different, and latest/final. Results are
written under `/data-1/eval_outputs/code_task/full_official/<label>/<benchmark>/`.
Merged HF weights are written under
`/data-1/model_weights/code_task/offline_eval/<label>/actor_step150/`; delete
only those merged eval copies when disk is tight, not the source checkpoints.

V2 latest unified-N3 diagnostic offline eval for Stage1 step 150:

```bash
bash recipe/on_policy_wdl_sft/code_task/check_code_offline_eval_v2_latest_n3_readiness.sh
DRY_RUN=1 bash recipe/on_policy_wdl_sft/code_task/run_code_offline_eval_v2_latest_n3_queue.sh
tmux new-session -d -s code_v2_latest_n3_eval \
  "cd /root/buaa/local_data1/verl07/verl && ALLOW_CODE_V2_LATEST_OFFLINE_EVAL=1 bash recipe/on_policy_wdl_sft/code_task/run_code_offline_eval_v2_latest_n3_queue.sh"
```

This queue evaluates the V2 beta `0.0` and beta `0.1` latest step-150
checkpoints on HumanEval+, MBPP+, BigCodeBench, and LiveCodeBench with one
shared diagnostic setting: `N_SAMPLES=3`, `TEMPERATURE=1.0`, `TOP_P=0.95`,
`MAX_TOKENS=4096`, `SEED=42`, and `ENABLE_THINKING=true`. Results are written
under `/data-1/eval_outputs/code_task/v2_latest_unified_n3/`; merged HF weights
are written under `/data-1/model_weights/code_task/offline_eval/v2_*_latest_step150/`.
The queue writes normalized `mean@3` and `pass@3` summaries to
`summary_v2_latest_unified_n3.json` and `summary_v2_latest_unified_n3.md`.

## Meituan/AFO

Dry-run dispatch examples:

```bash
DRY_RUN=1 EXPERIMENT=s1-code-smoke-beta-0 bash platform/hope_code_task/jupyter.sh
DRY_RUN=1 EXPERIMENT=s2-code-smoke-beta0-beta0 ALLOW_EXTERNAL_MODEL2_FOR_DRY_RUN=1 MODEL2_PATH=/path/to/model bash platform/hope_code_task/jupyter.sh
```

Pilot runs require `SANDBOX_FUSION_URL` and explicit approval.
