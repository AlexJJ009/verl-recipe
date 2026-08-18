#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/data-1/code/verl}
MANIFEST=${MANIFEST:-${REPO_ROOT}/recipe/on_policy_wdl_sft/experiment_manifest/code_qwen3_1p7b_stage123_cotmask_v3.yaml}
SCRATCH_ROOT=${SCRATCH_ROOT:-/data-1/tmp/verl_agent_scratch/code_stage123_gpu_utilization_probe}
CANDIDATES=${CANDIDATES:-0.35,0.40,0.45,0.50,0.55}
VALIDATION_TIMEOUT_SECONDS=${VALIDATION_TIMEOUT_SECONDS:-10800}
THROUGHPUT_TIMEOUT_SECONDS=${THROUGHPUT_TIMEOUT_SECONDS:-7200}
MINIMUM_HEADROOM_MIB=${MINIMUM_HEADROOM_MIB:-512}
IDLE_MAX_UTILIZATION=${IDLE_MAX_UTILIZATION:-5}
IDLE_MAX_MEMORY_USED_MIB=${IDLE_MAX_MEMORY_USED_MIB:-1024}

command=(
  python3 "${REPO_ROOT}/scripts/run_code_stage123_gpu_utilization_probe.py"
  --manifest "$MANIFEST"
  --scratch-root "$SCRATCH_ROOT"
  --candidates "$CANDIDATES"
  --validation-timeout-seconds "$VALIDATION_TIMEOUT_SECONDS"
  --throughput-timeout-seconds "$THROUGHPUT_TIMEOUT_SECONDS"
  --minimum-headroom-mib "$MINIMUM_HEADROOM_MIB"
  --idle-max-utilization "$IDLE_MAX_UTILIZATION"
  --idle-max-memory-used-mib "$IDLE_MAX_MEMORY_USED_MIB"
)

if [ "${DRY_RUN:-0}" = 1 ]; then
  command+=(--dry-run)
fi

cd "$REPO_ROOT"
exec "${command[@]}"
