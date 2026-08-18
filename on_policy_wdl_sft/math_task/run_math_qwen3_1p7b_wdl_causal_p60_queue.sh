#!/usr/bin/env bash
set -euo pipefail
REPO_HOST=${REPO_HOST:-/data-1/code/verl}
MANIFEST=${MATH_WDL_CAUSAL_MANIFEST:-${REPO_HOST}/recipe/on_policy_wdl_sft/experiment_manifest/math_qwen3_1p7b_wdl_causal_p60.yaml}
container_manifest=${MANIFEST/${REPO_HOST}/\/workspace\/verl}
if [ "${DRY_RUN:-0}" != "1" ] && [ -z "${TMUX:-}" ]; then
    echo "ERROR: launch the causal-P60 queue inside tmux" >&2
    exit 1
fi
if [ "${DRY_RUN:-0}" != "1" ]; then
    poll_sec=${RESOURCE_POLL_SEC:-60}
    max_gpu_util_total=${MAX_GPU_UTIL_TOTAL:-20}
    max_gpu_memory_used_mib=${MAX_GPU_MEMORY_USED_MIB:-4000}
    while true; do
        gpu_util_total=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
            | awk '{sum += $1} END {print sum + 0}')
        gpu_memory_max=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
            | awk '$1 > max {max = $1} END {print max + 0}')
        if [ "$gpu_util_total" -lt "$max_gpu_util_total" ] \
            && [ "$gpu_memory_max" -lt "$max_gpu_memory_used_mib" ]; then
            break
        fi
        echo "waiting for existing GPU work: util_total=${gpu_util_total} threshold<${max_gpu_util_total}; max_memory_mib=${gpu_memory_max} threshold<${max_gpu_memory_used_mib}; sleep ${poll_sec}s"
        sleep "$poll_sec"
    done
fi
args=(--manifest "$container_manifest")
[ "${DRY_RUN:-0}" = "1" ] && args+=(--dry-run)
exec env REPO_HOST="$REPO_HOST" /data-1/verl07/run_train.sh python \
    /workspace/verl/scripts/math_wdl_causal_queue.py "${args[@]}"
