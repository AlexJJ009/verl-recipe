#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Internal one-arm runner. Public entries expose only DYNPERM_ENABLED and
# DYNPERM_RHO; the arm is fixed by the entry-point filename.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -ne 1 ]; then
    echo "ERROR: internal DynPerm arm runner requires one static arm id" >&2
    exit 1
fi
arm_id="$1"
case "$arm_id" in
    fixed-m1-stage1)
        arm_script="${SCRIPT_DIR}/run_math_qwen3_1p7b_wdl_fixed_m1_stage1.sh"
        ;;
    standard-c)
        arm_script="${SCRIPT_DIR}/run_math_qwen3_1p7b_wdl_causal_arm_c.sh"
        ;;
    *)
        echo "ERROR: unsupported DynPerm arm: $arm_id" >&2
        exit 1
        ;;
esac

: "${DYNPERM_ENABLED:?set DYNPERM_ENABLED=true}"
: "${DYNPERM_RHO:?set DYNPERM_RHO in [0, 1]}"
case "${DYNPERM_ENABLED,,}" in
    true|1) export DYNPERM_ENABLED=true ;;
    *) echo "ERROR: the DynPerm P60 launcher requires DYNPERM_ENABLED=true" >&2; exit 1 ;;
esac
export TOTAL_TRAINING_STEPS=60

if [ "${DRY_RUN:-0}" != "1" ] && [ -z "${TMUX:-}" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: launch a real DynPerm P60 arm inside tmux or a Slurm allocation" >&2
    exit 1
fi

# Slurm owns node admission and exclusive allocation. Local tmux launches keep
# the legacy non-mutating GPU-idle wait.
if [ "${DRY_RUN:-0}" != "1" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
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
        echo "waiting for existing GPU work: util_total=${gpu_util_total}; max_memory_mib=${gpu_memory_max}; sleep ${poll_sec}s"
        sleep "$poll_sec"
    done
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
    STAGE2_DRY_RUN=1 bash "$arm_script"
else
    bash "$arm_script"
fi
