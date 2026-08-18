#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${DRY_RUN:-0}" != "1" ] && [ -z "${TMUX:-}" ]; then
    echo "ERROR: launch the fixed-M1 P60 queue inside tmux" >&2
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

run_arm() {
    local script="$1"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        STAGE2_DRY_RUN=1 bash "$script"
    else
        bash "$script"
    fi
}

# CS0 first gives the matched comparison with A; Stage1 then gives the matched
# comparison with C/D0. Both wrappers inherit the same P60 training contract.
run_arm "${SCRIPT_DIR}/run_math_qwen3_1p7b_wdl_fixed_m1_cold_start.sh"
run_arm "${SCRIPT_DIR}/run_math_qwen3_1p7b_wdl_fixed_m1_stage1.sh"
