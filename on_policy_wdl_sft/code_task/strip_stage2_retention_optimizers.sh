#!/usr/bin/env bash
# Strip optimizer shards from completed Stage2 retention checkpoints while
# preserving model shards for trajectory merge/eval.
set -euo pipefail

ROOT=${ROOT:-/data-1/checkpoints}
STATUS_FILE=${STATUS_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/run_code_task_stage2_retention_queue_status.tsv}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-60}
ONCE=${ONCE:-0}

strip_step_optimizers() {
    local step_dir="$1"
    [ -d "${step_dir}/actor" ] || return 0
    local model_count extra_count optim_count
    model_count=$(find "${step_dir}/actor" -maxdepth 1 -type f -name 'model_world_size_*_rank_*.pt' | wc -l)
    extra_count=$(find "${step_dir}/actor" -maxdepth 1 -type f -name 'extra_state_world_size_*_rank_*.pt' | wc -l)
    optim_count=$(find "${step_dir}/actor" -maxdepth 1 -type f -name 'optim_world_size_*_rank_*.pt' | wc -l)
    if [ "$model_count" -lt 8 ]; then
        echo "[strip-stage2] skip incomplete model step: ${step_dir} model_count=${model_count}"
        return 0
    fi
    if [ "$extra_count" -lt 8 ]; then
        echo "[strip-stage2] skip incomplete extra-state step: ${step_dir} extra_count=${extra_count}"
        return 0
    fi
    if [ "$optim_count" -gt 0 ]; then
        echo "[strip-stage2] stripping ${optim_count} optimizer shard(s): ${step_dir}"
        find "${step_dir}/actor" -maxdepth 1 -type f -name 'optim_world_size_*_rank_*.pt' -delete
        printf 'optimizer shards stripped for model-weight retention\n' > "${step_dir}/optimizer_stripped.txt"
    fi
}

status_for_prefix() {
    local prefix="$1"
    [ -f "$STATUS_FILE" ] || return 0
    awk -F '\t' -v p="$prefix" '$4 == p {status=$5} END {print status}' "$STATUS_FILE"
}

scan_once() {
    local root prefix status latest step step_num
    for root in "${ROOT}"/CODE-S2-RETENTION-BETA0-BETA0_* "${ROOT}"/CODE-S2-RETENTION-BETA01-BETA01_*; do
        [ -d "$root" ] || continue
        prefix=$(basename "$root" | sed -E 's/_[0-9]+$//')
        status=$(status_for_prefix "$prefix")
        latest=""
        if [ -f "${root}/latest_checkpointed_iteration.txt" ]; then
            latest=$(tr -dc '0-9' < "${root}/latest_checkpointed_iteration.txt")
        fi
        for step in "${root}"/global_step_*; do
            [ -d "$step" ] || continue
            step_num=$(basename "$step" | sed 's/global_step_//')
            if [ "$status" != "completed" ] && [ -n "$latest" ] && [ "$step_num" = "$latest" ]; then
                continue
            fi
            strip_step_optimizers "$step"
        done
    done
}

while true; do
    scan_once
    [ "$ONCE" = "1" ] && exit 0
    sleep "$INTERVAL_SECONDS"
done
