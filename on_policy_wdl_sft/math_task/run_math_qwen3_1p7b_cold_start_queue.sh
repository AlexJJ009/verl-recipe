#!/usr/bin/env bash
set -euo pipefail

REPO_HOST=${REPO_HOST:-/data-1/code/verl}
MANIFEST=${MATH_COLD_START_MANIFEST:-${REPO_HOST}/recipe/on_policy_wdl_sft/experiment_manifest/math_qwen3_1p7b_cold_start_cotmask_v3.yaml}
container_manifest=${MANIFEST/${REPO_HOST}/\/workspace\/verl}

if [ "${DRY_RUN:-0}" != "1" ] && [ -z "${TMUX:-}" ]; then
    echo "ERROR: launch this queue inside tmux" >&2
    exit 1
fi

args=(--manifest "$container_manifest")
[ "${DRY_RUN:-0}" = "1" ] && args+=(--dry-run)

exec env REPO_HOST="$REPO_HOST" /data-1/verl07/run_train.sh python \
    /workspace/verl/scripts/math_cold_start_queue.py "${args[@]}"
