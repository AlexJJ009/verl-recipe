#!/usr/bin/env bash
set -euo pipefail
REPO_HOST=${REPO_HOST:-/data-1/code/verl}
MANIFEST=${MATH_STAGE123_MANIFEST:-${REPO_HOST}/recipe/on_policy_wdl_sft/experiment_manifest/math_qwen3_1p7b_stage123_cotmask_v3.yaml}
container_manifest=${MANIFEST/${REPO_HOST}/\/workspace\/verl}
MATH7_VALIDATION_ROOT=${MATH7_VALIDATION_ROOT:-/data-1/dataset/math/qwen3_1p7b_math7_validation_v1}
if [ "${DRY_RUN:-0}" != "1" ] && [ -z "${TMUX:-}" ]; then
    echo "ERROR: launch this queue inside tmux" >&2
    exit 1
fi
if [ "${DRY_RUN:-0}" != "1" ]; then
    env REPO_HOST="$REPO_HOST" /data-1/verl07/run_train.sh python \
        /workspace/verl/recipe/on_policy_wdl_sft/math_task/prepare_math7_validation_data.py \
        --output-root "$MATH7_VALIDATION_ROOT"
fi
args=(--manifest "$container_manifest")
[ "${DRY_RUN:-0}" = "1" ] && args+=(--dry-run)
[ -n "${MATH_STAGE123_START_RUN:-}" ] && args+=(--start-run "$MATH_STAGE123_START_RUN")
exec env REPO_HOST="$REPO_HOST" /data-1/verl07/run_train.sh python \
    /workspace/verl/scripts/math_stage123_queue.py "${args[@]}"
