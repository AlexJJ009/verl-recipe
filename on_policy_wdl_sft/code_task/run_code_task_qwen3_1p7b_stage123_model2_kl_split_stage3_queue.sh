#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MANIFEST=${STAGE123_MANIFEST:-${REPO_ROOT}/recipe/on_policy_wdl_sft/experiment_manifest/stage123_model2_kl_split_stage3.yaml}
STATE_ROOT=${STAGE123_EXECUTION_STATE_ROOT:-/data-2/model_weights/code_task/qwen3_1p7b_stage123_model2_kl_split_stage3_v1/state}
container_manifest=${MANIFEST/${REPO_ROOT}/\/workspace\/verl}

exec env REPO_HOST="$REPO_ROOT" /data-1/verl07/run_train.sh python \
    /workspace/verl/scripts/stage123_matrix_queue.py \
    --manifest "$container_manifest" \
    --state-root "$STATE_ROOT" \
    "$@"
