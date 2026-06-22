#!/usr/bin/env bash
# Host-side sequential code eval queue.
set -euo pipefail

if [ "${DRY_RUN:-0}" != "1" ] && [ "${ALLOW_CODE_EVAL_QUEUE:-0}" != "1" ]; then
    echo "[code eval queue] ERROR: non-dry-run eval queue requires ALLOW_CODE_EVAL_QUEUE=1" >&2
    exit 1
fi

REPO_HOST=${REPO_HOST:-/data-1/verl07/verl}
REPO_CONTAINER=${REPO_CONTAINER:-/workspace/verl}
DOCKER_IMAGE=${DOCKER_IMAGE:-verl-harness}
MODEL_CASES=${MODEL_CASES:-baseline stage1_source stage2_best stage2_final}
echo "[code eval queue] DRY_RUN=${DRY_RUN:-0} MODEL_CASES=${MODEL_CASES}"
for case_name in $MODEL_CASES; do
    echo "[code eval queue] case=${case_name} MODEL_DIR_VAR=${case_name^^}_MODEL_DIR"
done
