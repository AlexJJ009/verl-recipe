#!/usr/bin/env bash
# Host entry point for Qwen3-1.7B Code Cold Start, validated every five steps.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_HOST=${REPO_HOST:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}
REPO_CONTAINER=${REPO_CONTAINER:-/workspace/verl}
DOCKER_IMAGE=${DOCKER_IMAGE:-verl-harness:latest}
MANIFEST_HOST=${CODE_COLD_START_MANIFEST:-${REPO_HOST}/recipe/on_policy_wdl_sft/experiment_manifest/code_qwen3_1p7b_cold_start_cotmask_v3.yaml}
MANIFEST_CONTAINER=${MANIFEST_CONTAINER:-${REPO_CONTAINER}/recipe/on_policy_wdl_sft/experiment_manifest/code_qwen3_1p7b_cold_start_cotmask_v3.yaml}

args=(--manifest "$MANIFEST_CONTAINER")
if [ "${DRY_RUN:-0}" = "1" ]; then
    args+=(--dry-run)
elif [ "${ALLOW_CODE_QWEN3_1P7B_COLD_START:-0}" != "1" ]; then
    echo "[code-cold-start] ERROR: non-dry-run requires ALLOW_CODE_QWEN3_1P7B_COLD_START=1" >&2
    exit 1
elif [ -z "${TMUX:-}" ]; then
    echo "[code-cold-start] ERROR: non-dry-run must be launched from a real host tmux session" >&2
    exit 1
fi

[ -f "$MANIFEST_HOST" ] || { echo "[code-cold-start] ERROR: manifest missing: $MANIFEST_HOST" >&2; exit 1; }

docker run --rm --gpus all --ipc=host --network=host --shm-size=64g \
    -e TMUX="${TMUX:-}" \
    -v /data-1:/data-1 \
    -v /data-2:/data-2 \
    -v "$REPO_HOST":"$REPO_CONTAINER" \
    -w "$REPO_CONTAINER" \
    "$DOCKER_IMAGE" \
    python3 scripts/code_cold_start_queue.py "${args[@]}"
