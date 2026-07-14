#!/usr/bin/env bash
# Thin Stage123 compatibility adapter. All lifecycle authority is Python-owned.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BATCH_MANIFEST=${EXPERIMENT_BATCH_MANIFEST:-}
STATE_ROOT=${STAGE123_EXECUTION_STATE_ROOT:-/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/state}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --batch-manifest) BATCH_MANIFEST=${2:?missing batch manifest}; shift 2 ;;
        --state-root) STATE_ROOT=${2:?missing state root}; shift 2 ;;
        --resume|--recovery-policy)
            echo "ERROR: Stage123 batch execution forbids retry/resume and recovery-policy overrides" >&2
            exit 2
            ;;
        *) echo "ERROR: unsupported Stage123 batch adapter argument: $1" >&2; exit 2 ;;
    esac
done

[ -n "$BATCH_MANIFEST" ] || {
    echo "ERROR: EXPERIMENT_BATCH_MANIFEST or --batch-manifest is required" >&2
    exit 2
}

exec python3 "${REPO_ROOT}/scripts/experiment_execution_core.py" batch-run \
    --manifest "$BATCH_MANIFEST" --state-root "$STATE_ROOT" --repo-root "$REPO_ROOT"
