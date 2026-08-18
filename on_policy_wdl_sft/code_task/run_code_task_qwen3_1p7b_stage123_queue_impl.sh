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
    if [ "${DRY_RUN:-0}" = 1 ]; then
        exec python3 "${REPO_ROOT}/scripts/stage123_dry_run_compat.py" \
            --manifest "${STAGE123_MANIFEST:-${REPO_ROOT}/recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml}" \
            --manifest-tool "${STAGE123_MANIFEST_TOOL:-${REPO_ROOT}/scripts/experiment_manifest.py}" \
            --python "${STAGE123_MANIFEST_PYTHON:-python3}" \
            --scratch-root "${STAGE123_SCRATCH_ROOT:-/data-1/tmp/verl_agent_scratch/qwen3_1p7b_stage123}"
    fi
    echo "ERROR: EXPERIMENT_BATCH_MANIFEST or --batch-manifest is required" >&2
    exit 2
}

if [ "${DRY_RUN:-0}" != "1" ]; then
python3 - "$BATCH_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
try:
    payload = json.loads(text)
except json.JSONDecodeError:
    import yaml
    payload = yaml.safe_load(text)
if payload.get("launch_allowed") is not True:
    raise SystemExit("ERROR: code Stage123 batch manifest is not launchable; retrain CoT-v3 Cold Start/Stage1 and regenerate formal admission")
PY
fi

exec python3 "${REPO_ROOT}/scripts/experiment_execution_core.py" batch-run \
    --manifest "$BATCH_MANIFEST" --state-root "$STATE_ROOT" --repo-root "$REPO_ROOT"
