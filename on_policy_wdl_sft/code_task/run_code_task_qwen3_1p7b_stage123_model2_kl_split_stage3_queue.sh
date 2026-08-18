#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MANIFEST=${STAGE123_MANIFEST:-${REPO_ROOT}/recipe/on_policy_wdl_sft/experiment_manifest/stage123_model2_kl_split_stage3.yaml}
STATE_ROOT=${STAGE123_EXECUTION_STATE_ROOT:-/data-2/model_weights/code_task/qwen3_1p7b_stage123_model2_kl_split_stage3_v2/state}
export STAGE123_MATRIX_ADMISSION=${STAGE123_MATRIX_ADMISSION:-/data-2/model_weights/code_task/qwen3_1p7b_stage123_model2_kl_split_stage3_v2/admission.json}
container_manifest=${MANIFEST/${REPO_ROOT}/\/workspace\/verl}

if [ "${DRY_RUN:-0}" != "1" ]; then
python3 - "$MANIFEST" <<'PY'
import sys
from pathlib import Path
import yaml

manifest = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("launch_allowed") is not True:
    raise SystemExit("ERROR: code split Stage123 manifest is not launchable; retrain CoT-v3 Cold Start/Stage1 and regenerate formal admission")
PY
fi

exec env REPO_HOST="$REPO_ROOT" /data-1/verl07/run_train.sh python \
    /workspace/verl/scripts/stage123_matrix_queue.py \
    --manifest "$container_manifest" \
    --state-root "$STATE_ROOT" \
    "$@"
