#!/usr/bin/env bash
set -euo pipefail
REPO_HOST=${REPO_HOST:-/data-1/code/verl}
MANIFEST=${CODE_STAGE123_MANIFEST:-${REPO_HOST}/recipe/on_policy_wdl_sft/experiment_manifest/code_qwen3_1p7b_stage123_cotmask_v3.yaml}
ADMISSION=${CODE_STAGE123_ADMISSION:-/data-2/model_weights/code_task/qwen3_1p7b_stage123_cotmask_v3_author_signature_v2_step20/admission.json}
if [ "${DRY_RUN:-0}" != 1 ]; then
    python3 - "$MANIFEST" "$ADMISSION" <<'PY'
import sys,yaml
manifest=yaml.safe_load(open(sys.argv[1],encoding="utf-8"))
if manifest.get("launch_allowed") is not True or manifest.get("status") != "launch_ready_gpu_probe_passed":
    raise SystemExit("Code Stage123 launch is blocked until the real 8K GPU probe passes and the frozen manifest is admitted")
if manifest.get("paths",{}).get("admission") != sys.argv[2]:
    raise SystemExit("Code Stage123 manifest/admission path mismatch")
PY
fi
container_manifest=${MANIFEST/${REPO_HOST}/\/workspace\/verl}
if [ "${DRY_RUN:-0}" != 1 ] && [ -z "${TMUX:-}" ]; then
    echo "ERROR: launch the Code Stage123 queue inside tmux" >&2
    exit 1
fi
args=(--manifest "$container_manifest")
[ "${DRY_RUN:-0}" = 1 ] && args+=(--dry-run)
[ -n "${CODE_STAGE123_START_RUN:-}" ] && args+=(--start-run "$CODE_STAGE123_START_RUN")
EVENT_LOG=$(python3 - "$MANIFEST" <<'PY'
import sys,yaml
print(yaml.safe_load(open(sys.argv[1],encoding="utf-8"))["paths"]["event_log"])
PY
)
if [ "${DRY_RUN:-0}" != 1 ]; then
    manifest_sha=$(sha256sum "$MANIFEST" | awk '{print $1}')
    model1_selection=$(python3 - "$MANIFEST" <<'PY'
import sys,yaml
print(yaml.safe_load(open(sys.argv[1],encoding="utf-8"))["paths"]["model1_selection"])
PY
)
    dataset_receipt=$(python3 - "$MANIFEST" <<'PY'
import sys,yaml
print(yaml.safe_load(open(sys.argv[1],encoding="utf-8"))["paths"]["dataset_receipt"])
PY
)
    model1_selection_sha=$(sha256sum "$model1_selection" | awk '{print $1}')
    dataset_receipt_sha=$(sha256sum "$dataset_receipt" | awk '{print $1}')
else
    manifest_sha=dry-run
    model1_selection_sha=dry-run
    dataset_receipt_sha=dry-run
fi
exec env REPO_HOST="$REPO_HOST" STAGE123_EVENT_LOG="$EVENT_LOG" \
    CODE_STAGE123_ADMISSION="$ADMISSION" \
    CODE_STAGE123_MANIFEST_SHA256="$manifest_sha" \
    CODE_STAGE123_MODEL1_SELECTION_SHA256="$model1_selection_sha" \
    CODE_STAGE123_DATASET_RECEIPT_SHA256="$dataset_receipt_sha" \
    /data-1/verl07/run_train.sh python /workspace/verl/scripts/math_stage123_queue.py "${args[@]}"
