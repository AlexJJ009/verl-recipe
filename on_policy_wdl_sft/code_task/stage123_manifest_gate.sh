#!/usr/bin/env bash
set -euo pipefail
STAGE123_GATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE123_GATE_REPO_ROOT="$(cd "${STAGE123_GATE_DIR}/../../.." && pwd)"

stage123_require_result() {
    local path=${1:?result path required}
    local expected=${2:?result type required}
    python3 - "$STAGE123_GATE_REPO_ROOT" "$path" "$expected" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1])/'scripts'))
from execution_results import load_and_validate
result=load_and_validate(Path(sys.argv[2]),sys.argv[3])
print(json.dumps(result.as_dict(),sort_keys=True))
raise SystemExit(0 if result.authorized else 1)
PY
}

stage123_require_formal_admission() {
    local run_id=${1:?run id required}
    [ "${DRY_RUN:-0}" = 1 ] && return 0
    if [ -n "${STAGE123_MATRIX_ADMISSION:-}" ]; then
        python3 "$STAGE123_GATE_REPO_ROOT/scripts/stage123_matrix_admission.py" validate \
            --admission "$STAGE123_MATRIX_ADMISSION" \
            --run-id "$run_id" \
            --repo-root "$STAGE123_GATE_REPO_ROOT" >/dev/null
        stage123_check_machine
        printf 'immutable matrix admission valid for %s\n' "$run_id"
        return 0
    fi
    if [ -n "${STAGE123_BATCH_ADMISSION_RECORD:-}" ]; then
        : "${STAGE123_ADMISSION_BUNDLE:?STAGE123_ADMISSION_BUNDLE required}"
        : "${STAGE123_BATCH_ADMISSION_RECORD_SHA256:?STAGE123_BATCH_ADMISSION_RECORD_SHA256 required}"
        : "${STAGE123_BATCH_ID:?STAGE123_BATCH_ID required}"
        : "${STAGE123_BATCH_MANIFEST_SHA256:?STAGE123_BATCH_MANIFEST_SHA256 required}"
        : "${STAGE123_BATCH_ITEM_ID:?STAGE123_BATCH_ITEM_ID required}"
        : "${STAGE123_BATCH_ADMISSION_BUNDLE_SHA256:?STAGE123_BATCH_ADMISSION_BUNDLE_SHA256 required}"
        : "${STAGE123_BATCH_COMMAND_SHA256:?STAGE123_BATCH_COMMAND_SHA256 required}"
        python3 "$STAGE123_GATE_REPO_ROOT/scripts/execution_results.py" admission validate-phase \
            --bundle "$STAGE123_ADMISSION_BUNDLE" \
            --record "$STAGE123_BATCH_ADMISSION_RECORD" \
            --record-sha256 "$STAGE123_BATCH_ADMISSION_RECORD_SHA256" \
            --run-id "$run_id" \
            --batch-id "$STAGE123_BATCH_ID" \
            --batch-manifest-sha256 "$STAGE123_BATCH_MANIFEST_SHA256" \
            --item-id "$STAGE123_BATCH_ITEM_ID" \
            --admission-bundle-sha256 "$STAGE123_BATCH_ADMISSION_BUNDLE_SHA256" \
            --command-sha256 "$STAGE123_BATCH_COMMAND_SHA256" \
            --repo-root "$STAGE123_GATE_REPO_ROOT" >/dev/null
        stage123_check_machine
        printf 'active batch item admission valid for %s\n' "$run_id"
        return 0
    fi
    if [ -n "${STAGE123_TREATMENT_REUSE_ADMISSION:-}" ]; then
        python3 "$STAGE123_GATE_REPO_ROOT/scripts/stage123_control_reuse.py" validate-treatment \
            --admission "$STAGE123_TREATMENT_REUSE_ADMISSION" --run-id "$run_id" >/dev/null
        printf 'authorized treatment-only admission valid for %s\n' "$run_id"
        return 0
    fi
    : "${STAGE123_ADMISSION_BUNDLE:?STAGE123_ADMISSION_BUNDLE required}"
    python3 "$STAGE123_GATE_REPO_ROOT/scripts/execution_results.py" admission validate \
        --bundle "$STAGE123_ADMISSION_BUNDLE" --require-accepted --repo-root "$STAGE123_GATE_REPO_ROOT" >/dev/null
    printf 'immutable admission bundle valid for %s\n' "$run_id"
}
