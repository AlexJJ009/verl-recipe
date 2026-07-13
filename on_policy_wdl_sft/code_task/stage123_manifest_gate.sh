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
    : "${STAGE123_PREFLIGHT_RESULT:?STAGE123_PREFLIGHT_RESULT required}"
    : "${STAGE123_CALIBRATION_RESULT:?STAGE123_CALIBRATION_RESULT required}"
    : "${STAGE123_ACCEPTANCE_REPORT:?STAGE123_ACCEPTANCE_REPORT required}"
    stage123_require_result "$STAGE123_PREFLIGHT_RESULT" preflight_result
    stage123_require_result "$STAGE123_CALIBRATION_RESULT" calibration_result
    stage123_require_result "$STAGE123_ACCEPTANCE_REPORT" acceptance_report
    printf 'formal admission result bindings valid for %s\n' "$run_id"
}
