#!/usr/bin/env bash
# Shared formal-launch receipt gate for Stage123 queue and direct phase wrappers.
set -euo pipefail
STAGE123_GATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE123_GATE_REPO_ROOT="$(cd "${STAGE123_GATE_DIR}/../../.." && pwd)"

stage123_require_preflight_receipt() {
    local run_id=${1:?run id required}
    [ "${DRY_RUN:-0}" = 1 ] && return 0
    : "${STAGE123_MANIFEST:?STAGE123_MANIFEST required}"
    : "${STAGE123_NORMALIZED_MANIFEST:?STAGE123_NORMALIZED_MANIFEST required}"
    : "${STAGE123_PREFLIGHT_REPORT:?STAGE123_PREFLIGHT_REPORT required}"
    : "${STAGE123_PREFLIGHT_RECEIPT:?STAGE123_PREFLIGHT_RECEIPT required}"
    : "${STAGE123_PREFLIGHT_POLICY:?STAGE123_PREFLIGHT_POLICY required}"
    : "${STAGE123_EXPECTED_PROFILE_HASH:?STAGE123_EXPECTED_PROFILE_HASH required}"
    : "${STAGE123_RECEIPT_MAX_AGE_SECONDS:?STAGE123_RECEIPT_MAX_AGE_SECONDS required}"
    python3 "${STAGE123_GATE_REPO_ROOT}/scripts/stage123_preflight_receipt.py" verify \
        --receipt "$STAGE123_PREFLIGHT_RECEIPT" \
        --normalized-manifest "$STAGE123_NORMALIZED_MANIFEST" \
        --report "$STAGE123_PREFLIGHT_REPORT" \
        --policy "$STAGE123_PREFLIGHT_POLICY" \
        --run-id "$run_id" \
        --profile-hash "$STAGE123_EXPECTED_PROFILE_HASH" \
        --max-age-seconds "$STAGE123_RECEIPT_MAX_AGE_SECONDS"
}

stage123_require_deployability_receipt() {
    [ "${DRY_RUN:-0}" = 1 ] && return 0
    : "${STAGE123_NORMALIZED_MANIFEST:?STAGE123_NORMALIZED_MANIFEST required}"
    : "${STAGE123_PREFLIGHT_RECEIPT:?STAGE123_PREFLIGHT_RECEIPT required}"
    : "${STAGE123_EXPECTED_PROFILE_HASH:?STAGE123_EXPECTED_PROFILE_HASH required}"
    : "${STAGE123_DEPLOYABILITY_RECEIPT:?STAGE123_DEPLOYABILITY_RECEIPT required}"
    : "${STAGE123_FORMAL_QUEUE_ID:?STAGE123_FORMAL_QUEUE_ID required}"
    : "${STAGE123_CALIBRATION_REPORT:?STAGE123_CALIBRATION_REPORT required}"
    : "${STAGE123_CALIBRATION_POLICY:?STAGE123_CALIBRATION_POLICY required}"
    : "${STAGE123_CALIBRATION_HISTORY_INDEX:?STAGE123_CALIBRATION_HISTORY_INDEX required}"
    : "${STAGE123_CALIBRATION_PREDICTION_CONTRACT:?STAGE123_CALIBRATION_PREDICTION_CONTRACT required}"
    local args=(
        --receipt "$STAGE123_DEPLOYABILITY_RECEIPT"
        --normalized-manifest "$STAGE123_NORMALIZED_MANIFEST"
        --preflight-receipt "$STAGE123_PREFLIGHT_RECEIPT"
        --report "$STAGE123_CALIBRATION_REPORT"
        --policy "$STAGE123_CALIBRATION_POLICY"
        --history-index "$STAGE123_CALIBRATION_HISTORY_INDEX"
        --prediction-contract "$STAGE123_CALIBRATION_PREDICTION_CONTRACT"
        --queue-identity "$STAGE123_FORMAL_QUEUE_ID"
        --profile-hash "$STAGE123_EXPECTED_PROFILE_HASH"
        --max-age-seconds "${STAGE123_DEPLOYABILITY_RECEIPT_MAX_AGE_SECONDS:-86400}"
        --future-skew-seconds "${STAGE123_DEPLOYABILITY_RECEIPT_FUTURE_SKEW_SECONDS:-300}"
    )
    if [ -n "${STAGE123_CALIBRATION_SEMANTIC_CONTRACT:-}" ]; then
        args+=(--semantic-contract "$STAGE123_CALIBRATION_SEMANTIC_CONTRACT")
    fi
    python3 "${STAGE123_GATE_DIR}/stage123_deployability_receipt.py" "${args[@]}"
}

stage123_require_formal_admission() {
    local run_id=${1:?run id required}
    stage123_require_preflight_receipt "$run_id"
    stage123_require_deployability_receipt
}
