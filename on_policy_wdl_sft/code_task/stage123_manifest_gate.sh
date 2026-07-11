#!/usr/bin/env bash
# Shared formal-launch receipt gate for Stage123 queue and direct phase wrappers.
set -euo pipefail

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
    python3 /workspace/verl/scripts/stage123_preflight_receipt.py verify \
        --receipt "$STAGE123_PREFLIGHT_RECEIPT" \
        --normalized-manifest "$STAGE123_NORMALIZED_MANIFEST" \
        --report "$STAGE123_PREFLIGHT_REPORT" \
        --policy "$STAGE123_PREFLIGHT_POLICY" \
        --run-id "$run_id" \
        --profile-hash "$STAGE123_EXPECTED_PROFILE_HASH" \
        --max-age-seconds "$STAGE123_RECEIPT_MAX_AGE_SECONDS"
}
