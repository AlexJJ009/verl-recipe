#!/usr/bin/env bash
# Stage2 wrapper for the Stage123 family; all resource values come from one profile.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/qwen3_1p7b_stage123_resource_profile.sh"
: "${STAGE123_RUN_ID:?STAGE123_RUN_ID required}"
source "${SCRIPT_DIR}/stage123_manifest_gate.sh"
stage123_require_formal_admission "$STAGE123_RUN_ID"
: "${BASE_MODEL_PATH:?formal Stage2 admission must provide Model1 path}"
: "${EXPECTED_MODEL1_PATH:?formal Stage2 admission must bind Model1 identity}"
stage123_print_profile STAGE2

export LR=${LR:-1e-6}
export LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-0}
export TRACK_JOINT_SUBMODEL_LOSSES=${TRACK_JOINT_SUBMODEL_LOSSES:-true}
export JOINT_VALIDATION_VIEWS=${JOINT_VALIDATION_VIEWS:-"[model1,model2]"}
export BEST_CKPT_METRIC_KEY=${BEST_CKPT_METRIC_KEY:-val-core/model2/code3_macro/acc/mean@3}

export RAY_TMPDIR="${STAGE123_RAY_TMPDIR:-/tmp/stage123-ray-${STAGE123_RUN_ID}}"
cleanup_stage123_ray() {
    ray stop --force >/dev/null 2>&1 || true
    rm -rf "$RAY_TMPDIR"
}
trap cleanup_stage123_ray EXIT
rm -rf "$RAY_TMPDIR"
ray start --head --port=22000 --min-worker-port=21000 --max-worker-port=21999 \
  --temp-dir="$RAY_TMPDIR" --include-dashboard=false --disable-usage-stats >/dev/null
export RAY_ADDRESS="127.0.0.1:22000"

set +e
mapfile -t macro_overrides < <(code_stage123_macro_overrides)
bash "${SCRIPT_DIR}/run_s2_code_model2_rollout_common.sh" "${macro_overrides[@]}" "$@"
status=$?
set -e
exit "$status"
