#!/usr/bin/env bash
# Stage3: resume Stage1-like single-model training from extracted Stage2 model2.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/qwen3_1p7b_stage123_resource_profile.sh"
: "${STAGE123_RUN_ID:?STAGE123_RUN_ID required}"
source "${SCRIPT_DIR}/stage123_manifest_gate.sh"
stage123_require_formal_admission "$STAGE123_RUN_ID"
: "${RUN_PREFIX:?}"
: "${STAGE2_SUBMODEL:=model2}"
: "${STAGE2_MODEL_PATH:=${STAGE2_MODEL2_PATH:-}}"
: "${STAGE2_MODEL_PATH:?}"
: "${STAGE2_PROVENANCE_FILE:?}"
[ -f "$STAGE2_PROVENANCE_FILE" ] || { echo "ERROR: missing Stage2 provenance" >&2; exit 1; }
python3 - "$STAGE2_PROVENANCE_FILE" "$STAGE2_MODEL_PATH" "$STAGE2_SUBMODEL" "${DRY_RUN:-0}" <<'PY' || {
import json
import sys

provenance_path, expected_model, submodel, dry_run = sys.argv[1:]
with open(provenance_path, encoding="utf-8") as handle:
    provenance = json.load(handle)
source = provenance.get("source", {})
if dry_run == "1":
    if provenance.get("preflight_result_sha256") != "dry-run":
        raise SystemExit("dry-run Stage2 provenance lacks dry-run result sentinel")
elif provenance.get("release_eligible") is not True:
    raise SystemExit("Stage2 provenance is not release eligible")
if source.get(f"extracted_{submodel}") != expected_model:
    raise SystemExit(f"Stage2 extracted {submodel} path mismatch")
PY
    echo "ERROR: Stage3 provenance mismatch" >&2
    exit 1
}
export INIT_MODEL_PATH="$STAGE2_MODEL_PATH"
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.1}
export LOSS_MODE=${LOSS_MODE:-wdl_sft}
export LR=${LR:-5e-7}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-40}
export CODE_TRAIN_FILE=${CODE_TRAIN_FILE:?Stage3 CODE_TRAIN_FILE required}
export TRAIN_FILE=${TRAIN_FILE:-$CODE_TRAIN_FILE}
export DATA_SHUFFLE=${DATA_SHUFFLE:-False}
export CODE_ONLINE_LCB_V5_SUBSET_VAL_FILE=${CODE_ONLINE_LCB_V5_SUBSET_VAL_FILE:-/data-1/dataset/code/verl_rl/online_full_livecodebench_v5/official_livecodebench_val.parquet}
export CODE_VAL_FILES=${CODE_VAL_FILES:-"['/data-1/dataset/code/verl_rl/online_full_humaneval_plus/official_humaneval_plus_val.parquet','/data-1/dataset/code/verl_rl/online_full_mbpp_plus/official_mbpp_plus_val.parquet','$CODE_ONLINE_LCB_V5_SUBSET_VAL_FILE']"}
export TEST_FILES="$CODE_VAL_FILES"
stage123_print_profile STAGE3

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
bash "${SCRIPT_DIR}/run_s1_code_base.sh" "$@"
status=$?
set -e
exit "$status"
