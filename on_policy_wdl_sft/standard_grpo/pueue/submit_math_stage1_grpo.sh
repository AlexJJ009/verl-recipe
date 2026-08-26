#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SCHEDULER_DIR="${RECIPE_ROOT}/on_policy_wdl_sft/standard_grpo/scheduler"
WORKER_GIT_PATH="on_policy_wdl_sft/standard_grpo/pueue/worker_math_stage1_grpo.sh"

: "${PUEUE_GRPO_REPO_ROOT:?absolute parent VERL checkout is required}"
: "${PUEUE_GRPO_OUTPUT_ROOT:?absolute repository-external output root is required}"
: "${PUEUE_GRPO_RECEIPT_ROOT:?absolute repository-external receipt root is required}"
: "${PUEUE_GRPO_RUNTIME_ENV_FILE:?absolute external runtime environment file is required}"

canonical_path() {
    realpath -m -- "$1"
}

assert_absolute_external() {
    local name=$1 value=$2 canonical recipe_canonical parent_canonical
    [[ "$value" = /* ]] || { echo "ERROR: ${name} must be absolute" >&2; exit 64; }
    canonical=$(canonical_path "$value")
    recipe_canonical=$(canonical_path "$RECIPE_ROOT")
    parent_canonical=$(canonical_path "${RECIPE_ROOT}/..")
    case "$canonical/" in
        "$recipe_canonical/"*|"$parent_canonical/"*)
            echo "ERROR: ${name} must stay outside both repositories: ${canonical}" >&2
            exit 64
            ;;
    esac
}

repo_root=$(canonical_path "$PUEUE_GRPO_REPO_ROOT")
output_root=$(canonical_path "$PUEUE_GRPO_OUTPUT_ROOT")
receipt_root=$(canonical_path "$PUEUE_GRPO_RECEIPT_ROOT")
runtime_env_file=$(canonical_path "$PUEUE_GRPO_RUNTIME_ENV_FILE")

[[ "$repo_root" = /* ]] || { echo "ERROR: PUEUE_GRPO_REPO_ROOT must be absolute" >&2; exit 64; }
[[ -d "$repo_root/verl" && -d "$repo_root/recipe" ]] || {
    echo "ERROR: PUEUE_GRPO_REPO_ROOT is not a VERL checkout: ${repo_root}" >&2
    exit 66
}
launched_recipe=$(canonical_path "$repo_root/recipe")
recipe_canonical=$(canonical_path "$RECIPE_ROOT")
[[ "$launched_recipe" == "$recipe_canonical" ]] || {
    echo "ERROR: PUEUE_GRPO_REPO_ROOT must contain the reviewed recipe checkout: ${recipe_canonical}" >&2
    exit 66
}
assert_absolute_external PUEUE_GRPO_OUTPUT_ROOT "$output_root"
assert_absolute_external PUEUE_GRPO_RECEIPT_ROOT "$receipt_root"
assert_absolute_external PUEUE_GRPO_RUNTIME_ENV_FILE "$runtime_env_file"
[[ -f "$runtime_env_file" ]] || { echo "ERROR: runtime environment file missing: ${runtime_env_file}" >&2; exit 66; }

python3 "${SCHEDULER_DIR}/verify_job_130_baseline.py" >/dev/null
python3 "${SCHEDULER_DIR}/verify_scheduler_audit.py" >/dev/null
python3 "${SCRIPT_DIR}/verify_pueue_adapter.py" >/dev/null

label=${PUEUE_GRPO_LABEL:-gon-36-math-stage1-grpo}
recipe_candidate=$(git -C "$RECIPE_ROOT" rev-parse HEAD)
runtime_env_sha256=__ADMITTED_RUNTIME_ENV_SHA256__
worker_runner='set -euo pipefail
repo_root=$1
candidate=$2
worker_path=$3
shift 3
git -C "$repo_root/recipe" cat-file -e "${candidate}:${worker_path}"
git -C "$repo_root/recipe" show "${candidate}:${worker_path}" | bash -s -- "$@"'

build_pueue_command() {
    local task_command
    task_command=(
        bash -c "$worker_runner" gon36-candidate-worker
        "$repo_root" "$recipe_candidate" "$WORKER_GIT_PATH"
        "$repo_root" "$output_root" "$receipt_root" "$runtime_env_file"
        "$recipe_candidate" "$runtime_env_sha256"
    )
    pueue_command=(
        pueue add
        --group gpu8
        --label "$label"
        --working-directory "$repo_root"
        --print-task-id
        --
        "${task_command[@]}"
    )
}
build_pueue_command

if [[ "${PUEUE_GRPO_DRY_RUN:-0}" = 1 ]]; then
    printf 'DRY_RUN'
    printf ' %q' "${pueue_command[@]}"
    printf '\n'
    exit 0
fi

[[ "${PUEUE_GRPO_ALLOW_SUBMIT:-0}" = 1 ]] || {
    echo "ERROR: real submission requires PUEUE_GRPO_ALLOW_SUBMIT=1" >&2
    exit 77
}
: "${PUEUE_GRPO_A800_ADMISSION_RECEIPT:?real submission requires a GON-35 A800 admission receipt}"
admission_receipt=$(canonical_path "$PUEUE_GRPO_A800_ADMISSION_RECEIPT")
assert_absolute_external PUEUE_GRPO_A800_ADMISSION_RECEIPT "$admission_receipt"
[[ -f "$admission_receipt" ]] || { echo "ERROR: A800 admission receipt missing: ${admission_receipt}" >&2; exit 66; }

runtime_env_sha256=$(python3 "${SCRIPT_DIR}/validate_a800_admission.py" \
    --receipt "$admission_receipt" \
    --recipe-candidate "$recipe_candidate" \
    --runtime-env-file "$runtime_env_file" \
    --output-root "$output_root" \
    --receipt-root "$receipt_root" \
    --print-runtime-env-sha256)
build_pueue_command

command -v pueue >/dev/null 2>&1 || { echo "ERROR: pueue is not installed" >&2; exit 69; }
mkdir -p "$output_root" "$receipt_root"
task_id=$("${pueue_command[@]}")
[[ "$task_id" =~ ^[0-9]+$ ]] || { echo "ERROR: pueue returned invalid task ID: ${task_id}" >&2; exit 70; }

python3 - "$receipt_root/submission-${task_id}.json" "$task_id" "$recipe_candidate" "$admission_receipt" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

target = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "batch_id": "GON-36",
    "linear_issue": "GON-41",
    "scheduler": "pueue",
    "group": "gpu8",
    "task_id": int(sys.argv[2]),
    "recipe_candidate_sha": sys.argv[3],
    "a800_admission_receipt": sys.argv[4],
    "submitted_at": datetime.now(timezone.utc).isoformat(),
}
target.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, target)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY
printf '%s\n' "$task_id"
