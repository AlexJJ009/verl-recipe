#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RECIPE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ARM_VALUES=(fixed-m1 d0 fixed-m1 d0 standard-c standard-c)
LAMBDA_VALUES=(0.7 0.7 0.9 0.9 0.5 0.8)
LR_VALUES=(1e-6 1e-6 1e-6 1e-6 5e-7 5e-7)
NICE_VALUES=(0 10 20 30 100 110)
PARENT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
RECIPE_SHA="$(git -C "$RECIPE_ROOT" rev-parse HEAD)"
IMAGE_ID="$(python3 "$REPO_ROOT/scripts/l40s/resolve_image_config_digest.py" verl-harness:latest)"

echo "Lambda matrix: parent=$PARENT_SHA recipe=$RECIPE_SHA image=$IMAGE_ID"
for i in "${!ARM_VALUES[@]}"; do echo "PLAN priority=$i arm=${ARM_VALUES[$i]} lambda=${LAMBDA_VALUES[$i]} lr=${LR_VALUES[$i]} P60"; done
[ "${LAMBDA_MATRIX_SUBMIT_AUTHORIZED:-0}" = 1 ] || { echo "PREVIEW ONLY"; exit 0; }

test "$(git -C "$REPO_ROOT" rev-parse HEAD:recipe)" = "$RECIPE_SHA"
test -z "$(git -C "$REPO_ROOT" status --porcelain)"
test -z "$(git -C "$RECIPE_ROOT" status --porcelain)"
: "${LAMBDA_MATRIX_LAUNCH_RECEIPT:?}"
: "${LAMBDA_MATRIX_EVIDENCE_RELAY_HOST:?}"
: "${LAMBDA_MATRIX_NODE_ROOT_MAP:?}"
: "${LAMBDA_MATRIX_STAGE_REL:?}"
: "${LAMBDA_MATRIX_ALLOWED_NODES:?}"

python3 - "$LAMBDA_MATRIX_LAUNCH_RECEIPT" "$PARENT_SHA" "$RECIPE_SHA" "$IMAGE_ID" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1]); parent, recipe, image = sys.argv[2:]
receipt = json.loads(path.read_text())
expected = {"status":"authorized", "experiment_id":"math_qwen3_1p7b_wdl_lambda_followup_p60",
    "max_training_steps":60, "parent_candidate_sha":parent, "recipe_candidate_sha":recipe, "image_id":image}
for key, value in expected.items():
    if receipt.get(key) != value: raise SystemExit(f"{key} mismatch")
expected_runs = {"lambda07-fixed-lr1e6", "lambda09-fixed-lr1e6", "lambda07-d0-lr1e6", "lambda09-d0-lr1e6", "lambda05-c-lr5e7", "lambda08-c-lr5e7"}
if {run["id"] for run in receipt.get("runs", [])} != expected_runs: raise SystemExit("exact six-run receipt required")
expected_cells = {
    ("fixed-m1", "0.7", "1e-6"), ("fixed-m1", "0.9", "1e-6"),
    ("d0", "0.7", "1e-6"), ("d0", "0.9", "1e-6"),
    ("standard-c", "0.5", "5e-7"), ("standard-c", "0.8", "5e-7"),
}
cells = {(run["arm"], str(run["fusion_lambda"]), run["lr"]) for run in receipt.get("runs", [])}
if cells != expected_cells: raise SystemExit("exact six-run cell contract required")
PY

submitted=()
rollback() { for job in "${submitted[@]}"; do scancel "$job" >/dev/null 2>&1 || true; done; }
trap rollback EXIT
jobs_file="$(dirname "$LAMBDA_MATRIX_LAUNCH_RECEIPT")/jobs.tsv"
printf 'job_id\tarm\tfusion_lambda\ttraining_lr\tnice\tparent\trecipe\timage\n' >"$jobs_file"
for i in "${!ARM_VALUES[@]}"; do
    arm="${ARM_VALUES[$i]}"; lam="${LAMBDA_VALUES[$i]}"; lr="${LR_VALUES[$i]}"; nice="${NICE_VALUES[$i]}"
    name="LF-${arm}-l${lam/./}-${lr//-/}-P60"
    job="$(sbatch --parsable --hold --nodelist="$LAMBDA_MATRIX_ALLOWED_NODES" --nice="$nice" \
        --job-name="$name" \
        --export="ALL,LAMBDA_ARM=$arm,FUSION_LAMBDA=$lam,TRAINING_LR=$lr,LAMBDA_MATRIX_PARENT_SHA=$PARENT_SHA,LAMBDA_MATRIX_RECIPE_SHA=$RECIPE_SHA,LAMBDA_MATRIX_IMAGE_ID=$IMAGE_ID,LAMBDA_MATRIX_EVIDENCE_RELAY_HOST=$LAMBDA_MATRIX_EVIDENCE_RELAY_HOST,LAMBDA_MATRIX_NODE_ROOT_MAP=$LAMBDA_MATRIX_NODE_ROOT_MAP,LAMBDA_MATRIX_STAGE_REL=$LAMBDA_MATRIX_STAGE_REL" \
        "$SCRIPT_DIR/slurm/run_math_qwen3_1p7b_wdl_lambda_matrix_p60.sbatch")"
    submitted+=("$job")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$job" "$arm" "$lam" "$lr" "$nice" "$PARENT_SHA" "$RECIPE_SHA" "$IMAGE_ID" >>"$jobs_file"
done
scontrol release "$(IFS=,; echo "${submitted[*]}")"
trap - EXIT
echo "SUBMITTED jobs=$(IFS=,; echo "${submitted[*]}") receipt=$jobs_file"
