#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RECIPE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PARENT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
RECIPE_SHA="$(git -C "$RECIPE_ROOT" rev-parse HEAD)"
IMAGE_ID="$(python3 "$REPO_ROOT/scripts/l40s/resolve_image_config_digest.py" verl-harness:latest)"

echo "Strict A: parent=$PARENT_SHA recipe=$RECIPE_SHA image=$IMAGE_ID"
echo "PLAN strict scorer / single-model On-Policy SFT / P60 / one shared A anchor"
[ "${STRICT_A_SUBMIT_AUTHORIZED:-0}" = 1 ] || { echo "PREVIEW ONLY"; exit 0; }
test "$(git -C "$REPO_ROOT" rev-parse HEAD:recipe)" = "$RECIPE_SHA"
test -z "$(git -C "$REPO_ROOT" status --porcelain)"
test -z "$(git -C "$RECIPE_ROOT" status --porcelain)"
: "${STRICT_A_LAUNCH_RECEIPT:?}"
: "${STRICT_A_EVIDENCE_RELAY_HOST:?}"
: "${STRICT_A_NODE_ROOT_MAP:?}"
: "${STRICT_A_STAGE_REL:?}"
: "${STRICT_A_ALLOWED_NODES:?}"

python3 - "$STRICT_A_LAUNCH_RECEIPT" "$PARENT_SHA" "$RECIPE_SHA" "$IMAGE_ID" <<'PY'
import json, sys
from pathlib import Path
receipt = json.loads(Path(sys.argv[1]).read_text())
expected = {"status":"authorized", "experiment_id":"math_qwen3_1p7b_opsft_a_strict_p60",
    "max_training_steps":60, "parent_candidate_sha":sys.argv[2], "recipe_candidate_sha":sys.argv[3],
    "image_id":sys.argv[4], "runs":[{"id":"strict-a", "method":"single-model-on-policy-sft"}]}
for key, value in expected.items():
    if receipt.get(key) != value: raise SystemExit(f"{key} mismatch")
PY

job="$(sbatch --parsable --hold --nodelist="$STRICT_A_ALLOWED_NODES" --nice=200 \
    --job-name=A-strict-P60 \
    --export="ALL,STRICT_A_PARENT_SHA=$PARENT_SHA,STRICT_A_RECIPE_SHA=$RECIPE_SHA,STRICT_A_IMAGE_ID=$IMAGE_ID,STRICT_A_EVIDENCE_RELAY_HOST=$STRICT_A_EVIDENCE_RELAY_HOST,STRICT_A_NODE_ROOT_MAP=$STRICT_A_NODE_ROOT_MAP,STRICT_A_STAGE_REL=$STRICT_A_STAGE_REL" \
    "$SCRIPT_DIR/slurm/run_math_qwen3_1p7b_opsft_a_strict_p60.sbatch")"
trap 'scancel "$job" >/dev/null 2>&1 || true' EXIT
jobs_file="$(dirname "$STRICT_A_LAUNCH_RECEIPT")/jobs.tsv"
printf 'job_id\tmethod\tfinal_step\tparent\trecipe\timage\n%s\tsingle-model-on-policy-sft\t60\t%s\t%s\t%s\n' \
    "$job" "$PARENT_SHA" "$RECIPE_SHA" "$IMAGE_ID" >"$jobs_file"
scontrol release "$job"
trap - EXIT
echo "SUBMITTED job=$job receipt=$jobs_file"
