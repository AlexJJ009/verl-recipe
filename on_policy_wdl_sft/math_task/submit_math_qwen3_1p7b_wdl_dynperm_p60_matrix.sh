#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Submit the frozen 2x4 DynPerm matrix to the three-node L40S partition.
# Default behavior is a non-mutating plan preview. Real submission additionally
# requires DYNPERM_SUBMIT_AUTHORIZED=1 and four candidate-bound launch receipts.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RECIPE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ "$#" -ne 0 ]; then
    echo "ERROR: matrix submitter accepts no positional config overrides" >&2
    exit 1
fi

RHO_VALUES=(0 1 0.25 0.5)
ARM_IDS=(fixed-m1-stage1 standard-c)
ARM_SBATCH=(
    "${SCRIPT_DIR}/slurm/run_math_qwen3_1p7b_wdl_dynperm_fixed_m1_p60.sbatch"
    "${SCRIPT_DIR}/slurm/run_math_qwen3_1p7b_wdl_dynperm_standard_c_p60.sbatch"
)
ARM_NICE=(0 1000)
PARENT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
RECIPE_SHA="$(git -C "$RECIPE_ROOT" rev-parse HEAD)"
IMAGE_ID="$(docker image inspect verl-harness:latest --format '{{.Id}}')"
RECEIPT_ROOT=/data-2/model_weights/math_task/qwen3_1p7b_wdl_dynperm/admission

canonical_tag() {
    python3 - "$1" <<'PY'
import math
import sys

rho = float(sys.argv[1])
if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
    raise SystemExit("rho must be finite and in [0, 1]")
canonical = format(rho, ".12g")
print(f"rho{canonical.replace('.', 'p')}")
PY
}

echo "DynPerm P60 matrix plan: parent=${PARENT_SHA} recipe=${RECIPE_SHA} image=${IMAGE_ID}"
echo "Scientific treatment interface: DYNPERM_ENABLED=true and DYNPERM_RHO in {0,1,0.25,0.5}"
echo "Submission priority: all fixed-m1-stage1 jobs before lower-priority standard-c jobs"
for arm_index in "${!ARM_IDS[@]}"; do
    for rho in "${RHO_VALUES[@]}"; do
        dose_tag="$(canonical_tag "$rho")"
        echo "PLAN arm=${ARM_IDS[$arm_index]} rho=${rho} horizon=60 receipt=${RECEIPT_ROOT}/p60-${dose_tag}.json"
    done
done

if [ "${DYNPERM_SUBMIT_AUTHORIZED:-0}" != "1" ]; then
    echo "PREVIEW ONLY: set DYNPERM_SUBMIT_AUTHORIZED=1 only after merge and explicit formal launch authorization"
    exit 0
fi

test "$(git -C "$REPO_ROOT" branch --show-current)" = codex/stage123-validation-protocol-rerun
test "$(git -C "$RECIPE_ROOT" branch --show-current)" = codex/stage123-model2-kl-split-stage3
test "$(git -C "$REPO_ROOT" rev-parse HEAD:recipe)" = "$RECIPE_SHA"
test -z "$(git -C "$REPO_ROOT" status --porcelain)"
test -z "$(git -C "$RECIPE_ROOT" status --porcelain)"
test "$(sinfo -N -h -p l40s -o '%N' | sort -u | wc -l)" -eq 3
: "${DYNPERM_EVIDENCE_RELAY_HOST:?set the controller SSH relay only after formal launch authorization}"
[[ "$DYNPERM_EVIDENCE_RELAY_HOST" =~ ^[A-Za-z0-9._@:-]+$ ]]
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
    "$DYNPERM_EVIDENCE_RELAY_HOST" true

submission_root="${RECEIPT_ROOT}/submissions/$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$submission_root"
mkdir -p "$submission_root"

for arm_index in "${!ARM_IDS[@]}"; do
    arm_id="${ARM_IDS[$arm_index]}"
    sbatch_file="${ARM_SBATCH[$arm_index]}"
    nice_value="${ARM_NICE[$arm_index]}"
    for rho in "${RHO_VALUES[@]}"; do
        dose_tag="$(canonical_tag "$rho")"
        launch_receipt="${RECEIPT_ROOT}/p60-${dose_tag}.json"
        test -f "$launch_receipt"
        job_name="DP-${dose_tag}-${arm_id}"
        job_id="$(sbatch --parsable --nice="$nice_value" --job-name="$job_name" \
            --export="ALL,DYNPERM_ENABLED=true,DYNPERM_RHO=${rho},DYNPERM_PARENT_SHA=${PARENT_SHA},DYNPERM_RECIPE_SHA=${RECIPE_SHA},DYNPERM_IMAGE_ID=${IMAGE_ID},DYNPERM_LAUNCH_RECEIPT=${launch_receipt},DYNPERM_EVIDENCE_RELAY_HOST=${DYNPERM_EVIDENCE_RELAY_HOST}" \
            "$sbatch_file")"
        relay_job_root="/data-1/code/_artifacts/verl-v0.7/dynperm-formal-p60/${PARENT_SHA}/${job_id}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s:%s\n' \
            "$job_id" "$arm_id" "$rho" "$PARENT_SHA" "$RECIPE_SHA" "$IMAGE_ID" \
            "$DYNPERM_EVIDENCE_RELAY_HOST" "$relay_job_root" \
            >>"${submission_root}/jobs.tsv"
        echo "SUBMITTED job=${job_id} arm=${arm_id} rho=${rho} nice=${nice_value}"
    done
done
echo "Submission receipt: ${submission_root}/jobs.tsv"
