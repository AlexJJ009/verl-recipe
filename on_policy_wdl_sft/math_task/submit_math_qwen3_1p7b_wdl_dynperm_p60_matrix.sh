#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Submit the frozen 2x4 DynPerm matrix to an explicitly allowed subset of the
# L40S partition.
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
IMAGE_ID="$(python3 "$REPO_ROOT/scripts/l40s/resolve_image_config_digest.py" verl-harness:latest)"
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
: "${DYNPERM_EVIDENCE_RELAY_HOST:?set the controller SSH relay only after formal launch authorization}"
: "${DYNPERM_NODE_ROOT_MAP:?set the semicolon-separated Slurm node-root map after staging}"
: "${DYNPERM_STAGE_REL:?set the candidate-bound workspace/jobs stage path}"
: "${DYNPERM_ALLOWED_NODES:?set the comma-separated Slurm nodes admitted for this submission}"
[[ "$DYNPERM_EVIDENCE_RELAY_HOST" =~ ^[A-Za-z0-9._@:-]+$ ]]
[[ "$DYNPERM_STAGE_REL" =~ ^workspace/jobs/[A-Za-z0-9._-]+$ ]]
[[ "$DYNPERM_ALLOWED_NODES" =~ ^[A-Za-z0-9._-]+(,[A-Za-z0-9._-]+)*$ ]]
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
    "$DYNPERM_EVIDENCE_RELAY_HOST" true

IFS=',' read -r -a slurm_nodes <<<"$DYNPERM_ALLOWED_NODES"
allowed_node_list="$(IFS=,; echo "${slurm_nodes[*]}")"
test "$allowed_node_list" = "$DYNPERM_ALLOWED_NODES"
mapfile -t partition_nodes < <(sinfo -N -h -p l40s -o '%N' | sort -u)
declare -A seen_allowed_nodes=()
for node in "${slurm_nodes[@]}"; do
    [ -z "${seen_allowed_nodes[$node]:-}" ]
    seen_allowed_nodes[$node]=1
    printf '%s\n' "${partition_nodes[@]}" | grep -Fx -- "$node" >/dev/null
done

node_root_for() {
    local wanted=$1 entry found=""
    local -a entries
    IFS=';' read -r -a entries <<<"$DYNPERM_NODE_ROOT_MAP"
    for entry in "${entries[@]}"; do
        if [ "${entry%%=*}" = "$wanted" ]; then
            [ -z "$found" ] || return 1
            found="${entry#*=}"
        fi
    done
    [ -n "$found" ] && [ "${found#/}" != "$found" ]
    printf '%s\n' "$found"
}

submission_root="${RECEIPT_ROOT}/submissions/$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$submission_root"
mkdir -p "$submission_root"

# Validate the entire authorization set before the first sbatch call. A missing
# or stale later dose must never leave a partially authorized matrix queued.
launch_receipts=()
for rho in "${RHO_VALUES[@]}"; do
    dose_tag="$(canonical_tag "$rho")"
    launch_receipt="${RECEIPT_ROOT}/p60-${dose_tag}.json"
    test -f "$launch_receipt"
    launch_receipts+=("$launch_receipt")
done
python3 - "$PARENT_SHA" "$RECIPE_SHA" "$IMAGE_ID" "${launch_receipts[@]}" <<'PY'
import json
import sys
from pathlib import Path

parent_sha, recipe_sha, image_id, *paths = sys.argv[1:]
rhos = [0.0, 1.0, 0.25, 0.5]
if len(paths) != len(rhos):
    raise SystemExit("incomplete DynPerm launch receipt set")
for rho, raw_path in zip(rhos, paths, strict=True):
    path = Path(raw_path)
    receipt = json.loads(path.read_text())
    expected = {
        "status": "authorized",
        "experiment_id": "math_qwen3_1p7b_wdl_dynperm_p60",
        "rho": rho,
        "max_training_steps": 60,
        "parent_candidate_sha": parent_sha,
        "recipe_candidate_sha": recipe_sha,
        "image_id": image_id,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise SystemExit(f"{path}: {key} mismatch")
    if set(receipt.get("arms", [])) != {"fixed-m1-stage1", "standard-c"}:
        raise SystemExit(f"{path}: exact two-arm authorization required")
print("DynPerm four-dose launch receipt set PASS")
PY

# Exercise the real worker-to-controller rsync path on every admitted Slurm
# node before reserving GPUs. The resulting tiny hostname files are durable
# relay evidence.
relay_preflight_root="/data-1/code/_artifacts/verl-v0.7/dynperm-formal-p60/${PARENT_SHA}/preflight/$(basename "$submission_root")"
for node in "${slurm_nodes[@]}"; do
    node_addr="$(scontrol show node "$node" -o | tr ' ' '\n' | sed -n 's/^NodeAddr=//p')"
    node_root="$(node_root_for "$node")"
    [[ "$node_addr" =~ ^[A-Za-z0-9._:-]+$ ]]
    [[ "$node" =~ ^[A-Za-z0-9._-]+$ ]]
    ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$node_addr" \
        "rsync --archive --mkpath --protect-args -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10' /etc/hostname '${DYNPERM_EVIDENCE_RELAY_HOST}:${relay_preflight_root}/${node}/'"
    ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
        "$DYNPERM_EVIDENCE_RELAY_HOST" test -s "${relay_preflight_root}/${node}/hostname"
    ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$node_addr" \
        bash -s -- "$node_root" "$DYNPERM_STAGE_REL" "$PARENT_SHA" "$RECIPE_SHA" "$IMAGE_ID" <<'REMOTE'
set -euo pipefail
node_root=$1
stage_rel=$2
parent_sha=$3
recipe_sha=$4
image_id=$5
node_root="$(realpath -e "$node_root")"
test "$node_root" != /
workspace="$(realpath -e "${node_root}/${stage_rel}")"
case "$workspace" in
    "$node_root"/workspace/jobs/*) ;;
    *) exit 64 ;;
esac
test "$(tr -d '\n' <"${workspace}/.candidate-parent-sha")" = "$parent_sha"
test "$(tr -d '\n' <"${workspace}/.candidate-recipe-sha")" = "$recipe_sha"
repo="${workspace}/repo"
data1="${workspace}/runtime/data-1"
data2="${workspace}/runtime/data-2"
test "$(git -C "$repo" rev-parse HEAD)" = "$parent_sha"
test "$(git -C "$repo/recipe" rev-parse HEAD)" = "$recipe_sha"
test "$(git -C "$repo" rev-parse HEAD:recipe)" = "$recipe_sha"
test -z "$(git -C "$repo" status --porcelain)"
test -z "$(git -C "$repo/recipe" status --porcelain)"
test -r "$repo/recipe/on_policy_wdl_sft/math_task/slurm/run_math_qwen3_1p7b_wdl_dynperm_p60_job.sh"
local_image_ref="$(docker image inspect verl-harness:latest --format '{{.Id}}')"
[[ "$local_image_ref" =~ ^sha256:[0-9a-f]{64}$ ]]
test "$(python3 "$repo/scripts/l40s/resolve_image_config_digest.py" "$local_image_ref")" = "$image_id"
test -f "${data1}/dataset/math/qwen3_1p7b_stage123_seed20260719/stage1_control_stage2_then_stage3.parquet"
test -d "${data1}/dataset/math/qwen3_1p7b_math7_validation_v1"
test -f "${data1}/code/_artifacts/verl-v0.7/linear-gon-34-dynperm-mvp/slurm-job-146/runtime-output/gpu_fsdp_smoke_receipt.json"
test -d "${data2}/model_weights/math_task/qwen3_1p7b_cold_start_cotmask_v3/candidates/step_20"
test -d "${data2}/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/restored_from_causal_p60_joint_20260812/final_model"
test -f "${data2}/model_weights/math_task/qwen3_1p7b_wdl_causal_p60/admission/manipulation_receipt.json"
for tag in rho0 rho1 rho0p25 rho0p5; do
    test -f "${data2}/model_weights/math_task/qwen3_1p7b_wdl_dynperm/admission/p60-${tag}.json"
done
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"
REMOTE
done

submitted_job_ids=()
rollback_held_jobs() {
    local job_id
    for job_id in "${submitted_job_ids[@]}"; do
        scancel "$job_id" >/dev/null 2>&1 || true
    done
}
trap rollback_held_jobs EXIT

for arm_index in "${!ARM_IDS[@]}"; do
    arm_id="${ARM_IDS[$arm_index]}"
    sbatch_file="${ARM_SBATCH[$arm_index]}"
    nice_value="${ARM_NICE[$arm_index]}"
    for rho in "${RHO_VALUES[@]}"; do
        dose_tag="$(canonical_tag "$rho")"
        launch_receipt="${RECEIPT_ROOT}/p60-${dose_tag}.json"
        job_name="DP-${dose_tag}-${arm_id}"
        if ! job_id="$(sbatch --parsable --hold --nodelist="$allowed_node_list" \
            --nice="$nice_value" --job-name="$job_name" \
            --export="ALL,DYNPERM_ENABLED=true,DYNPERM_RHO=${rho},DYNPERM_PARENT_SHA=${PARENT_SHA},DYNPERM_RECIPE_SHA=${RECIPE_SHA},DYNPERM_IMAGE_ID=${IMAGE_ID},DYNPERM_LAUNCH_RECEIPT=${launch_receipt},DYNPERM_EVIDENCE_RELAY_HOST=${DYNPERM_EVIDENCE_RELAY_HOST},DYNPERM_NODE_ROOT_MAP=${DYNPERM_NODE_ROOT_MAP},DYNPERM_STAGE_REL=${DYNPERM_STAGE_REL}" \
            "$sbatch_file")"; then
            rollback_held_jobs
            exit 1
        fi
        submitted_job_ids+=("$job_id")
        relay_job_root="/data-1/code/_artifacts/verl-v0.7/dynperm-formal-p60/${PARENT_SHA}/${job_id}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\tallowed_nodes=%s\t%s:%s\n' \
            "$job_id" "$arm_id" "$rho" "$PARENT_SHA" "$RECIPE_SHA" "$IMAGE_ID" \
            "$allowed_node_list" "$DYNPERM_EVIDENCE_RELAY_HOST" "$relay_job_root" \
            >>"${submission_root}/jobs.tsv"
        echo "SUBMITTED job=${job_id} arm=${arm_id} rho=${rho} nice=${nice_value}"
    done
done
job_id_list="$(IFS=,; echo "${submitted_job_ids[*]}")"
if ! scontrol release "$job_id_list"; then
    rollback_held_jobs
    exit 1
fi
printf '%s\n' "$job_id_list" >"${submission_root}/released-job-ids.txt"
trap - EXIT
echo "Submission receipt: ${submission_root}/jobs.tsv"
