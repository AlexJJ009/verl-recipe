#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Internal Slurm job body. The arm is fixed by the submitted sbatch file; the
# only scientific treatment inputs remain DYNPERM_ENABLED and DYNPERM_RHO.
set -euo pipefail

die() {
    echo "DynPerm P60 Slurm admission failed: $*" >&2
    exit 64
}

if [ "$#" -ne 1 ]; then
    die "one static arm id is required"
fi
arm_id="$1"
case "$arm_id" in
    fixed-m1-stage1)
        arm_entry=run_math_qwen3_1p7b_wdl_dynperm_fixed_m1_p60.sh
        expected_gradient=zero
        ;;
    standard-c)
        arm_entry=run_math_qwen3_1p7b_wdl_dynperm_standard_c_p60.sh
        expected_gradient=nonzero
        ;;
    *) die "unsupported arm: $arm_id" ;;
esac

: "${SLURM_JOB_ID:?must run inside Slurm}"
: "${SLURMD_NODENAME:?Slurm node identity required}"
: "${CUDA_VISIBLE_DEVICES:?Slurm GPU allocation required}"
: "${DYNPERM_ENABLED:?required}"
: "${DYNPERM_RHO:?required}"
: "${DYNPERM_PARENT_SHA:?exact parent candidate required}"
: "${DYNPERM_RECIPE_SHA:?exact recipe candidate required}"
: "${DYNPERM_IMAGE_ID:?exact container image id required}"
: "${DYNPERM_LAUNCH_RECEIPT:?candidate-bound launch receipt required}"

case "${DYNPERM_ENABLED,,}" in
    true|1) export DYNPERM_ENABLED=true ;;
    *) die "DYNPERM_ENABLED must be true" ;;
esac
[[ "$DYNPERM_PARENT_SHA" =~ ^[0-9a-f]{40}$ ]] || die "invalid parent SHA"
[[ "$DYNPERM_RECIPE_SHA" =~ ^[0-9a-f]{40}$ ]] || die "invalid recipe SHA"

rho_and_tag="$(python3 - "$DYNPERM_RHO" <<'PY'
import math
import sys

rho = float(sys.argv[1])
if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
    raise SystemExit("DYNPERM_RHO must be finite and in [0, 1]")
canonical = format(rho, ".12g")
print(f"{canonical} rho{canonical.replace('.', 'p')}")
PY
)"
read -r DYNPERM_RHO dose_tag <<<"$rho_and_tag"
export DYNPERM_RHO

repo_host=/data-1/code/verl
recipe_host="$repo_host/recipe"
test -d "$repo_host/.git" || die "formal parent checkout missing: $repo_host"
test -e "$recipe_host/.git" || die "formal recipe checkout missing: $recipe_host"
test "$(git -C "$repo_host" branch --show-current)" = codex/stage123-validation-protocol-rerun \
    || die "parent checkout is not the formal training branch"
test "$(git -C "$recipe_host" branch --show-current)" = codex/stage123-model2-kl-split-stage3 \
    || die "recipe checkout is not the formal training branch"
test "$(git -C "$repo_host" rev-parse HEAD)" = "$DYNPERM_PARENT_SHA" \
    || die "parent candidate mismatch"
test "$(git -C "$recipe_host" rev-parse HEAD)" = "$DYNPERM_RECIPE_SHA" \
    || die "recipe candidate mismatch"
test "$(git -C "$repo_host" rev-parse HEAD:recipe)" = "$DYNPERM_RECIPE_SHA" \
    || die "parent gitlink does not bind the recipe candidate"
test -z "$(git -C "$repo_host" status --porcelain)" || die "parent checkout is dirty"
test -z "$(git -C "$recipe_host" status --porcelain)" || die "recipe checkout is dirty"

image_id="$(docker image inspect verl-harness:latest --format '{{.Id}}')" \
    || die "verl-harness:latest is unavailable"
test "$image_id" = "$DYNPERM_IMAGE_ID" || die "container image identity mismatch"

# Slurm owns the exclusive allocation, but foreign processes can exist outside
# Slurm. Refuse to disturb them; never kill or requeue another workload.
foreign_gpu_processes="$(nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name \
    --format=csv,noheader,nounits 2>/dev/null || true)"
test -z "$foreign_gpu_processes" || die "foreign GPU compute process present"
mem_available_gib="$(awk '/MemAvailable:/ {print int($2 / 1024 / 1024)}' /proc/meminfo)"
test "$mem_available_gib" -ge 300 || die "less than 300 GiB host memory available"

artifact_root="/data-2/model_weights/math_task/qwen3_1p7b_wdl_dynperm/${dose_tag}/${arm_id}-p60"
job_root="${artifact_root}/slurm/${SLURM_JOB_ID}"
test ! -e "$job_root" || die "job artifact root already exists: $job_root"
mkdir -p "$job_root"
container_name="dynperm-p60-${SLURM_JOB_ID}-${arm_id}"
training_pid=""

cleanup() {
    rc=$?
    trap - EXIT TERM INT
    if [ -n "$training_pid" ]; then
        kill "$training_pid" >/dev/null 2>&1 || true
        wait "$training_pid" >/dev/null 2>&1 || true
    fi
    docker rm --force "$container_name" >/dev/null 2>&1 || true
    printf '{"job_id":"%s","node":"%s","arm":"%s","rho":%s,"exit_code":%s,"parent_sha":"%s","recipe_sha":"%s","image_id":"%s"}\n' \
        "$SLURM_JOB_ID" "$SLURMD_NODENAME" "$arm_id" "$DYNPERM_RHO" "$rc" \
        "$DYNPERM_PARENT_SHA" "$DYNPERM_RECIPE_SHA" "$DYNPERM_IMAGE_ID" \
        >"${job_root}/terminal.json"
    exit "$rc"
}
trap cleanup EXIT TERM INT

printf '{"job_id":"%s","node":"%s","arm":"%s","rho":%s,"parent_sha":"%s","recipe_sha":"%s","image_id":"%s","launch_receipt":"%s"}\n' \
    "$SLURM_JOB_ID" "$SLURMD_NODENAME" "$arm_id" "$DYNPERM_RHO" \
    "$DYNPERM_PARENT_SHA" "$DYNPERM_RECIPE_SHA" "$DYNPERM_IMAGE_ID" \
    "$DYNPERM_LAUNCH_RECEIPT" >"${job_root}/admission.json"

export REPO_HOST="$repo_host"
export REPO_CONTAINER=/workspace/verl
export DOCKER_IMAGE="$DYNPERM_IMAGE_ID"
export DOCKER_CONTAINER_NAME="$container_name"
export WANDB_MODE=offline

bash "$repo_host/scripts/l40s/run_train.sh" \
    bash "recipe/on_policy_wdl_sft/math_task/${arm_entry}" \
    >"${job_root}/stdout.log" 2>"${job_root}/stderr.log" &
training_pid=$!

run_prefix="MATH-WDL-DYNPERM-${dose_tag^^}-${arm_id^^}-P60-QWEN3-1P7B"
if ! python3 "$repo_host/scripts/math_wdl_first_step_gate.py" \
    --metrics-root "${artifact_root}/logs/metrics" \
    --project OnPolicyWDLSFT-Math-1P7B-DynPerm-P60 \
    --run-prefix "$run_prefix" \
    --expected-model1-gradient "$expected_gradient" \
    --dynperm-rho "$DYNPERM_RHO" \
    --slurm-job-id "$SLURM_JOB_ID" \
    --timeout-seconds 43200 \
    --receipt "${job_root}/first-step.json" \
    >"${job_root}/first-step.log" 2>&1; then
    echo "first-step admission failed; stopping only this job's training container" >&2
    kill "$training_pid" >/dev/null 2>&1 || true
    wait "$training_pid" >/dev/null 2>&1 || true
    training_pid=""
    exit 65
fi

wait "$training_pid"
training_pid=""
