#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Internal Slurm job body. The arm is fixed by the submitted sbatch file; the
# only scientific treatment inputs remain DYNPERM_ENABLED and DYNPERM_RHO.
set -euo pipefail

arm_id="${1:-unassigned}"
bootstrap_job_id="${SLURM_JOB_ID:-unassigned-$$}"
if [[ ! "$bootstrap_job_id" =~ ^[0-9]+$ ]]; then
    bootstrap_job_id="invalid-$$"
fi
bootstrap_relay_root="/data-1/code/_artifacts/verl-v0.7/dynperm-formal-p60/bootstrap/${bootstrap_job_id}"
bootstrap_reason="pre-admission shell failure"

bootstrap_cleanup() {
    local original_rc=$?
    local final_rc=$original_rc
    local bootstrap_receipt local_sha remote_sha
    trap - EXIT TERM INT
    set +e
    bootstrap_receipt="$(python3 - "$bootstrap_job_id" "${SLURMD_NODENAME:-unknown}" "$arm_id" \
        "$original_rc" "$bootstrap_reason" <<'PY'
import json
import sys

job_id, node, arm, exit_code, reason = sys.argv[1:]
print(json.dumps({
    "job_id": job_id,
    "node": node,
    "arm": arm,
    "phase": "pre-admission",
    "original_exit_code": int(exit_code),
    "reason": reason,
}, sort_keys=True))
PY
    )"
    if [ -z "$bootstrap_receipt" ]; then
        final_rc=70
    fi
    if [[ "${DYNPERM_EVIDENCE_RELAY_HOST:-}" =~ ^[A-Za-z0-9._@:-]+$ ]] \
        && [ -n "$bootstrap_receipt" ]; then
        local_sha="$(printf '%s\n' "$bootstrap_receipt" | sha256sum | awk '{print $1}')"
        printf '%s\n' "$bootstrap_receipt" | \
            ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
                "$DYNPERM_EVIDENCE_RELAY_HOST" \
                "mkdir -p -- '${bootstrap_relay_root}' && cat > '${bootstrap_relay_root}/bootstrap-terminal.json'"
        remote_sha="$(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
            "$DYNPERM_EVIDENCE_RELAY_HOST" sha256sum -- \
            "${bootstrap_relay_root}/bootstrap-terminal.json" 2>/dev/null | awk '{print $1}')"
        [ -n "$local_sha" ] && [ "$remote_sha" = "$local_sha" ] || final_rc=70
    else
        final_rc=70
    fi
    exit "$final_rc"
}
trap bootstrap_cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

die() {
    bootstrap_reason="$*"
    echo "DynPerm P60 Slurm admission failed: $*" >&2
    exit 64
}

require_env() {
    local name=$1
    [ -n "${!name:-}" ] || die "$name is required"
}

if [ "$#" -ne 1 ]; then
    die "one static arm id is required"
fi
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

for required_name in \
    SLURM_JOB_ID SLURMD_NODENAME CUDA_VISIBLE_DEVICES DYNPERM_ENABLED \
    DYNPERM_RHO DYNPERM_PARENT_SHA DYNPERM_RECIPE_SHA DYNPERM_IMAGE_ID \
    DYNPERM_LAUNCH_RECEIPT DYNPERM_EVIDENCE_RELAY_HOST \
    DYNPERM_NODE_ROOT_MAP DYNPERM_STAGE_REL; do
    require_env "$required_name"
done

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
case "$DYNPERM_RHO" in
    0|0.25|0.5|1) ;;
    *) die "formal DynPerm P60 rho must be one of 0, 0.25, 0.5, or 1" ;;
esac
export DYNPERM_RHO

[[ "$DYNPERM_STAGE_REL" =~ ^workspace/jobs/[A-Za-z0-9._-]+$ ]] \
    || die "invalid candidate stage path"
node_root=""
IFS=';' read -r -a node_roots <<<"$DYNPERM_NODE_ROOT_MAP"
for entry in "${node_roots[@]}"; do
    if [ "${entry%%=*}" = "$SLURMD_NODENAME" ]; then
        node_root="${entry#*=}"
        break
    fi
done
[ -n "$node_root" ] || die "node root is not mapped"
node_root="$(realpath -e "$node_root")" || die "node root is unavailable"
[ "$node_root" != / ] || die "node root must not be filesystem root"
workspace="$(realpath -e "${node_root}/${DYNPERM_STAGE_REL}")" \
    || die "candidate stage is unavailable"
case "$workspace" in
    "$node_root"/workspace/jobs/*) ;;
    *) die "candidate stage escapes node root" ;;
esac
data1_host="$(realpath -e "${workspace}/runtime/data-1")" \
    || die "staged data-1 root is unavailable"
data2_host="$(realpath -e "${workspace}/runtime/data-2")" \
    || die "staged data-2 root is unavailable"
repo_host="$(realpath -e "${workspace}/repo")" || die "staged repo is unavailable"
recipe_host="$repo_host/recipe"
test "$(tr -d '\n' <"${workspace}/.candidate-parent-sha")" = "$DYNPERM_PARENT_SHA" \
    || die "staged parent marker mismatch"
test "$(tr -d '\n' <"${workspace}/.candidate-recipe-sha")" = "$DYNPERM_RECIPE_SHA" \
    || die "staged recipe marker mismatch"
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

artifact_root="${data2_host}/model_weights/math_task/qwen3_1p7b_wdl_dynperm/${dose_tag}/${arm_id}-p60"
job_root="${artifact_root}/slurm/${SLURM_JOB_ID}"
test ! -e "$job_root" || die "job artifact root already exists: $job_root"
mkdir -p "$job_root"
container_name="dynperm-p60-${SLURM_JOB_ID}-${arm_id}"
training_pid=""
gate_pid=""
relay_root="/data-1/code/_artifacts/verl-v0.7/dynperm-formal-p60/${DYNPERM_PARENT_SHA}"
relay_job_root="${relay_root}/${SLURM_JOB_ID}"
[[ "$DYNPERM_EVIDENCE_RELAY_HOST" =~ ^[A-Za-z0-9._@:-]+$ ]] \
    || die "invalid evidence relay host"

relay_files() {
    local -a files=()
    local name
    for name in "$@"; do
        if [ -f "${job_root}/${name}" ]; then
            files+=("${job_root}/${name}")
        fi
    done
    [ "${#files[@]}" -gt 0 ] || return 0
    rsync --archive --mkpath --protect-args \
        -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10' \
        "${files[@]}" "${DYNPERM_EVIDENCE_RELAY_HOST}:${relay_job_root}/"
}

relay_terminal_verified() {
    local local_sha remote_sha
    local_sha="$(sha256sum "${job_root}/terminal.json" | awk '{print $1}')"
    if ! rsync --archive --mkpath --protect-args \
        -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10' \
        "${job_root}/terminal.json" "${DYNPERM_EVIDENCE_RELAY_HOST}:${relay_job_root}/"; then
        echo "WARNING: terminal rsync returned nonzero; checking the controller SHA" >&2
    fi
    remote_sha="$(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
        "$DYNPERM_EVIDENCE_RELAY_HOST" sha256sum -- "${relay_job_root}/terminal.json" \
        2>/dev/null | awk '{print $1}')" || return 1
    [ "$remote_sha" = "$local_sha" ]
}

cleanup() {
    rc=$?
    training_exit_code=$rc
    evidence_set_relayed=true
    trap - EXIT TERM INT
    if [ -n "$training_pid" ]; then
        kill "$training_pid" >/dev/null 2>&1 || true
        wait "$training_pid" >/dev/null 2>&1 || true
    fi
    if [ -n "$gate_pid" ]; then
        kill "$gate_pid" >/dev/null 2>&1 || true
        wait "$gate_pid" >/dev/null 2>&1 || true
    fi
    docker rm --force "$container_name" >/dev/null 2>&1 || true
    tail -n 200 "${job_root}/stdout.log" >"${job_root}/stdout.tail.log" 2>/dev/null || true
    tail -n 200 "${job_root}/stderr.log" >"${job_root}/stderr.tail.log" 2>/dev/null || true
    if ! relay_files admission.json first-step.json first-step.log stdout.tail.log stderr.tail.log; then
        echo "ERROR: failed to relay terminal evidence set to controller" >&2
        evidence_set_relayed=false
        rc=70
    fi
    printf '{"job_id":"%s","node":"%s","arm":"%s","rho":%s,"training_exit_code":%s,"evidence_set_relayed":%s,"parent_sha":"%s","recipe_sha":"%s","image_id":"%s"}\n' \
        "$SLURM_JOB_ID" "$SLURMD_NODENAME" "$arm_id" "$DYNPERM_RHO" "$training_exit_code" \
        "$evidence_set_relayed" \
        "$DYNPERM_PARENT_SHA" "$DYNPERM_RECIPE_SHA" "$DYNPERM_IMAGE_ID" \
        >"${job_root}/terminal.json"
    if ! relay_terminal_verified; then
        echo "ERROR: terminal receipt is not verifiable on controller" >&2
        rc=70
    fi
    exit "$rc"
}
trap - EXIT TERM INT
trap cleanup EXIT TERM INT

printf '{"job_id":"%s","node":"%s","arm":"%s","rho":%s,"parent_sha":"%s","recipe_sha":"%s","image_id":"%s","launch_receipt":"%s","node_local_job_root":"%s","relay_job_root":"%s"}\n' \
    "$SLURM_JOB_ID" "$SLURMD_NODENAME" "$arm_id" "$DYNPERM_RHO" \
    "$DYNPERM_PARENT_SHA" "$DYNPERM_RECIPE_SHA" "$DYNPERM_IMAGE_ID" \
    "$DYNPERM_LAUNCH_RECEIPT" "$job_root" "$relay_job_root" >"${job_root}/admission.json"
relay_files admission.json || die "controller evidence relay admission failed"

export REPO_HOST="$repo_host"
export REPO_CONTAINER=/workspace/verl
export DOCKER_IMAGE="$DYNPERM_IMAGE_ID"
export DOCKER_CONTAINER_NAME="$container_name"
export DATA1_HOST="$data1_host"
export DATA2_HOST="$data2_host"
export REPO_MOUNT_MODE=ro
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=offline

bash "$repo_host/scripts/l40s/run_train.sh" \
    bash "recipe/on_policy_wdl_sft/math_task/${arm_entry}" \
    >"${job_root}/stdout.log" 2>"${job_root}/stderr.log" &
training_pid=$!

run_prefix="MATH-WDL-DYNPERM-${dose_tag^^}-${arm_id^^}-P60-QWEN3-1P7B"
python3 "$repo_host/scripts/math_wdl_first_step_gate.py" \
    --metrics-root "${artifact_root}/logs/metrics" \
    --project OnPolicyWDLSFT-Math-1P7B-DynPerm-P60 \
    --run-prefix "$run_prefix" \
    --expected-model1-gradient "$expected_gradient" \
    --dynperm-rho "$DYNPERM_RHO" \
    --slurm-job-id "$SLURM_JOB_ID" \
    --timeout-seconds 43200 \
    --receipt "${job_root}/first-step.json" \
    >"${job_root}/first-step.log" 2>&1 &
gate_pid=$!

while kill -0 "$training_pid" >/dev/null 2>&1 \
    && kill -0 "$gate_pid" >/dev/null 2>&1; do
    sleep 5
done

if ! kill -0 "$training_pid" >/dev/null 2>&1; then
    set +e
    wait "$training_pid"
    training_rc=$?
    set -e
    training_pid=""
    kill "$gate_pid" >/dev/null 2>&1 || true
    wait "$gate_pid" >/dev/null 2>&1 || true
    gate_pid=""
    echo "training exited before first-step admission completed" >&2
    if [ "$training_rc" -eq 0 ]; then
        exit 66
    fi
    exit "$training_rc"
fi

if ! wait "$gate_pid"; then
    gate_pid=""
    echo "first-step admission failed; stopping only this job's training container" >&2
    kill "$training_pid" >/dev/null 2>&1 || true
    wait "$training_pid" >/dev/null 2>&1 || true
    training_pid=""
    exit 65
fi
gate_pid=""
if ! relay_files first-step.json first-step.log; then
    echo "first-step evidence relay failed; stopping only this job's training container" >&2
    kill "$training_pid" >/dev/null 2>&1 || true
    wait "$training_pid" >/dev/null 2>&1 || true
    training_pid=""
    exit 70
fi

wait "$training_pid"
training_pid=""
