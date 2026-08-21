#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=160
#SBATCH --gres=gpu:L40S:8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
dispatch_job_id="${SLURM_JOB_ID:-invalid-$$}"
[[ "$dispatch_job_id" =~ ^[0-9]+$ ]] || dispatch_job_id="invalid-$$"
dispatch_relay_root="/data-1/code/_artifacts/verl-v0.7/dynperm-formal-p60/bootstrap/${dispatch_job_id}"
dispatch_cleanup() {
    local original_rc=$? final_rc receipt local_sha remote_sha
    final_rc=$original_rc
    trap - EXIT TERM INT
    set +e
    receipt="$(printf '{"job_id":"%s","phase":"sbatch-dispatch","original_exit_code":%s}\n' "$dispatch_job_id" "$original_rc")"
    if [[ "${DYNPERM_EVIDENCE_RELAY_HOST:-}" =~ ^[A-Za-z0-9._@:-]+$ ]]; then
        local_sha="$(printf '%s' "$receipt" | sha256sum | awk '{print $1}')"
        printf '%s' "$receipt" | ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
            "$DYNPERM_EVIDENCE_RELAY_HOST" \
            "mkdir -p -- '${dispatch_relay_root}' && cat > '${dispatch_relay_root}/dispatch-terminal.json'"
        remote_sha="$(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
            "$DYNPERM_EVIDENCE_RELAY_HOST" sha256sum -- \
            "${dispatch_relay_root}/dispatch-terminal.json" 2>/dev/null | awk '{print $1}')"
        [ -n "$local_sha" ] && [ "$remote_sha" = "$local_sha" ] || final_rc=70
    else
        final_rc=70
    fi
    exit "$final_rc"
}
trap dispatch_cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
: "${DYNPERM_NODE_ROOT_MAP:?node-root map required}"
: "${DYNPERM_STAGE_REL:?candidate stage path required}"
node_root=""
IFS=';' read -r -a node_roots <<<"$DYNPERM_NODE_ROOT_MAP"
for entry in "${node_roots[@]}"; do
    if [ "${entry%%=*}" = "${SLURMD_NODENAME:?}" ]; then
        node_root="${entry#*=}"
        break
    fi
done
[ -n "$node_root" ]
[[ "$DYNPERM_STAGE_REL" =~ ^workspace/jobs/[A-Za-z0-9._-]+$ ]]
node_root="$(realpath -e "$node_root")"
[ "$node_root" != / ]
workspace="$(realpath -e "${node_root}/${DYNPERM_STAGE_REL}")"
case "$workspace" in "$node_root"/workspace/jobs/*) ;; *) exit 64 ;; esac
trap - EXIT TERM INT
exec bash "${workspace}/repo/recipe/on_policy_wdl_sft/math_task/slurm/run_math_qwen3_1p7b_wdl_dynperm_p60_job.sh" fixed-m1-stage1
