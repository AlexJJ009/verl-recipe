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
exec bash "${workspace}/repo/recipe/on_policy_wdl_sft/math_task/slurm/run_math_qwen3_1p7b_wdl_dynperm_p60_job.sh" standard-c
