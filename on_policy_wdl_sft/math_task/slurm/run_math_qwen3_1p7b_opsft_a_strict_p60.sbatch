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
: "${STRICT_A_NODE_ROOT_MAP:?}"
: "${STRICT_A_STAGE_REL:?}"
[[ "$STRICT_A_STAGE_REL" =~ ^workspace/jobs/[A-Za-z0-9._-]+$ ]]
node_root=""
IFS=';' read -r -a node_roots <<<"$STRICT_A_NODE_ROOT_MAP"
for entry in "${node_roots[@]}"; do
    if [ "${entry%%=*}" = "${SLURMD_NODENAME:?}" ]; then node_root="${entry#*=}"; break; fi
done
[ -n "$node_root" ]
node_root="$(realpath -e "$node_root")"
[ "$node_root" != / ]
workspace="$(realpath -e "${node_root}/${STRICT_A_STAGE_REL}")"
case "$workspace" in "$node_root"/workspace/jobs/*) ;; *) exit 64 ;; esac
job_body="${workspace}/repo/recipe/on_policy_wdl_sft/math_task/slurm/run_math_qwen3_1p7b_opsft_a_strict_p60_job.sh"
test -r "$job_body"
exec bash "$job_body"
