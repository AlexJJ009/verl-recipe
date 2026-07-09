#!/usr/bin/env bash
# Sync the completed/interrupted staged-v1 W&B offline runs to the shared project.

set -euo pipefail

WANDB_PROJECT=${WANDB_PROJECT:-OnPolicySFT-Then-WDLSFT-StagedV1}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

runs=(
  /data-1/wandb_runs/ONPOLICY-SFT-Qwen3-4B-MATH-S1-BASE-V1/wandb/offline-run-20260528_071432-7utidal6
  /data-1/wandb_runs/ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA0-V1/wandb/offline-run-20260528_101005-tlg5b94n
  /data-1/wandb_runs/ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA01-V1/wandb/offline-run-20260528_151816-4qjnywm0
  /data-1/wandb_runs/ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA02-V1/wandb/offline-run-20260528_202817-dqbd7zq3
  /data-1/wandb_runs/ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA03-V1/wandb/offline-run-20260529_014831-9oicims3
  /data-1/wandb_runs/ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA04-V1/wandb/offline-run-20260529_071742-z5987ls0
  /data-1/wandb_runs/WDL-SFT-STAGED-V1-S2-FROM-S1-BETA0-BETA0/wandb/offline-run-20260529_164940-j2qmgtqi
  /data-1/wandb_runs/WDL-SFT-STAGED-V1-S2-FROM-S1-BETA01-BETA01/wandb/offline-run-20260529_231457-0ywgguty
)
run_names=(
  ONPOLICY-SFT-Qwen3-4B-MATH-S1-BASE-V1_1779952267
  ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA0-V1_1779962803
  ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA01-V1_1779981295
  ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA02-V1_1779999847
  ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA03-V1_1780019059
  ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA04-V1_1780038811
  WDL-SFT-STAGED-V1-S2-FROM-S1-BETA0-BETA0_1780073162
  WDL-SFT-STAGED-V1-S2-FROM-S1-BETA01-BETA01_1780096269
)

echo "[staged_v1/wandb] start $(date -Is)"
echo "[staged_v1/wandb] project=${WANDB_PROJECT}"

for idx in "${!runs[@]}"; do
  run_dir="${runs[$idx]}"
  run_name="${run_names[$idx]}"
  if [[ ! -d "${run_dir}" ]]; then
    echo "[staged_v1/wandb] ERROR missing run dir: ${run_dir}" >&2
    exit 2
  fi
  echo "[staged_v1/wandb] syncing ${run_name} ${run_dir}"
  RELEASE_GATE_RUN_NAME="${run_name}" WANDB_PROJECT="${WANDB_PROJECT}" MARK_SYNCED=true bash "${SCRIPT_DIR}/sync_wandb_offline.sh" "${run_dir}"
  echo "[staged_v1/wandb] synced ${run_dir}"
done

echo "[staged_v1/wandb] complete $(date -Is)"
