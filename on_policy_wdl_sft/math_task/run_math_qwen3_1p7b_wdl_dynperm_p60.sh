#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Local two-arm queue. The Slurm matrix uses the arm-specific entry points so
# all three nodes can work concurrently.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -ne 0 ]; then
    echo "ERROR: DynPerm P60 accepts no positional config overrides" >&2
    exit 1
fi
: "${DYNPERM_ENABLED:?set DYNPERM_ENABLED=true}"
: "${DYNPERM_RHO:?set DYNPERM_RHO in [0, 1]}"

# Fixed-Model1 is intentionally first. Model1 update state remains owned by
# the two static arm wrappers and is not a third DynPerm treatment variable.
bash "${SCRIPT_DIR}/run_math_qwen3_1p7b_wdl_dynperm_fixed_m1_p60.sh"
bash "${SCRIPT_DIR}/run_math_qwen3_1p7b_wdl_dynperm_standard_c_p60.sh"
