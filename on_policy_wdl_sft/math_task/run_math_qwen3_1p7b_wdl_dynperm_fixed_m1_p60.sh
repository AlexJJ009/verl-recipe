#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Public fixed-Model1 C entry. Scientific treatment knobs:
# DYNPERM_ENABLED and DYNPERM_RHO only.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -ne 0 ]; then
    echo "ERROR: fixed-Model1 DynPerm P60 accepts no positional config overrides" >&2
    exit 1
fi
exec bash "${SCRIPT_DIR}/run_math_qwen3_1p7b_wdl_dynperm_p60_arm_common.sh" fixed-m1-stage1
