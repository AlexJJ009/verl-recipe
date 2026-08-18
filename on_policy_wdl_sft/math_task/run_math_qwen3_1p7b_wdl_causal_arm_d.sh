#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_PREFIX=${RUN_PREFIX:-MATH-WDL-CAUSAL-P60-ARM-D-QWEN3-1P7B}
export FUSION_LAMBDA=1.0
export FUSION_MODE=mixture
export MODEL_PATH=${MODEL_PATH:-/data-1/.cache/huggingface/math-wdl-causal-p60-arm-d}

# The already-running queue may have loaded its D0 -> D -> C order before the
# 2026-07-27 amendment. Keep the omission gate in this wrapper so that both that
# live process and future direct calls skip D unless it is explicitly requested.
if [ "${RUN_OPTIONAL_D:-0}" != "1" ]; then
    receipt=${WDL_MANIPULATION_RECEIPT:-/data-2/model_weights/math_task/qwen3_1p7b_wdl_causal_p60/admission/manipulation_receipt.json}
    python3 - "$receipt" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"ERROR: cannot omit optional D without receipt: {path}")
receipt = json.loads(path.read_text())
required = (
    "D_is_direct_model2",
    "D_ignores_and_does_not_update_model1",
    "D_and_D0_are_model1_invariant",
)
missing = [name for name in required if receipt.get("checks", {}).get(name) is not True]
if receipt.get("status") != "pass" or missing:
    raise SystemExit(f"ERROR: cannot omit optional D; failed equivalence checks: {missing}")
print(
    "OPTIONAL ARM D OMITTED: direct-Model2 equivalence probe passed; "
    "reuse historical Stage1 control A. Set RUN_OPTIONAL_D=1 to run it explicitly."
)
PY
    exit 0
fi
exec bash "${SCRIPT_DIR}/run_math_qwen3_1p7b_wdl_causal_p60_common.sh" "$@"
