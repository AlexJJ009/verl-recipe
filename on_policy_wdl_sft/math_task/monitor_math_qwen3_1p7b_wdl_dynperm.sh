#!/usr/bin/env bash
set -euo pipefail

REPO_HOST=${REPO_HOST:-/data-1/code/verl}
QUEUE_TMUX=${QUEUE_TMUX:-math-wdl-dynperm-p60}
: "${DYNPERM_RHO:?monitor requires the same DYNPERM_RHO as the P60 launcher}"
POLL_SEC=${POLL_SEC:-60}
DYNPERM_DOSE_TAG="$(python3 - "$DYNPERM_RHO" <<'PY'
import math
import sys

rho = float(sys.argv[1])
if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
    raise SystemExit("ERROR: DYNPERM_RHO must be finite and in [0, 1]")
canonical = format(rho, ".12g")
print(f"rho{canonical.replace('.', 'p')}")
PY
)"

WANDB_PROJECT=OnPolicyWDLSFT-Math-1P7B-DynPerm-P60
ARMS=(fixed-m1-stage1 standard-c)

for arm in "${ARMS[@]}"; do
    expected_gradient=nonzero
    [ "$arm" = fixed-m1-stage1 ] && expected_gradient=zero
    arm_upper=${arm^^}
    RUN_PREFIX="MATH-WDL-DYNPERM-${DYNPERM_DOSE_TAG^^}-${arm_upper}-P60-QWEN3-1P7B"
    ARTIFACT_ROOT="/data-2/model_weights/math_task/qwen3_1p7b_wdl_dynperm/${DYNPERM_DOSE_TAG}/${arm}-p60"
    mkdir -p "$ARTIFACT_ROOT/admission"
    python3 "${REPO_HOST}/scripts/math_wdl_first_step_gate.py" \
        --metrics-root "$ARTIFACT_ROOT/logs/metrics" \
        --project "$WANDB_PROJECT" \
        --run-prefix "$RUN_PREFIX" \
        --expected-model1-gradient "$expected_gradient" \
        --dynperm-rho "$DYNPERM_RHO" \
        --queue-tmux "$QUEUE_TMUX" \
        --timeout-seconds "${FIRST_STEP_TIMEOUT_SECONDS:-43200}" \
        --receipt "$ARTIFACT_ROOT/admission/first_step_${RUN_PREFIX}_$(date +%s).json"
done

while tmux has-session -t "$QUEUE_TMUX" 2>/dev/null; do
    echo "DynPerm P60 first-step gates passed; queue still active. No unowned tmux/Slurm job will be mutated."
    sleep "$POLL_SEC"
done
