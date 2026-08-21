#!/usr/bin/env bash
set -euo pipefail

REPO_HOST=${REPO_HOST:-/data-1/code/verl}
QUEUE_TMUX=${QUEUE_TMUX:-math-wdl-dynperm}
DYNPERM_RHO=${DYNPERM_RHO:-1.0}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-20}
case "$DYNPERM_RHO" in 0|0.0) DOSE=RHO000 ;; 1|1.0) DOSE=RHO100 ;; *) exit 2 ;; esac
case "$TOTAL_TRAINING_STEPS" in 20|30|60) ;; *) exit 2 ;; esac

RUN_PREFIX="MATH-WDL-DYNPERM-${DOSE}-P${TOTAL_TRAINING_STEPS}-QWEN3-1P7B"
WANDB_PROJECT=OnPolicyWDLSFT-Math-1P7B-DynPerm
ARTIFACT_ROOT="/data-2/model_weights/math_task/qwen3_1p7b_wdl_dynperm/${DOSE,,}-p${TOTAL_TRAINING_STEPS}"
METRICS_ROOT=${METRICS_ROOT:-$ARTIFACT_ROOT/logs/metrics}
VALIDATION_ROOT=${VALIDATION_ROOT:-$ARTIFACT_ROOT/logs/validation}
ADMISSION_ROOT=${ADMISSION_ROOT:-$ARTIFACT_ROOT/admission}
FORMAT_GATE_LOG=${FORMAT_GATE_LOG:-$ARTIFACT_ROOT/format_gate.log}
POLL_SEC=${POLL_SEC:-60}

mkdir -p "$ADMISSION_ROOT"
python3 "${REPO_HOST}/scripts/math_wdl_first_step_gate.py" \
    --metrics-root "$METRICS_ROOT" \
    --project "$WANDB_PROJECT" \
    --run-prefix "$RUN_PREFIX" \
    --expected-model1-gradient nonzero \
    --dynperm-rho "$DYNPERM_RHO" \
    --queue-tmux "$QUEUE_TMUX" \
    --timeout-seconds "${FIRST_STEP_TIMEOUT_SECONDS:-21600}" \
    --receipt "$ADMISSION_ROOT/first_step_${RUN_PREFIX}_$(date +%s).json"

while tmux has-session -t "$QUEUE_TMUX" 2>/dev/null; do
    set +e
    python3 "${REPO_HOST}/scripts/math_wdl_format_gate.py" \
        --validation-root "$VALIDATION_ROOT" \
        --run-prefix "$RUN_PREFIX" \
        --max-drop 0.05 \
        --required-consecutive 2 >>"$FORMAT_GATE_LOG" 2>&1
    gate_rc=$?
    set -e
    if [ "$gate_rc" -ne 0 ]; then
        echo "ERROR: DynPerm format gate failed rc=${gate_rc}; monitor will not mutate an unowned tmux/Slurm job" >&2
        exit "$gate_rc"
    fi
    sleep "$POLL_SEC"
done
