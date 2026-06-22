#!/usr/bin/env bash
# Thin code-task queue monitor; delegates generic logic.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export MONITOR_NAME=${MONITOR_NAME:-code_task_queue}
export CKPT_ROOT=${CKPT_ROOT:-/data-1/checkpoints}
export METRICS_ROOT=${METRICS_ROOT:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/metrics}
export EXTRA_METRICS_ROOTS=${EXTRA_METRICS_ROOTS:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/staged_v1/metrics}
export WANDB_PROJECT=${WANDB_PROJECT:-OnPolicyWDLSFT-CodeTask}
export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/monitor_code_task_queue_notify.log}
export POLL_SEC=${POLL_SEC:-300}
export WXPUSHER_NOTIFY=${WXPUSHER_NOTIFY:-1}
export WXPUSHER_SCRIPT=${WXPUSHER_SCRIPT:-/root/agent-core/skills/wxpusher-notify/scripts/wxpusher_notify.py}

QUEUE_MODE=${QUEUE_MODE:-smoke}
if [ "$QUEUE_MODE" = "full" ]; then
    export QUEUE_TMUX=${QUEUE_TMUX:-code_task_full_queue}
    RUN_PREFIXES=(
        "ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V2"
        "ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V2"
    )
    TMUX_NAMES=("code_task_s1_kodcode_beta0" "code_task_s1_kodcode_beta01")
    FINAL_STEPS=(150 150)
elif [ "$QUEUE_MODE" = "retention" ]; then
    export QUEUE_TMUX=${QUEUE_TMUX:-code_task_retention_queue}
    RUN_PREFIXES=(
        "${RETENTION_BETA0_PREFIX:-ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V2-RETENTION}"
        "${RETENTION_BETA01_PREFIX:-ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V2-RETENTION}"
    )
    TMUX_NAMES=("code_task_s1_kodcode_beta0_retention" "code_task_s1_kodcode_beta01_retention")
    FINAL_STEPS=(150 150)
elif [ "$QUEUE_MODE" = "stage2_retention" ]; then
    export QUEUE_TMUX=${QUEUE_TMUX:-code_task_stage2_retention_queue}
    STAGE2_TOTAL_TRAINING_STEPS=${STAGE2_TOTAL_TRAINING_STEPS:-40}
    RUN_PREFIXES=(
        "${STAGE2_BETA0_PREFIX:-CODE-S2-RETENTION-BETA0-BETA0}"
        "${STAGE2_BETA01_PREFIX:-CODE-S2-RETENTION-BETA01-BETA01}"
    )
    TMUX_NAMES=("code_task_s2_retention_beta0_beta0" "code_task_s2_retention_beta01_beta01")
    FINAL_STEPS=("$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS")
elif [ "$QUEUE_MODE" = "deepcoder_stage1" ]; then
    if [ "$MONITOR_NAME" = "code_task_queue" ]; then
        export MONITOR_NAME=code_task_deepcoder_stage1
    fi
    export QUEUE_TMUX=${QUEUE_TMUX:-code_task_deepcoder_stage1_queue}
    export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/monitor_code_task_deepcoder_stage1_notify.log}
    RUN_PREFIXES=(
        "ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA0-V1-RETENTION"
        "ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA01-V1-RETENTION"
    )
    TMUX_NAMES=("code_task_s1_deepcoder_beta0" "code_task_s1_deepcoder_beta01")
    FINAL_STEPS=(150 150)
elif [ "$QUEUE_MODE" = "pilot" ]; then
    export QUEUE_TMUX=${QUEUE_TMUX:-code_task_pilot_queue}
    RUN_PREFIXES=("CODE-S1-PILOT-BETA0" "CODE-S2-PILOT-BETA0-BETA0")
    TMUX_NAMES=("code_task_s1_pilot_beta0" "code_task_s2_pilot_beta0_beta0")
    FINAL_STEPS=(40 20)
else
    export QUEUE_TMUX=${QUEUE_TMUX:-code_task_smoke_queue}
    RUN_PREFIXES=("CODE-S1-SMOKE-BETA0" "CODE-S2-SMOKE-BETA0-BETA0")
    TMUX_NAMES=("code_task_s1_smoke_beta0" "code_task_s2_smoke_beta0_beta0")
    FINAL_STEPS=(5 5)
fi

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/training_queue_monitor.sh"
training_queue_monitor_main
