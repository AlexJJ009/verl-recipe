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
export TRAINING_RELEASE_SUCCESS_HOOK=${TRAINING_RELEASE_SUCCESS_HOOK:-${REPO_ROOT}/scripts/code_task_training_release_hook.sh}

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
elif [ "$QUEUE_MODE" = "kodcode_instruct2507_ctx8k_stage2_p60" ]; then
    if [ "$MONITOR_NAME" = "code_task_queue" ]; then
        export MONITOR_NAME=code_task_kodcode_instruct2507_ctx8k_stage2_p60
    fi
    export QUEUE_TMUX=${QUEUE_TMUX:-code_task_kodcode_i2507_ctx8k_s2_p60_queue}
    export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/monitor_code_task_kodcode_instruct2507_ctx8k_stage2_p60_notify.log}
    STAGE2_TOTAL_TRAINING_STEPS=${STAGE2_TOTAL_TRAINING_STEPS:-40}
    RUN_PREFIXES=(
        "${STAGE2_BETA0_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P60-BETA0-BETA0-V1}"
        "${STAGE2_BETA01_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P60-BETA01-BETA01-V1}"
    )
    TMUX_NAMES=("code_task_s2_kodcode_i2507_ctx8k_p60_beta0_beta0" "code_task_s2_kodcode_i2507_ctx8k_p60_beta01_beta01")
    FINAL_STEPS=("$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS")
elif [ "$QUEUE_MODE" = "kodcode_instruct2507_ctx8k_stage2_p40" ]; then
    if [ "$MONITOR_NAME" = "code_task_queue" ]; then
        export MONITOR_NAME=code_task_kodcode_instruct2507_ctx8k_stage2_p40
    fi
    export QUEUE_TMUX=${QUEUE_TMUX:-code_task_kodcode_i2507_ctx8k_s2_p40_queue}
    export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/monitor_code_task_kodcode_instruct2507_ctx8k_stage2_p40_notify.log}
    STAGE2_TOTAL_TRAINING_STEPS=${STAGE2_TOTAL_TRAINING_STEPS:-40}
    RUN_PREFIXES=(
        "${STAGE2_BETA0_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA0-BETA0-V1}"
        "${STAGE2_BETA01_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-V1}"
    )
    TMUX_NAMES=("code_task_s2_kodcode_i2507_ctx8k_p40_beta0_beta0" "code_task_s2_kodcode_i2507_ctx8k_p40_beta01_beta01")
    FINAL_STEPS=("$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS")
elif [ "$QUEUE_MODE" = "kodcode_instruct2507_ctx8k_stage2_lambda_sweep" ]; then
    if [ "$MONITOR_NAME" = "code_task_queue" ]; then
        export MONITOR_NAME=code_task_kodcode_instruct2507_ctx8k_stage2_lambda_sweep
    fi
    export QUEUE_TMUX=${QUEUE_TMUX:-code_task_kodcode_i2507_ctx8k_s2_lambda_sweep_queue}
    export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/monitor_code_task_kodcode_instruct2507_ctx8k_stage2_lambda_sweep_notify.log}
    STAGE2_TOTAL_TRAINING_STEPS=${STAGE2_TOTAL_TRAINING_STEPS:-60}
    RUN_PREFIXES=(
        "${STAGE2_P40_BETA01_LAMBDA06_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-LAMBDA06-V1}"
        "${STAGE2_P40_BETA01_LAMBDA07_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-LAMBDA07-V1}"
        "${STAGE2_P40_BETA01_LAMBDA08_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-LAMBDA08-V1}"
        "${STAGE2_P40_BETA01_LAMBDA09_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-LAMBDA09-V1}"
        "${STAGE2_P40_BETA01_LAMBDA10_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-LAMBDA10-V1}"
    )
    TMUX_NAMES=(
        "code_task_s2_kodcode_i2507_ctx8k_p40_beta01_lambda06"
        "code_task_s2_kodcode_i2507_ctx8k_p40_beta01_lambda07"
        "code_task_s2_kodcode_i2507_ctx8k_p40_beta01_lambda08"
        "code_task_s2_kodcode_i2507_ctx8k_p40_beta01_lambda09"
        "code_task_s2_kodcode_i2507_ctx8k_p40_beta01_lambda10"
    )
    FINAL_STEPS=("$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS")
elif [ "$QUEUE_MODE" = "kodcode_instruct2507_ctx8k_stage2_p40_subkl_lambda_sweep" ]; then
    if [ "$MONITOR_NAME" = "code_task_queue" ]; then
        export MONITOR_NAME=code_task_kodcode_instruct2507_ctx8k_stage2_p40_subkl_lambda_sweep
    fi
    export QUEUE_TMUX=${QUEUE_TMUX:-code_task_kodcode_i2507_ctx8k_s2_p40_subkl_lambda_queue}
    export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/monitor_code_task_kodcode_instruct2507_ctx8k_stage2_p40_subkl_lambda_sweep_notify.log}
    STAGE2_TOTAL_TRAINING_STEPS=${STAGE2_TOTAL_TRAINING_STEPS:-60}
    RUN_PREFIXES=(
        "${STAGE2_P40_BETA01_SUBKL_M1_LAMBDA05_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-SUBKL-M1-LAMBDA05-V1}"
        "${STAGE2_P40_BETA01_SUBKL_M1_LAMBDA06_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-SUBKL-M1-LAMBDA06-V1}"
        "${STAGE2_P40_BETA01_SUBKL_M1_LAMBDA07_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-SUBKL-M1-LAMBDA07-V1}"
        "${STAGE2_P40_BETA01_SUBKL_M1_LAMBDA08_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-SUBKL-M1-LAMBDA08-V1}"
        "${STAGE2_P40_BETA01_SUBKL_M1_LAMBDA09_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-SUBKL-M1-LAMBDA09-V1}"
        "${STAGE2_P40_BETA01_SUBKL_M2_LAMBDA05_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-SUBKL-M2-LAMBDA05-V1}"
        "${STAGE2_P40_BETA01_SUBKL_M2_LAMBDA06_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-SUBKL-M2-LAMBDA06-V1}"
        "${STAGE2_P40_BETA01_SUBKL_M2_LAMBDA07_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-SUBKL-M2-LAMBDA07-V1}"
        "${STAGE2_P40_BETA01_SUBKL_M2_LAMBDA08_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-SUBKL-M2-LAMBDA08-V1}"
        "${STAGE2_P40_BETA01_SUBKL_M2_LAMBDA09_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-SUBKL-M2-LAMBDA09-V1}"
        "${STAGE2_P40_BETA01_SUBKL_BOTH_LAMBDA05_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-SUBKL-BOTH-LAMBDA05-V1}"
        "${STAGE2_P40_BETA01_SUBKL_BOTH_LAMBDA06_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-SUBKL-BOTH-LAMBDA06-V1}"
        "${STAGE2_P40_BETA01_SUBKL_BOTH_LAMBDA07_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-SUBKL-BOTH-LAMBDA07-V1}"
        "${STAGE2_P40_BETA01_SUBKL_BOTH_LAMBDA08_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-SUBKL-BOTH-LAMBDA08-V1}"
        "${STAGE2_P40_BETA01_SUBKL_BOTH_LAMBDA09_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-SUBKL-BOTH-LAMBDA09-V1}"
    )
    TMUX_NAMES=(
        "code_task_s2_p40_subkl_m1_lambda05"
        "code_task_s2_p40_subkl_m1_lambda06"
        "code_task_s2_p40_subkl_m1_lambda07"
        "code_task_s2_p40_subkl_m1_lambda08"
        "code_task_s2_p40_subkl_m1_lambda09"
        "code_task_s2_p40_subkl_m2_lambda05"
        "code_task_s2_p40_subkl_m2_lambda06"
        "code_task_s2_p40_subkl_m2_lambda07"
        "code_task_s2_p40_subkl_m2_lambda08"
        "code_task_s2_p40_subkl_m2_lambda09"
        "code_task_s2_p40_subkl_both_lambda05"
        "code_task_s2_p40_subkl_both_lambda06"
        "code_task_s2_p40_subkl_both_lambda07"
        "code_task_s2_p40_subkl_both_lambda08"
        "code_task_s2_p40_subkl_both_lambda09"
    )
    FINAL_STEPS=(
        "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS"
        "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS"
        "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS"
    )
elif [ "$QUEUE_MODE" = "kodcode_qwen3_1p7b_instruct_ctx8k_stage2_p40_m2kl_vs_nokl_lambda_sweep" ]; then
    if [ "$MONITOR_NAME" = "code_task_queue" ]; then
        export MONITOR_NAME=code_task_kodcode_qwen3_1p7b_instruct_ctx8k_stage2_p40_m2kl_vs_nokl_lambda_sweep
    fi
    export QUEUE_TMUX=${QUEUE_TMUX:-code_task_kodcode_qwen3_1p7b_s2_p40_m2kl_vs_nokl_queue}
    export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/monitor_code_task_kodcode_qwen3_1p7b_instruct_ctx8k_stage2_p40_m2kl_vs_nokl_lambda_sweep_notify.log}
    STAGE2_TOTAL_TRAINING_STEPS=${STAGE2_TOTAL_TRAINING_STEPS:-60}
    RUN_PREFIXES=(
        "${QWEN3_1P7B_S2_BETA0_NOKL_LAMBDA05_PREFIX:-CODE-S2-KODCODE-Qwen3-1P7B-INSTRUCT-CTX8K-P40-BETA0-BETA0-LAMBDA05-NOKL-V1}"
        "${QWEN3_1P7B_S2_BETA0_NOKL_LAMBDA06_PREFIX:-CODE-S2-KODCODE-Qwen3-1P7B-INSTRUCT-CTX8K-P40-BETA0-BETA0-LAMBDA06-NOKL-V1}"
        "${QWEN3_1P7B_S2_BETA0_NOKL_LAMBDA07_PREFIX:-CODE-S2-KODCODE-Qwen3-1P7B-INSTRUCT-CTX8K-P40-BETA0-BETA0-LAMBDA07-NOKL-V1}"
        "${QWEN3_1P7B_S2_BETA0_NOKL_LAMBDA08_PREFIX:-CODE-S2-KODCODE-Qwen3-1P7B-INSTRUCT-CTX8K-P40-BETA0-BETA0-LAMBDA08-NOKL-V1}"
        "${QWEN3_1P7B_S2_BETA0_NOKL_LAMBDA09_PREFIX:-CODE-S2-KODCODE-Qwen3-1P7B-INSTRUCT-CTX8K-P40-BETA0-BETA0-LAMBDA09-NOKL-V1}"
        "${QWEN3_1P7B_S2_BETA0_M2KL_LAMBDA05_PREFIX:-CODE-S2-KODCODE-Qwen3-1P7B-INSTRUCT-CTX8K-P40-BETA0-BETA0-LAMBDA05-M2KL-V1}"
        "${QWEN3_1P7B_S2_BETA0_M2KL_LAMBDA06_PREFIX:-CODE-S2-KODCODE-Qwen3-1P7B-INSTRUCT-CTX8K-P40-BETA0-BETA0-LAMBDA06-M2KL-V1}"
        "${QWEN3_1P7B_S2_BETA0_M2KL_LAMBDA07_PREFIX:-CODE-S2-KODCODE-Qwen3-1P7B-INSTRUCT-CTX8K-P40-BETA0-BETA0-LAMBDA07-M2KL-V1}"
        "${QWEN3_1P7B_S2_BETA0_M2KL_LAMBDA08_PREFIX:-CODE-S2-KODCODE-Qwen3-1P7B-INSTRUCT-CTX8K-P40-BETA0-BETA0-LAMBDA08-M2KL-V1}"
        "${QWEN3_1P7B_S2_BETA0_M2KL_LAMBDA09_PREFIX:-CODE-S2-KODCODE-Qwen3-1P7B-INSTRUCT-CTX8K-P40-BETA0-BETA0-LAMBDA09-M2KL-V1}"
        "${QWEN3_1P7B_S2_BETA01_NOKL_LAMBDA05_PREFIX:-CODE-S2-KODCODE-Qwen3-1P7B-INSTRUCT-CTX8K-P40-BETA01-BETA01-LAMBDA05-NOKL-V1}"
        "${QWEN3_1P7B_S2_BETA01_NOKL_LAMBDA06_PREFIX:-CODE-S2-KODCODE-Qwen3-1P7B-INSTRUCT-CTX8K-P40-BETA01-BETA01-LAMBDA06-NOKL-V1}"
        "${QWEN3_1P7B_S2_BETA01_NOKL_LAMBDA07_PREFIX:-CODE-S2-KODCODE-Qwen3-1P7B-INSTRUCT-CTX8K-P40-BETA01-BETA01-LAMBDA07-NOKL-V1}"
        "${QWEN3_1P7B_S2_BETA01_NOKL_LAMBDA08_PREFIX:-CODE-S2-KODCODE-Qwen3-1P7B-INSTRUCT-CTX8K-P40-BETA01-BETA01-LAMBDA08-NOKL-V1}"
        "${QWEN3_1P7B_S2_BETA01_NOKL_LAMBDA09_PREFIX:-CODE-S2-KODCODE-Qwen3-1P7B-INSTRUCT-CTX8K-P40-BETA01-BETA01-LAMBDA09-NOKL-V1}"
        "${QWEN3_1P7B_S2_BETA01_M2KL_LAMBDA05_PREFIX:-CODE-S2-KODCODE-Qwen3-1P7B-INSTRUCT-CTX8K-P40-BETA01-BETA01-LAMBDA05-M2KL-V1}"
        "${QWEN3_1P7B_S2_BETA01_M2KL_LAMBDA06_PREFIX:-CODE-S2-KODCODE-Qwen3-1P7B-INSTRUCT-CTX8K-P40-BETA01-BETA01-LAMBDA06-M2KL-V1}"
        "${QWEN3_1P7B_S2_BETA01_M2KL_LAMBDA07_PREFIX:-CODE-S2-KODCODE-Qwen3-1P7B-INSTRUCT-CTX8K-P40-BETA01-BETA01-LAMBDA07-M2KL-V1}"
        "${QWEN3_1P7B_S2_BETA01_M2KL_LAMBDA08_PREFIX:-CODE-S2-KODCODE-Qwen3-1P7B-INSTRUCT-CTX8K-P40-BETA01-BETA01-LAMBDA08-M2KL-V1}"
        "${QWEN3_1P7B_S2_BETA01_M2KL_LAMBDA09_PREFIX:-CODE-S2-KODCODE-Qwen3-1P7B-INSTRUCT-CTX8K-P40-BETA01-BETA01-LAMBDA09-M2KL-V1}"
    )
    TMUX_NAMES=(
        "code_task_s2_q17b_b0_nokl_l05"
        "code_task_s2_q17b_b0_nokl_l06"
        "code_task_s2_q17b_b0_nokl_l07"
        "code_task_s2_q17b_b0_nokl_l08"
        "code_task_s2_q17b_b0_nokl_l09"
        "code_task_s2_q17b_b0_m2kl_l05"
        "code_task_s2_q17b_b0_m2kl_l06"
        "code_task_s2_q17b_b0_m2kl_l07"
        "code_task_s2_q17b_b0_m2kl_l08"
        "code_task_s2_q17b_b0_m2kl_l09"
        "code_task_s2_q17b_b01_nokl_l05"
        "code_task_s2_q17b_b01_nokl_l06"
        "code_task_s2_q17b_b01_nokl_l07"
        "code_task_s2_q17b_b01_nokl_l08"
        "code_task_s2_q17b_b01_nokl_l09"
        "code_task_s2_q17b_b01_m2kl_l05"
        "code_task_s2_q17b_b01_m2kl_l06"
        "code_task_s2_q17b_b01_m2kl_l07"
        "code_task_s2_q17b_b01_m2kl_l08"
        "code_task_s2_q17b_b01_m2kl_l09"
    )
    FINAL_STEPS=(
        "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS"
        "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS"
        "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS"
        "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS"
    )
elif [ "$QUEUE_MODE" = "kodcode_qwen3_1p7b_coldstart_ctx8k_stage2_p40_m2kl_vs_nokl" ]; then
    if [ "$MONITOR_NAME" = "code_task_queue" ]; then
        export MONITOR_NAME=code_task_kodcode_qwen3_1p7b_coldstart_ctx8k_stage2_p40_m2kl_vs_nokl
    fi
    export QUEUE_TMUX=${QUEUE_TMUX:-code_task_kodcode_qwen3_1p7b_coldstart_s2_p40_m2kl_vs_nokl_queue}
    export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/monitor_code_task_kodcode_qwen3_1p7b_instruct_ctx8k_coldstart_stage2_p40_m2kl_vs_nokl_notify.log}
    STAGE2_TOTAL_TRAINING_STEPS=${STAGE2_TOTAL_TRAINING_STEPS:-60}
    COLDSTART_STAGE2_FUSION_LAMBDAS=${COLDSTART_STAGE2_FUSION_LAMBDAS:-0.8}
    RUN_PREFIXES=()
    TMUX_NAMES=()
    FINAL_STEPS=()
    for beta_tag in beta0 beta01; do
        beta_prefix_tag=BETA0
        [ "$beta_tag" = "beta01" ] && beta_prefix_tag=BETA01
        for kl_mode in nokl model2; do
            for fusion_lambda in $COLDSTART_STAGE2_FUSION_LAMBDAS; do
                lambda_tag="lambda$(printf '%s' "$fusion_lambda" | tr -d '.')"
                kl_suffix=NOKL
                [ "$kl_mode" = "model2" ] && kl_suffix=M2KL
                RUN_PREFIXES+=("CODE-S2-KODCODE-Qwen3-1P7B-COLDSTART-CTX8K-P40-${beta_prefix_tag}-${beta_prefix_tag}-${lambda_tag^^}-${kl_suffix}-V1")
                TMUX_NAMES+=("code_task_s2_q17b_cold_${beta_tag}_${kl_mode}_${lambda_tag}")
                FINAL_STEPS+=("$STAGE2_TOTAL_TRAINING_STEPS")
            done
        done
    done
elif [ "$QUEUE_MODE" = "kodcode_qwen3_1p7b_coldstart_fraction_stage2_p40_m2kl_vs_nokl" ]; then
    if [ "$MONITOR_NAME" = "code_task_queue" ]; then
        export MONITOR_NAME=code_task_kodcode_qwen3_1p7b_coldstart_fraction_stage2_p40_m2kl_vs_nokl
    fi
    export QUEUE_TMUX=${QUEUE_TMUX:-code_task_kodcode_qwen3_1p7b_coldstart_fraction_s2_p40_m2kl_vs_nokl_queue}
    export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/monitor_code_task_kodcode_qwen3_1p7b_coldstart_fraction_stage2_p40_m2kl_vs_nokl_notify.log}
    STAGE2_TOTAL_TRAINING_STEPS=${STAGE2_TOTAL_TRAINING_STEPS:-60}
    COLDSTART_FRACTION_STAGE2_FUSION_LAMBDAS=${COLDSTART_FRACTION_STAGE2_FUSION_LAMBDAS:-0.8}
    RUN_PREFIXES=()
    TMUX_NAMES=()
    FINAL_STEPS=()
    for fraction_tag in frac25 frac50; do
        fraction_prefix_tag=FRAC25
        [ "$fraction_tag" = "frac50" ] && fraction_prefix_tag=FRAC50
        for kl_mode in nokl model2; do
            for fusion_lambda in $COLDSTART_FRACTION_STAGE2_FUSION_LAMBDAS; do
                lambda_tag="lambda$(printf '%s' "$fusion_lambda" | tr -d '.')"
                kl_suffix=NOKL
                [ "$kl_mode" = "model2" ] && kl_suffix=M2KL
                RUN_PREFIXES+=("CODE-S2-KODCODE-Qwen3-1P7B-COLDSTART-${fraction_prefix_tag}-CTX8K-P40-BETA01-BETA01-${lambda_tag^^}-${kl_suffix}-V1")
                TMUX_NAMES+=("code_task_s2_q17b_cold_${fraction_tag}_b01_${kl_mode}_${lambda_tag}")
                FINAL_STEPS+=("$STAGE2_TOTAL_TRAINING_STEPS")
            done
        done
    done
elif [ "$QUEUE_MODE" = "submodel_kl_smoke" ]; then
    if [ "$MONITOR_NAME" = "code_task_queue" ]; then
        export MONITOR_NAME=submodel_kl_smoke
    fi
    export QUEUE_TMUX=${QUEUE_TMUX:-submodel_kl_smoke_queue}
    export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/monitor_submodel_kl_smoke_notify.log}
    export TRAINING_RELEASE_SUCCESS_HOOK=
    STAGE2_TOTAL_TRAINING_STEPS=${STAGE2_TOTAL_TRAINING_STEPS:-${TOTAL_TRAINING_STEPS:-5}}
    RUN_PREFIXES=(
        "${SUBMODEL_KL_BOTH_OFF_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-SUBKL-BOTHOFF-8GPU-SMOKE}"
        "${SUBMODEL_KL_MODEL1_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-SUBKL-MODEL1-8GPU-SMOKE}"
        "${SUBMODEL_KL_MODEL2_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-SUBKL-MODEL2-8GPU-SMOKE}"
        "${SUBMODEL_KL_BOTH_ON_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-SUBKL-BOTHON-8GPU-SMOKE}"
        "${SUBMODEL_KL_MODEL1_K1_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-SUBKL-MODEL1-K1-8GPU-SMOKE}"
        "${SUBMODEL_KL_MODEL2_K1_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-SUBKL-MODEL2-K1-8GPU-SMOKE}"
        "${SUBMODEL_KL_BOTH_ON_K1_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-SUBKL-BOTHON-K1-8GPU-SMOKE}"
        "${SUBMODEL_KL_MODEL1_K2_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-SUBKL-MODEL1-K2-8GPU-SMOKE}"
        "${SUBMODEL_KL_MODEL2_K2_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-SUBKL-MODEL2-K2-8GPU-SMOKE}"
        "${SUBMODEL_KL_BOTH_ON_K2_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-SUBKL-BOTHON-K2-8GPU-SMOKE}"
    )
    TMUX_NAMES=(
        "code_task_subkl_both_off_smoke"
        "code_task_subkl_model1_smoke"
        "code_task_subkl_model2_smoke"
        "code_task_subkl_both_on_smoke"
        "code_task_subkl_model1_k1_smoke"
        "code_task_subkl_model2_k1_smoke"
        "code_task_subkl_both_on_k1_smoke"
        "code_task_subkl_model1_k2_smoke"
        "code_task_subkl_model2_k2_smoke"
        "code_task_subkl_both_on_k2_smoke"
    )
    FINAL_STEPS=(
        "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS"
        "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS"
        "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS" "$STAGE2_TOTAL_TRAINING_STEPS"
    )
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
elif [ "$QUEUE_MODE" = "deepcoder_instruct2507_r8k_stage1" ]; then
    if [ "$MONITOR_NAME" = "code_task_queue" ]; then
        export MONITOR_NAME=code_task_deepcoder_instruct2507_r8k_stage1
    fi
    export QUEUE_TMUX=${QUEUE_TMUX:-code_task_deepcoder_instruct2507_r8k_stage1_queue}
    export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/monitor_code_task_deepcoder_instruct2507_r8k_stage1_notify.log}
    RUN_PREFIXES=(
        "ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-DEEPCODER-R8K-S1-BETA0-V1"
        "ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-DEEPCODER-R8K-S1-BETA01-V1"
    )
    TMUX_NAMES=("code_task_s1_deepcoder_i2507_r8k_beta0" "code_task_s1_deepcoder_i2507_r8k_beta01")
    FINAL_STEPS=(150 150)
elif [ "$QUEUE_MODE" = "kodcode_instruct2507_ctx8k_stage1" ]; then
    if [ "$MONITOR_NAME" = "code_task_queue" ]; then
        export MONITOR_NAME=code_task_kodcode_instruct2507_ctx8k_stage1
    fi
    export QUEUE_TMUX=${QUEUE_TMUX:-code_task_kodcode_instruct2507_ctx8k_stage1_queue}
    export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/monitor_code_task_kodcode_instruct2507_ctx8k_stage1_notify.log}
    RUN_PREFIXES=(
        "ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA0-V1"
        "ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA01-V1"
    )
    TMUX_NAMES=("code_task_s1_kodcode_i2507_ctx8k_beta0" "code_task_s1_kodcode_i2507_ctx8k_beta01")
    FINAL_STEPS=(150 150)
elif [ "$QUEUE_MODE" = "kodcode_qwen3_1p7b_instruct_ctx8k_stage1" ]; then
    if [ "$MONITOR_NAME" = "code_task_queue" ]; then
        export MONITOR_NAME=code_task_kodcode_qwen3_1p7b_instruct_ctx8k_stage1
    fi
    export QUEUE_TMUX=${QUEUE_TMUX:-code_task_kodcode_qwen3_1p7b_instruct_ctx8k_stage1_queue}
    export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/monitor_code_task_kodcode_qwen3_1p7b_instruct_ctx8k_stage1_notify.log}
    RUN_PREFIXES=(
        "ONPOLICY-SFT-Qwen3-1P7B-INSTRUCT-CODE-KODCODE-CTX8K-S1-BETA0-V1"
        "ONPOLICY-SFT-Qwen3-1P7B-INSTRUCT-CODE-KODCODE-CTX8K-S1-BETA01-V1"
    )
    TMUX_NAMES=("code_task_s1_kodcode_qwen3_1p7b_beta0" "code_task_s1_kodcode_qwen3_1p7b_beta01")
    FINAL_STEPS=(150 150)
elif [ "$QUEUE_MODE" = "kodcode_qwen3_1p7b_coldstart_ctx8k_stage1" ]; then
    if [ "$MONITOR_NAME" = "code_task_queue" ]; then
        export MONITOR_NAME=code_task_kodcode_qwen3_1p7b_coldstart_ctx8k_stage1
    fi
    export QUEUE_TMUX=${QUEUE_TMUX:-code_task_kodcode_qwen3_1p7b_coldstart_stage1_queue}
    export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/monitor_code_task_kodcode_qwen3_1p7b_instruct_ctx8k_coldstart_stage1_notify.log}
    RUN_PREFIXES=(
        "ONPOLICY-SFT-Qwen3-1P7B-COLDSTART-CODE-KODCODE-CTX8K-S1-BETA0-V1"
        "ONPOLICY-SFT-Qwen3-1P7B-COLDSTART-CODE-KODCODE-CTX8K-S1-BETA01-V1"
    )
    TMUX_NAMES=("code_task_s1_q17b_coldstart_beta0" "code_task_s1_q17b_coldstart_beta01")
    FINAL_STEPS=(150 150)
elif [ "$QUEUE_MODE" = "kodcode_qwen3_1p7b_coldstart_fraction_ctx8k_stage1" ]; then
    if [ "$MONITOR_NAME" = "code_task_queue" ]; then
        export MONITOR_NAME=code_task_kodcode_qwen3_1p7b_coldstart_fraction_ctx8k_stage1
    fi
    export QUEUE_TMUX=${QUEUE_TMUX:-code_task_kodcode_qwen3_1p7b_coldstart_fraction_stage1_queue}
    export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/monitor_code_task_kodcode_qwen3_1p7b_coldstart_fraction_stage1_notify.log}
    RUN_PREFIXES=(
        "ONPOLICY-SFT-Qwen3-1P7B-COLDSTART-FRAC25-CODE-KODCODE-CTX8K-S1-BETA0-V1"
        "ONPOLICY-SFT-Qwen3-1P7B-COLDSTART-FRAC25-CODE-KODCODE-CTX8K-S1-BETA01-V1"
        "ONPOLICY-SFT-Qwen3-1P7B-COLDSTART-FRAC50-CODE-KODCODE-CTX8K-S1-BETA0-V1"
        "ONPOLICY-SFT-Qwen3-1P7B-COLDSTART-FRAC50-CODE-KODCODE-CTX8K-S1-BETA01-V1"
    )
    TMUX_NAMES=(
        "code_task_s1_q17b_coldstart_frac25_beta0"
        "code_task_s1_q17b_coldstart_frac25_beta01"
        "code_task_s1_q17b_coldstart_frac50_beta0"
        "code_task_s1_q17b_coldstart_frac50_beta01"
    )
    FINAL_STEPS=(150 150 150 150)
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
