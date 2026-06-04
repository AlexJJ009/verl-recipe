#!/usr/bin/env bash
# Host-side sequential Math-7 offline eval queue for the P60 plateau handoff runs.

set -euo pipefail

REPO_HOST=${REPO_HOST:-/data-1/verl07/verl}
REPO_CONTAINER=${REPO_CONTAINER:-/workspace/verl}
DOCKER_IMAGE=${DOCKER_IMAGE:-verl-harness}
LAUNCHER=${LAUNCHER:-/data-1/verl07/run_train.sh}
LOG_DIR=${LOG_DIR:-"${REPO_HOST}/recipe/on_policy_wdl_sft/staged_v1/eval_logs"}
QUEUE_LOG=${QUEUE_LOG:-"${LOG_DIR}/run_plateau_p60_math7_eval_queue.log"}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data-1/model_weights/staged_v1/plateau_handoff_p60/math7_eval}
N_SAMPLES=${N_SAMPLES:-3}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}
MAX_TOKENS=${MAX_TOKENS:-4096}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-8}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
SEED=${SEED:-42}
FORCE_EVAL=${FORCE_EVAL:-0}
START_INDEX=${START_INDEX:-0}
END_INDEX=${END_INDEX:-5}
POLL_SEC=${POLL_SEC:-300}
MIN_FREE_GB=${MIN_FREE_GB:-120}
MAX_GPU_UTIL_TOTAL=${MAX_GPU_UTIL_TOTAL:-50}
QUEUE_TMUX=${QUEUE_TMUX:-staged_v1_p60_math7_eval_queue}
RUNNER="${REPO_CONTAINER}/recipe/on_policy_wdl_sft/staged_v1/run_math7_eval_model2_common.sh"

LABELS=(
    "s1_p60_beta0_step60_model2"
    "s2_p60_beta0_step20_best_model2"
    "s2_p60_beta0_step40_final_model2"
    "s1_p60_beta01_step60_model2"
    "s2_p60_beta01_step35_best_model2"
    "s2_p60_beta01_step40_final_model2"
)

SOURCE_MODEL_PATHS=(
    "/data-1/model_weights/staged_v1/plateau_handoff_p60/model2-from-s1-p60-beta0-step60"
    ""
    ""
    "/data-1/model_weights/staged_v1/plateau_handoff_p60/model2-from-s1-p60-beta01-step60"
    ""
    ""
)

FSDP_ACTOR_DIRS=(
    ""
    "/data-1/checkpoints/WDL-SFT-STAGED-V1-S2-PLATEAU-P60-BETA0-BETA0_1780389822/global_step_20/actor"
    "/data-1/checkpoints/WDL-SFT-STAGED-V1-S2-PLATEAU-P60-BETA0-BETA0_1780389822/global_step_40/actor"
    ""
    "/data-1/checkpoints/WDL-SFT-STAGED-V1-S2-PLATEAU-P60-BETA01-BETA01_1780460682/global_step_35/actor"
    "/data-1/checkpoints/WDL-SFT-STAGED-V1-S2-PLATEAU-P60-BETA01-BETA01_1780460682/global_step_40/actor"
)

MERGED_JOINT_PATHS=(
    ""
    "/data-1/model_weights/staged_v1/plateau_handoff_p60/s2-p60-beta0-step20-best-joint"
    "/data-1/model_weights/staged_v1/plateau_handoff_p60/s2-p60-beta0-step40-final-joint"
    ""
    "/data-1/model_weights/staged_v1/plateau_handoff_p60/s2-p60-beta01-step35-best-joint"
    "/data-1/model_weights/staged_v1/plateau_handoff_p60/s2-p60-beta01-step40-final-joint"
)

log() {
    mkdir -p "${LOG_DIR}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${QUEUE_LOG}"
}

free_gb() {
    df -Pk /data-1 | awk 'NR==2 {print int($4 / 1024 / 1024)}'
}

gpu_util_total() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
        | awk '{s+=$1} END {print s+0}'
}

wait_for_resources() {
    while true; do
        local free util
        free=$(free_gb)
        util=$(gpu_util_total)
        if [ "${free}" -ge "${MIN_FREE_GB}" ] && [ "${util}" -lt "${MAX_GPU_UTIL_TOTAL}" ]; then
            log "resources ok: free=${free}G gpu_util_total=${util}"
            return
        fi
        log "waiting resources: free=${free}G need>=${MIN_FREE_GB}G gpu_util_total=${util} need<${MAX_GPU_UTIL_TOTAL}; sleep ${POLL_SEC}s"
        sleep "${POLL_SEC}"
    done
}

run_case() {
    local i="$1"
    local label="${LABELS[$i]}"
    local source="${SOURCE_MODEL_PATHS[$i]}"
    local actor="${FSDP_ACTOR_DIRS[$i]}"
    local merged="${MERGED_JOINT_PATHS[$i]}"
    local case_log="${LOG_DIR}/${label}.log"
    local env_cmd

    wait_for_resources
    log "start case ${i}: ${label}; log=${case_log}"

    env_cmd="MODEL_LABEL=${label} OUTPUT_ROOT=${OUTPUT_ROOT} N_SAMPLES=${N_SAMPLES} TEMPERATURE=${TEMPERATURE} TOP_P=${TOP_P} MAX_TOKENS=${MAX_TOKENS} TENSOR_PARALLEL=${TENSOR_PARALLEL} GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION} SEED=${SEED} FORCE_EVAL=${FORCE_EVAL}"
    if [ -n "${source}" ]; then
        env_cmd="${env_cmd} SOURCE_MODEL_PATH=${source}"
    else
        env_cmd="${env_cmd} FSDP_ACTOR_DIR=${actor} MERGED_JOINT_PATH=${merged}"
    fi

    if [ -x "${LAUNCHER}" ]; then
        bash "${LAUNCHER}" bash -lc "cd ${REPO_CONTAINER} && ${env_cmd} bash ${RUNNER}" 2>&1 | tee "${case_log}"
    else
        docker run --rm --gpus all --ipc=host --shm-size=64g \
            -v /data-1:/data-1 \
            -v "${REPO_HOST}:${REPO_CONTAINER}" \
            -w "${REPO_CONTAINER}" \
            "${DOCKER_IMAGE}" \
            bash -lc "${env_cmd} bash ${RUNNER}" 2>&1 | tee "${case_log}"
    fi

    log "complete case ${i}: ${label}"
}

main() {
    mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
    log "P60 Math-7 eval queue start: START_INDEX=${START_INDEX} END_INDEX=${END_INDEX} n=${N_SAMPLES} temperature=${TEMPERATURE} top_p=${TOP_P}"
    for i in "${!LABELS[@]}"; do
        if [ "${i}" -lt "${START_INDEX}" ] || [ "${i}" -gt "${END_INDEX}" ]; then
            log "skip case ${i}: ${LABELS[$i]}"
            continue
        fi
        run_case "${i}"
    done
    log "P60 Math-7 eval queue complete. Results root: ${OUTPUT_ROOT}"
}

main "$@"
