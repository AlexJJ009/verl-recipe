#!/usr/bin/env bash
# Host-side queue for V2 Stage1 latest-step code offline eval under one N=3
# diagnostic setting across all code benchmarks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_HOST=${REPO_HOST:-/root/buaa/local_data1/verl07/verl}
REPO_CONTAINER=${REPO_CONTAINER:-/workspace/verl}
DOCKER_IMAGE=${DOCKER_IMAGE:-verl-harness}
LOG_DIR=${LOG_DIR:-"${REPO_HOST}/recipe/on_policy_wdl_sft/code_task/eval_logs"}
QUEUE_LOG=${QUEUE_LOG:-"${LOG_DIR}/run_code_offline_eval_v2_latest_n3_queue.log"}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data-1/eval_outputs/code_task/v2_latest_unified_n3}
PROJECT_CACHE_ROOT=${PROJECT_CACHE_ROOT:-/data-1/.cache}
HF_HOME=${CODE_TASK_HF_HOME:-$PROJECT_CACHE_ROOT/huggingface}
HF_DATASETS_CACHE=${CODE_TASK_HF_DATASETS_CACHE:-$HF_HOME/datasets}
HUGGINGFACE_HUB_CACHE=${CODE_TASK_HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
TRANSFORMERS_CACHE=${CODE_TASK_TRANSFORMERS_CACHE:-$HF_HOME}
XDG_CACHE_HOME=${CODE_TASK_XDG_CACHE_HOME:-$PROJECT_CACHE_ROOT}
EVALPLUS_CACHE_HOST=${EVALPLUS_CACHE_HOST:-$PROJECT_CACHE_ROOT/evalplus}
CODE_OFFICIAL_SOURCE_ROOT=${CODE_OFFICIAL_SOURCE_ROOT:-/data-1/dataset/code/official_sources}
BIGCODEBENCH_OVERRIDE_PATH=${BIGCODEBENCH_OVERRIDE_PATH:-$CODE_OFFICIAL_SOURCE_ROOT/bigcodebench/BigCodeBench-v0.1.4.jsonl}
LCB_REPO_DIR=${LCB_REPO_DIR:-/data-1/code_eval_envs/LiveCodeBench}
LCB_PYTHON=${LCB_PYTHON:-/opt/venv/bin/python}
LCB_RELEASE_VERSION=${LCB_RELEASE_VERSION:-release_v5}
BCB_CALIBRATED=${BCB_CALIBRATED:-1}

N_SAMPLES=${N_SAMPLES:-3}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}
MAX_TOKENS=${MAX_TOKENS:-4096}
SEED=${SEED:-42}
ENABLE_THINKING=${ENABLE_THINKING:-true}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-4}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
CODE_OFFICIAL_EVAL_PARALLEL=${CODE_OFFICIAL_EVAL_PARALLEL:-8}
DOCKER_GPUS=${DOCKER_GPUS:-all}
CASE_CUDA_VISIBLE_DEVICES=${CASE_CUDA_VISIBLE_DEVICES:-}
BENCHMARKS=${BENCHMARKS:-"humaneval mbpp bigcodebench livecodebench"}
START_INDEX=${START_INDEX:-0}
END_INDEX=${END_INDEX:-1}
POLL_SEC=${POLL_SEC:-300}
MIN_FREE_GB=${MIN_FREE_GB:-80}
MAX_GPU_UTIL_TOTAL=${MAX_GPU_UTIL_TOTAL:-50}
DRY_RUN=${DRY_RUN:-0}

LABELS=("v2_beta0_latest_step150" "v2_beta01_latest_step150")
ACTOR_DIRS=(
    "/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V2_1780685616/global_step_150/actor"
    "/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V2_1780736174/global_step_150/actor"
)
MERGED_DIRS=(
    "/data-1/model_weights/code_task/offline_eval/v2_beta0_latest_step150/actor_step150"
    "/data-1/model_weights/code_task/offline_eval/v2_beta01_latest_step150/actor_step150"
)

log() {
    mkdir -p "${LOG_DIR}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${QUEUE_LOG}"
}

notify() {
    local title="$1" body="$2"
    if [ "${WXPUSHER_NOTIFY:-1}" = "1" ]; then
        python3 /root/agent-core/skills/wxpusher-notify/scripts/wxpusher_notify.py \
            --title "$title" \
            --body "$body" || true
    fi
}

free_gb() {
    df -Pk /data-1 | awk 'NR==2 {print int($4 / 1024 / 1024)}'
}

gpu_util_total() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
        | awk '{s+=$1} END {print s+0}'
}

validate_inputs() {
    local ok=1
    for actor in "${ACTOR_DIRS[@]}"; do
        if [ ! -d "${actor}" ]; then
            log "ERROR: missing actor checkpoint: ${actor}"
            ok=0
        fi
    done
    if [ ! -d "${EVALPLUS_CACHE_HOST}" ]; then
        log "ERROR: missing EvalPlus cache: ${EVALPLUS_CACHE_HOST}"
        ok=0
    fi
    if [[ " ${BENCHMARKS} " == *" bigcodebench "* ]] && [ ! -f "${BIGCODEBENCH_OVERRIDE_PATH}" ]; then
        log "ERROR: missing BigCodeBench official JSONL: ${BIGCODEBENCH_OVERRIDE_PATH}"
        ok=0
    fi
    if [ ! -d "${REPO_HOST}" ]; then
        log "ERROR: missing REPO_HOST: ${REPO_HOST}"
        ok=0
    fi
    if [ "${ok}" != "1" ]; then
        exit 2
    fi
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
    local i="$1" benchmark="$2"
    local label="${LABELS[$i]}"
    local actor="${ACTOR_DIRS[$i]}"
    local merged="${MERGED_DIRS[$i]}"
    local case_log="${LOG_DIR}/${label}_${benchmark}_n${N_SAMPLES}.log"
    local docker_gpus_arg="${DOCKER_GPUS}"

    if [[ "${docker_gpus_arg}" != "all" && "${docker_gpus_arg}" != \"* ]]; then
        docker_gpus_arg="\"${docker_gpus_arg}\""
    fi

    log "case: index=${i} label=${label} benchmark=${benchmark} actor=${actor} merged=${merged} output=${OUTPUT_ROOT}/${label}/${benchmark}"
    if [ "${DRY_RUN}" = "1" ]; then
        return
    fi

    wait_for_resources

    local env_prefix
    env_prefix="MODEL_LABEL=${label} BENCHMARK=${benchmark} FSDP_ACTOR_DIR=${actor} MERGED_MODEL_DIR=${merged} OUTPUT_ROOT=${OUTPUT_ROOT} N_SAMPLES=${N_SAMPLES} TEMPERATURE=${TEMPERATURE} TOP_P=${TOP_P} MAX_TOKENS=${MAX_TOKENS} SEED=${SEED} ENABLE_THINKING=${ENABLE_THINKING} TENSOR_PARALLEL=${TENSOR_PARALLEL} GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION} CODE_OFFICIAL_EVAL_PARALLEL=${CODE_OFFICIAL_EVAL_PARALLEL} PROJECT_CACHE_ROOT=${PROJECT_CACHE_ROOT} HF_HOME=${HF_HOME} HF_DATASETS_CACHE=${HF_DATASETS_CACHE} HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE} TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} XDG_CACHE_HOME=${XDG_CACHE_HOME} CODE_OFFICIAL_SOURCE_ROOT=${CODE_OFFICIAL_SOURCE_ROOT} BIGCODEBENCH_OVERRIDE_PATH=${BIGCODEBENCH_OVERRIDE_PATH} LCB_REPO_DIR=${LCB_REPO_DIR} LCB_PYTHON=${LCB_PYTHON} LCB_RELEASE_VERSION=${LCB_RELEASE_VERSION} BCB_CALIBRATED=${BCB_CALIBRATED} HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WXPUSHER_NOTIFY=0"
    if [ -n "${CASE_CUDA_VISIBLE_DEVICES}" ]; then
        env_prefix="${env_prefix} CUDA_VISIBLE_DEVICES=${CASE_CUDA_VISIBLE_DEVICES}"
    fi

    log "start case ${i}: ${label}/${benchmark}; log=${case_log}"
    if docker run --rm --gpus "${docker_gpus_arg}" --ipc=host --shm-size=64g \
        -v /data-1:/data-1 \
        -v "${EVALPLUS_CACHE_HOST}:/data-1/.cache/evalplus:ro" \
        -v "${REPO_HOST}:${REPO_CONTAINER}" \
        -w "${REPO_CONTAINER}" \
        "${DOCKER_IMAGE}" \
        bash -lc "${env_prefix} bash recipe/on_policy_wdl_sft/code_task/run_code_offline_eval_case.sh" \
        2>&1 | tee "${case_log}"; then
        log "complete case ${i}: ${label}/${benchmark}"
    else
        log "failed case ${i}: ${label}/${benchmark}"
        notify "Code V2 latest offline eval failed" "Status: failed\nWhat happened: ${label}/${benchmark} failed.\nEvidence: ${case_log}\nNext action: inspect official summary/log."
        return 1
    fi
}

main() {
    mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
    validate_inputs
    log "V2 latest unified N=3 code offline eval queue start"
    log "settings: benchmarks='${BENCHMARKS}' n=${N_SAMPLES} temperature=${TEMPERATURE} top_p=${TOP_P} max_tokens=${MAX_TOKENS} seed=${SEED} enable_thinking=${ENABLE_THINKING} tp=${TENSOR_PARALLEL} lcb_release=${LCB_RELEASE_VERSION} bcb_calibrated=${BCB_CALIBRATED} output_root=${OUTPUT_ROOT}"
    if [ "${DRY_RUN}" != "1" ] && [ "${ALLOW_CODE_V2_LATEST_OFFLINE_EVAL:-0}" != "1" ]; then
        log "ERROR: non-dry-run requires ALLOW_CODE_V2_LATEST_OFFLINE_EVAL=1"
        exit 1
    fi

    for i in "${!LABELS[@]}"; do
        if [ "${i}" -lt "${START_INDEX}" ] || [ "${i}" -gt "${END_INDEX}" ]; then
            log "skip index ${i}: ${LABELS[$i]}"
            continue
        fi
        for benchmark in ${BENCHMARKS}; do
            run_case "${i}" "${benchmark}"
        done
    done
    if [ "${DRY_RUN}" != "1" ]; then
        python3 "${SCRIPT_DIR}/summarize_code_offline_eval_n3.py" \
            --output-root "${OUTPUT_ROOT}" \
            --labels "${LABELS[@]}" \
            --summary-json "${OUTPUT_ROOT}/summary_v2_latest_unified_n3.json" \
            --summary-md "${OUTPUT_ROOT}/summary_v2_latest_unified_n3.md"
    fi
    log "V2 latest unified N=3 code offline eval queue complete. Results root: ${OUTPUT_ROOT}"
}

main "$@"
