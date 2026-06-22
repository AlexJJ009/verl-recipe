#!/usr/bin/env bash
# Full official code benchmark eval for the latest Stage2 P70 checkpoints at
# local step 30, i.e. Stage1-equivalent effective step 100.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_HOST=${REPO_HOST:-/data-1/verl07/verl}
REPO_CONTAINER=${REPO_CONTAINER:-/workspace/verl}
DOCKER_IMAGE=${DOCKER_IMAGE:-verl-harness}
LOG_DIR=${LOG_DIR:-"${REPO_HOST}/recipe/on_policy_wdl_sft/code_task/eval_logs"}
QUEUE_LOG=${QUEUE_LOG:-"${LOG_DIR}/run_code_stage2_effective100_offline_n3_queue.log"}
STATUS_FILE=${STATUS_FILE:-"${SCRIPT_DIR}/run_code_stage2_effective100_offline_n3_status.tsv"}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data-1/eval_outputs/code_task/stage2_effective100_unified_n3}
TEMP_MODEL_ROOT=${TEMP_MODEL_ROOT:-/data-1/model_weights/code_task/stage2_effective100_unified_n3/tmp}
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
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASHINFER}
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
BENCHMARKS=${BENCHMARKS:-"humaneval mbpp bigcodebench livecodebench"}
START_INDEX=${START_INDEX:-0}
END_INDEX=${END_INDEX:-1}
POLL_SEC=${POLL_SEC:-300}
MIN_FREE_GB=${MIN_FREE_GB:-80}
MAX_GPU_UTIL_TOTAL=${MAX_GPU_UTIL_TOTAL:-50}
DOCKER_GPUS=${DOCKER_GPUS:-all}
DRY_RUN=${DRY_RUN:-0}
FORCE_EVAL=${FORCE_EVAL:-0}

LABELS=(
    "s2_p70_beta0_effective100_step030"
    "s2_p70_beta01_effective100_step030"
)

ACTOR_DIRS=(
    "/data-1/checkpoints/CODE-S2-RETENTION-BETA0-BETA0_1780898035/global_step_30/actor"
    "/data-1/checkpoints/CODE-S2-RETENTION-BETA01-BETA01_1780902470/global_step_30/actor"
)

log() {
    mkdir -p "${LOG_DIR}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${QUEUE_LOG}"
}

record_status() {
    local index="$1" label="$2" status="$3" detail="$4"
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$index" "$label" "$status" "$detail" >> "${STATUS_FILE}"
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

validate_inputs() {
    local ok=1 actor model_count extra_count
    for actor in "${ACTOR_DIRS[@]}"; do
        if [ ! -d "${actor}" ]; then
            log "ERROR: missing actor checkpoint: ${actor}"
            ok=0
            continue
        fi
        model_count=$(find "${actor}" -maxdepth 1 -type f -name 'model_world_size_*_rank_*.pt' | wc -l)
        extra_count=$(find "${actor}" -maxdepth 1 -type f -name 'extra_state_world_size_*_rank_*.pt' | wc -l)
        if [ "${model_count}" -ne 8 ] || [ "${extra_count}" -ne 8 ]; then
            log "ERROR: incomplete actor checkpoint: ${actor} model=${model_count} extra=${extra_count}"
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

run_checkpoint() {
    local i="$1"
    local label="${LABELS[$i]}"
    local actor="${ACTOR_DIRS[$i]}"
    local joint_dir="${TEMP_MODEL_ROOT}/${label}_joint"
    local model2_dir="${TEMP_MODEL_ROOT}/${label}_model2"
    local case_log="${LOG_DIR}/${label}_official_n${N_SAMPLES}.log"

    log "checkpoint index=${i} label=${label} actor=${actor} output=${OUTPUT_ROOT}/${label}"
    if [ "${DRY_RUN}" = "1" ]; then
        return 0
    fi

    wait_for_resources
    mkdir -p "${TEMP_MODEL_ROOT}" "${OUTPUT_ROOT}"
    rm -rf "${joint_dir}" "${model2_dir}"

    if docker run --rm --gpus "${DOCKER_GPUS}" --ipc=host --shm-size=64g \
        --env LABEL="${label}" \
        --env ACTOR="${actor}" \
        --env JOINT_DIR="${joint_dir}" \
        --env MODEL2_DIR="${model2_dir}" \
        --env OUTPUT_ROOT="${OUTPUT_ROOT}" \
        --env BENCHMARKS="${BENCHMARKS}" \
        --env N_SAMPLES="${N_SAMPLES}" \
        --env TEMPERATURE="${TEMPERATURE}" \
        --env TOP_P="${TOP_P}" \
        --env MAX_TOKENS="${MAX_TOKENS}" \
        --env SEED="${SEED}" \
        --env ENABLE_THINKING="${ENABLE_THINKING}" \
        --env TENSOR_PARALLEL="${TENSOR_PARALLEL}" \
        --env GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
        --env CODE_OFFICIAL_EVAL_PARALLEL="${CODE_OFFICIAL_EVAL_PARALLEL}" \
        --env PROJECT_CACHE_ROOT="${PROJECT_CACHE_ROOT}" \
        --env HF_HOME="${HF_HOME}" \
        --env HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
        --env HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}" \
        --env TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
        --env XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
        --env CODE_OFFICIAL_SOURCE_ROOT="${CODE_OFFICIAL_SOURCE_ROOT}" \
        --env BIGCODEBENCH_OVERRIDE_PATH="${BIGCODEBENCH_OVERRIDE_PATH}" \
        --env LCB_REPO_DIR="${LCB_REPO_DIR}" \
        --env LCB_PYTHON="${LCB_PYTHON}" \
        --env VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND}" \
        --env BCB_CALIBRATED="${BCB_CALIBRATED}" \
        --env HF_HUB_OFFLINE=1 \
        --env HF_DATASETS_OFFLINE=1 \
        --env WXPUSHER_NOTIFY=0 \
        --env FORCE_EVAL="${FORCE_EVAL}" \
        -v /data-1:/data-1 \
        -v "${EVALPLUS_CACHE_HOST}:/data-1/.cache/evalplus:ro" \
        -v "${REPO_HOST}:${REPO_CONTAINER}" \
        -w "${REPO_CONTAINER}" \
        "${DOCKER_IMAGE}" \
        bash -lc ' \
            set -euo pipefail
            trap '\''rm -rf "${JOINT_DIR}" "${MODEL2_DIR}"'\'' EXIT
            echo "[stage2-effective100] merge joint: ${ACTOR} -> ${JOINT_DIR}"
            python3 -u -m verl.model_merger merge \
                --backend fsdp \
                --local_dir "${ACTOR}" \
                --target_dir "${JOINT_DIR}" \
                --trust-remote-code
            echo "[stage2-effective100] extract model2: ${JOINT_DIR} -> ${MODEL2_DIR}"
            python3 -u recipe/joint_training/extract_sub_model.py \
                --joint_model_path "${JOINT_DIR}" \
                --output_path "${MODEL2_DIR}" \
                --sub_model_index 1
            for benchmark in ${BENCHMARKS}; do
                echo "[stage2-effective100] eval ${LABEL}/${benchmark}"
                MODEL_LABEL="${LABEL}" \
                BENCHMARK="${benchmark}" \
                FSDP_ACTOR_DIR="${ACTOR}" \
                MERGED_MODEL_DIR="${MODEL2_DIR}" \
                OUTPUT_ROOT="${OUTPUT_ROOT}" \
                N_SAMPLES="${N_SAMPLES}" \
                TEMPERATURE="${TEMPERATURE}" \
                TOP_P="${TOP_P}" \
                MAX_TOKENS="${MAX_TOKENS}" \
                SEED="${SEED}" \
                ENABLE_THINKING="${ENABLE_THINKING}" \
                TENSOR_PARALLEL="${TENSOR_PARALLEL}" \
                GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
                CODE_OFFICIAL_EVAL_PARALLEL="${CODE_OFFICIAL_EVAL_PARALLEL}" \
                VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND}" \
                BIGCODEBENCH_OVERRIDE_PATH="${BIGCODEBENCH_OVERRIDE_PATH}" \
                LCB_PYTHON="${LCB_PYTHON}" \
                BCB_CALIBRATED="${BCB_CALIBRATED}" \
                FORCE_EVAL="${FORCE_EVAL}" \
                WXPUSHER_NOTIFY=0 \
                    bash recipe/on_policy_wdl_sft/code_task/run_code_offline_eval_case.sh
            done
        ' 2>&1 | tee "${case_log}"; then
        record_status "${i}" "${label}" "completed" "benchmarks=${BENCHMARKS};output=${OUTPUT_ROOT}/${label};log=${case_log}"
        log "completed checkpoint index=${i} label=${label}"
    else
        record_status "${i}" "${label}" "failed" "log=${case_log}"
        log "failed checkpoint index=${i} label=${label}"
        return 1
    fi
}

main() {
    mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}" "${TEMP_MODEL_ROOT}"
    validate_inputs
    log "Stage2 effective100 unified N=3 code offline eval queue start: benchmarks='${BENCHMARKS}' output=${OUTPUT_ROOT}"
    log "settings: n=${N_SAMPLES} temperature=${TEMPERATURE} top_p=${TOP_P} max_tokens=${MAX_TOKENS} seed=${SEED} enable_thinking=${ENABLE_THINKING} tp=${TENSOR_PARALLEL} start=${START_INDEX} end=${END_INDEX}"
    if [ "${DRY_RUN}" != "1" ] && [ "${ALLOW_CODE_STAGE2_EFFECTIVE100_OFFLINE_EVAL:-0}" != "1" ]; then
        log "ERROR: non-dry-run requires ALLOW_CODE_STAGE2_EFFECTIVE100_OFFLINE_EVAL=1"
        exit 1
    fi

    local i
    for i in "${!LABELS[@]}"; do
        if [ "${i}" -lt "${START_INDEX}" ] || [ "${i}" -gt "${END_INDEX}" ]; then
            log "skip index=${i} label=${LABELS[$i]}"
            continue
        fi
        run_checkpoint "${i}"
    done

    if [ "${DRY_RUN}" != "1" ]; then
        python3 "${SCRIPT_DIR}/summarize_code_offline_eval_n3.py" \
            --output-root "${OUTPUT_ROOT}" \
            --labels "${LABELS[@]}" \
            --summary-json "${OUTPUT_ROOT}/summary_stage2_effective100_unified_n3.json" \
            --summary-md "${OUTPUT_ROOT}/summary_stage2_effective100_unified_n3.md"
    fi
    log "Stage2 effective100 unified N=3 code offline eval queue complete. Results root: ${OUTPUT_ROOT}"
}

main "$@"
