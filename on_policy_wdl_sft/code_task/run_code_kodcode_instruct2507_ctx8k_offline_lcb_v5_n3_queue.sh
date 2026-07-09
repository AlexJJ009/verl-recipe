#!/usr/bin/env bash
# Host-side LiveCodeBench release_v5 official eval queue for KodCode Instruct-2507 CTX8K Stage1 checkpoints.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_HOST=${REPO_HOST:-/data-1/verl07/verl}
REPO_CONTAINER=${REPO_CONTAINER:-/workspace/verl}
DOCKER_IMAGE=${DOCKER_IMAGE:-verl-harness}
LOG_DIR=${LOG_DIR:-"${REPO_HOST}/recipe/on_policy_wdl_sft/code_task/eval_logs"}
QUEUE_LOG=${QUEUE_LOG:-"${LOG_DIR}/run_code_kodcode_instruct2507_ctx8k_offline_lcb_v5_n3_queue.log"}
STATUS_FILE=${STATUS_FILE:-"${LOG_DIR}/run_code_kodcode_instruct2507_ctx8k_offline_lcb_v5_n3_status.tsv"}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data-1/eval_outputs/code_task/kodcode_instruct2507_ctx8k_lcb_v5_n3}
MODEL_WEIGHT_ROOT=${MODEL_WEIGHT_ROOT:-/data-1/model_weights/code_task/offline_eval}
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
LCB_JSONL_DIR=${LCB_JSONL_DIR:-$HF_HOME/hub/datasets--livecodebench--code_generation_lite/snapshots/0fe84c3912ea0c4d4a78037083943e8f0c4dd505}
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
BENCHMARKS=${BENCHMARKS:-"livecodebench"}
START_INDEX=${START_INDEX:-0}
END_INDEX=${END_INDEX:-3}
POLL_SEC=${POLL_SEC:-300}
MIN_FREE_GB=${MIN_FREE_GB:-220}
MAX_GPU_UTIL_TOTAL=${MAX_GPU_UTIL_TOTAL:-50}
DRY_RUN=${DRY_RUN:-0}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}

LABELS=(
    "kodcode_i2507_ctx8k_beta0_step145"
    "kodcode_i2507_ctx8k_beta0_step150"
    "kodcode_i2507_ctx8k_beta01_step115"
    "kodcode_i2507_ctx8k_beta01_step150"
)
ACTOR_DIRS=(
    "/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA0-V1_1782371396/global_step_145/actor"
    "/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA0-V1_1782371396/global_step_150/actor"
    "/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA01-V1_1782398871/global_step_115/actor"
    "/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA01-V1_1782398871/global_step_150/actor"
)
MERGED_DIRS=(
    "${MODEL_WEIGHT_ROOT}/kodcode_i2507_ctx8k_beta0_step145/actor_step145"
    "${MODEL_WEIGHT_ROOT}/kodcode_i2507_ctx8k_beta0_step150/actor_step150"
    "${MODEL_WEIGHT_ROOT}/kodcode_i2507_ctx8k_beta01_step115/actor_step115"
    "${MODEL_WEIGHT_ROOT}/kodcode_i2507_ctx8k_beta01_step150/actor_step150"
)

log() {
    mkdir -p "${LOG_DIR}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${QUEUE_LOG}"
}

record_status() {
    local idx="$1" label="$2" benchmark="$3" phase="$4" status="$5" evidence="$6"
    mkdir -p "$(dirname "${STATUS_FILE}")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "$idx" "$label" "$benchmark" "$phase" "$status" "$evidence" >>"${STATUS_FILE}"
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

merged_ready() {
    local merged="$1"
    [ -f "${merged}/model.safetensors.index.json" ] || [ -f "${merged}/model.safetensors" ]
}

validation_parquet_for_benchmark() {
    local benchmark="$1"
    case "${benchmark}" in
        livecodebench)
            case "${LCB_RELEASE_VERSION}" in
                release_v1) echo "/data-1/dataset/code/verl_rl/online_full_livecodebench/official_livecodebench_val.parquet" ;;
                release_v5) echo "/data-1/dataset/code/verl_rl/online_full_livecodebench_v5/official_livecodebench_val.parquet" ;;
                *) echo "/data-1/dataset/code/verl_rl/online_full_livecodebench_${LCB_RELEASE_VERSION}/official_livecodebench_val.parquet" ;;
            esac
            ;;
        *) echo "ERROR: this queue is LCB-only; unsupported benchmark ${benchmark}" >&2; return 2 ;;
    esac
}

validate_inputs() {
    local ok=1 free
    free=$(free_gb)
    log "disk check: /data-1 free=${free}G min_free=${MIN_FREE_GB}G output_root=${OUTPUT_ROOT} model_weight_root=${MODEL_WEIGHT_ROOT}"
    if [ "${free}" -lt "${MIN_FREE_GB}" ]; then
        log "ERROR: /data-1 free space below threshold"
        ok=0
    fi
    for actor in "${ACTOR_DIRS[@]}"; do
        if [ ! -d "${actor}" ]; then
            log "ERROR: missing actor checkpoint: ${actor}"
            ok=0
            continue
        fi
        local model_count extra_count
        model_count=$(find "${actor}" -maxdepth 1 -type f -name 'model_world_size_*_rank_*.pt' | wc -l)
        extra_count=$(find "${actor}" -maxdepth 1 -type f -name 'extra_state_world_size_*_rank_*.pt' | wc -l)
        if [ "${model_count}" -ne 8 ] || [ "${extra_count}" -ne 8 ]; then
            log "ERROR: incomplete actor checkpoint: ${actor} model=${model_count} extra=${extra_count}"
            ok=0
        fi
    done
    for benchmark in ${BENCHMARKS}; do
        local parquet
        parquet="$(validation_parquet_for_benchmark "${benchmark}")" || ok=0
        if [ -n "${parquet:-}" ] && [ ! -f "${parquet}" ]; then
            log "ERROR: missing validation parquet for ${benchmark}: ${parquet}"
            ok=0
        fi
    done
    if [ ! -d "${EVALPLUS_CACHE_HOST}" ]; then
        log "ERROR: missing EvalPlus cache: ${EVALPLUS_CACHE_HOST}"
        ok=0
    fi
    if [ ! -d "${LCB_REPO_DIR}" ]; then
        log "ERROR: missing LiveCodeBench repo: ${LCB_REPO_DIR}"
        ok=0
    fi
    if [ ! -d "${LCB_JSONL_DIR}" ]; then
        log "ERROR: missing LiveCodeBench local JSONL snapshot: ${LCB_JSONL_DIR}"
        ok=0
    fi
    if ! docker image inspect "${DOCKER_IMAGE}" >/dev/null 2>&1; then
        log "ERROR: missing docker image: ${DOCKER_IMAGE}"
        ok=0
    fi
    if docker run --rm \
        -v /data-1:/data-1 \
        -v "${REPO_HOST}:${REPO_CONTAINER}" \
        -w "${REPO_CONTAINER}" \
        --env LCB_REPO_DIR="${LCB_REPO_DIR}" \
        --env LCB_PYTHON="${LCB_PYTHON}" \
        --env LCB_RELEASE_VERSION="${LCB_RELEASE_VERSION}" \
        --env LCB_JSONL_DIR="${LCB_JSONL_DIR}" \
        --env PROJECT_CACHE_ROOT="${PROJECT_CACHE_ROOT}" \
        --env HF_HOME="${HF_HOME}" \
        --env HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
        --env HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}" \
        --env TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
        --env XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
        --env HF_HUB_OFFLINE=1 \
        --env HF_DATASETS_OFFLINE=1 \
        "${DOCKER_IMAGE}" \
        bash -lc 'test -x "${LCB_PYTHON}" && cd "${LCB_REPO_DIR}" && PYTHONPATH="${LCB_REPO_DIR}:${PYTHONPATH:-}" "${LCB_PYTHON}" - <<'"'"'PY'"'"'
import json
import os
from pathlib import Path

import lcb_runner.runner.custom_evaluator
from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

release_files = {
    "release_v1": ["test.jsonl"],
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
}
files = release_files[os.environ["LCB_RELEASE_VERSION"]]
root = Path(os.environ["LCB_JSONL_DIR"])
total = 0
first = None
last = None
for name in files:
    path = root / name
    assert path.is_file(), path
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
                row = CodeGenerationProblem(**json.loads(line))
                first = first or row.question_id
                last = row.question_id
assert total > 0, "empty LiveCodeBench local JSONL loader"
print(total, first, last)
PY' \
        >/dev/null 2>&1; then
        log "LiveCodeBench container probe ok: release=${LCB_RELEASE_VERSION} python=${LCB_PYTHON} jsonl=${LCB_JSONL_DIR}"
    else
        log "ERROR: LiveCodeBench import/local-loader probe failed in container via ${LCB_PYTHON} release=${LCB_RELEASE_VERSION}"
        ok=0
    fi
    for i in "${!LABELS[@]}"; do
        if merged_ready "${MERGED_DIRS[$i]}"; then
            log "merged ready: ${LABELS[$i]} -> ${MERGED_DIRS[$i]}"
            record_status "$i" "${LABELS[$i]}" "livecodebench" "merge-check" "present" "${MERGED_DIRS[$i]}"
        else
            log "merged missing; queue will merge before case: ${LABELS[$i]} -> ${MERGED_DIRS[$i]}"
            record_status "$i" "${LABELS[$i]}" "livecodebench" "merge-check" "missing" "${MERGED_DIRS[$i]}"
        fi
    done
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
    local parquet
    parquet="$(validation_parquet_for_benchmark "${benchmark}")"
    local case_dir="${OUTPUT_ROOT}/${label}/${benchmark}"
    local case_log="${LOG_DIR}/${label}_${benchmark}_n${N_SAMPLES}.log"
    local docker_gpus_arg="${DOCKER_GPUS}"

    if [[ "${docker_gpus_arg}" != "all" && "${docker_gpus_arg}" != \"* ]]; then
        docker_gpus_arg="\"${docker_gpus_arg}\""
    fi

    log "case: index=${i} label=${label} benchmark=${benchmark} actor=${actor} merged=${merged} validation=${parquet} output=${case_dir} official_parallel=${CODE_OFFICIAL_EVAL_PARALLEL}"
    if [ "${SKIP_COMPLETED}" = "1" ] && [ -s "${case_dir}/official_summary.json" ]; then
        log "skip completed case ${i}: ${label}/${benchmark}; summary=${case_dir}/official_summary.json"
        record_status "$i" "$label" "$benchmark" "case" "skipped-completed" "${case_dir}/official_summary.json"
        return
    fi
    if merged_ready "${merged}"; then
        log "merge check before case: present ${merged}"
        record_status "$i" "$label" "$benchmark" "merge-before" "present" "$merged"
    else
        log "merge check before case: missing ${merged}; run_code_offline_eval_case.sh will merge it"
        record_status "$i" "$label" "$benchmark" "merge-before" "missing" "$merged"
    fi
    if [ "${DRY_RUN}" = "1" ]; then
        record_status "$i" "$label" "$benchmark" "case" "dry-run" "$case_dir"
        return
    fi

    wait_for_resources

    local env_prefix
    env_prefix="MODEL_LABEL=${label} BENCHMARK=${benchmark} VALIDATION_PARQUET=${parquet} FSDP_ACTOR_DIR=${actor} MERGED_MODEL_DIR=${merged} OUTPUT_ROOT=${OUTPUT_ROOT} N_SAMPLES=${N_SAMPLES} TEMPERATURE=${TEMPERATURE} TOP_P=${TOP_P} MAX_TOKENS=${MAX_TOKENS} SEED=${SEED} ENABLE_THINKING=${ENABLE_THINKING} TENSOR_PARALLEL=${TENSOR_PARALLEL} GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION} CODE_OFFICIAL_EVAL_PARALLEL=${CODE_OFFICIAL_EVAL_PARALLEL} PROJECT_CACHE_ROOT=${PROJECT_CACHE_ROOT} HF_HOME=${HF_HOME} HF_DATASETS_CACHE=${HF_DATASETS_CACHE} HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE} TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} XDG_CACHE_HOME=${XDG_CACHE_HOME} CODE_OFFICIAL_SOURCE_ROOT=${CODE_OFFICIAL_SOURCE_ROOT} BIGCODEBENCH_OVERRIDE_PATH=${BIGCODEBENCH_OVERRIDE_PATH} LCB_REPO_DIR=${LCB_REPO_DIR} LCB_PYTHON=${LCB_PYTHON} LCB_RELEASE_VERSION=${LCB_RELEASE_VERSION} LCB_JSONL_DIR=${LCB_JSONL_DIR} HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WXPUSHER_NOTIFY=0"
    if [ -n "${CASE_CUDA_VISIBLE_DEVICES}" ]; then
        env_prefix="${env_prefix} CUDA_VISIBLE_DEVICES=${CASE_CUDA_VISIBLE_DEVICES}"
    fi

    record_status "$i" "$label" "$benchmark" "case" "running" "$case_log"
    log "start case ${i}: ${label}/${benchmark}; log=${case_log}"
    if docker run --rm --gpus "${docker_gpus_arg}" --ipc=host --shm-size=64g \
        -v /data-1:/data-1 \
        -v "${EVALPLUS_CACHE_HOST}:/data-1/.cache/evalplus:ro" \
        -v "${REPO_HOST}:${REPO_CONTAINER}" \
        -w "${REPO_CONTAINER}" \
        "${DOCKER_IMAGE}" \
        bash -lc "${env_prefix} bash recipe/on_policy_wdl_sft/code_task/run_code_offline_eval_case.sh" \
        2>&1 | tee "${case_log}"; then
        if merged_ready "${merged}"; then
            record_status "$i" "$label" "$benchmark" "merge-after" "present" "$merged"
        else
            record_status "$i" "$label" "$benchmark" "merge-after" "missing" "$merged"
            log "ERROR: merged model still missing after successful case: ${merged}"
            return 1
        fi
        if [ -s "${case_dir}/official_summary.json" ]; then
            record_status "$i" "$label" "$benchmark" "case" "completed" "${case_dir}/official_summary.json"
            log "complete case ${i}: ${label}/${benchmark}"
        else
            record_status "$i" "$label" "$benchmark" "case" "missing-summary" "${case_dir}/official_summary.json"
            log "ERROR: official summary missing after case: ${case_dir}/official_summary.json"
            return 1
        fi
    else
        record_status "$i" "$label" "$benchmark" "case" "failed" "$case_log"
        log "failed case ${i}: ${label}/${benchmark}"
        notify "KodCode Instruct CTX8K LCB eval failed" "Status: failed\nWhat happened: ${label}/${benchmark} failed.\nEvidence: ${case_log}\nNext action: inspect case log and status file."
        return 1
    fi
}

main() {
    mkdir -p "${LOG_DIR}"
    if [ "${DRY_RUN}" != "1" ]; then
        mkdir -p "${OUTPUT_ROOT}" "${MODEL_WEIGHT_ROOT}"
    fi
    log "KodCode Instruct2507 CTX8K LCB release_v5 N=3 offline eval queue start"
    log "settings: labels=${#LABELS[@]} benchmarks='${BENCHMARKS}' n=${N_SAMPLES} temperature=${TEMPERATURE} top_p=${TOP_P} max_tokens=${MAX_TOKENS} seed=${SEED} enable_thinking=${ENABLE_THINKING} tp=${TENSOR_PARALLEL} official_parallel=${CODE_OFFICIAL_EVAL_PARALLEL} skip_completed=${SKIP_COMPLETED} lcb_release=${LCB_RELEASE_VERSION} output_root=${OUTPUT_ROOT} status=${STATUS_FILE}"
    validate_inputs
    if [ "${DRY_RUN}" != "1" ] && [ "${ALLOW_KODCODE_INSTRUCT2507_CTX8K_LCB_V5_OFFLINE_EVAL:-0}" != "1" ]; then
        log "ERROR: non-dry-run requires ALLOW_KODCODE_INSTRUCT2507_CTX8K_LCB_V5_OFFLINE_EVAL=1"
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
            --benchmarks livecodebench \
            --summary-json "${OUTPUT_ROOT}/summary_kodcode_instruct2507_ctx8k_lcb_v5_n3.json" \
            --summary-md "${OUTPUT_ROOT}/summary_kodcode_instruct2507_ctx8k_lcb_v5_n3.md" \
            --title "KodCode Instruct-2507 CTX8K LiveCodeBench V5 N=3 Offline Eval" \
            --setting-description "Generation setting: LiveCodeBench release_v5, n=3, temperature=1.0, top_p=0.95, max_tokens=4096, seed=42, enable_thinking=true."
    fi
    if [ "${DRY_RUN}" = "1" ]; then
        log "KodCode Instruct2507 CTX8K LCB release_v5 N=3 offline eval queue dry-run complete. Planned results root: ${OUTPUT_ROOT}"
    else
        log "KodCode Instruct2507 CTX8K LCB release_v5 N=3 offline eval queue complete. Results root: ${OUTPUT_ROOT}"
        notify "KodCode Instruct CTX8K LCB eval complete" "Status: completed\nWhat happened: LCB release_v5 N=3 official eval queue finished.\nEvidence: ${OUTPUT_ROOT}/summary_kodcode_instruct2507_ctx8k_lcb_v5_n3.json\nNext action: compare against DeepCoder Instruct2507 R8K LCB."
    fi
}

main "$@"
