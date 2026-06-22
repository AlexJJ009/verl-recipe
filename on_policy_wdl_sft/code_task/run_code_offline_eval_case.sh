#!/usr/bin/env bash
# Merge a code-task Stage1 checkpoint if needed, generate with vLLM, convert to
# official evaluator inputs, then run official scoring.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

: "${MODEL_LABEL:?MODEL_LABEL is required}"
: "${FSDP_ACTOR_DIR:?FSDP_ACTOR_DIR is required}"

BENCHMARK=${BENCHMARK:-humaneval}
case "$BENCHMARK" in
    humaneval)
        DATASET_NAME="HumanEval+"
        VALIDATION_PARQUET=${VALIDATION_PARQUET:-/data-1/dataset/code/verl_rl/online_full_humaneval_plus/official_humaneval_plus_val.parquet}
        ;;
    mbpp)
        DATASET_NAME="MBPP+"
        VALIDATION_PARQUET=${VALIDATION_PARQUET:-/data-1/dataset/code/verl_rl/online_full_mbpp_plus/official_mbpp_plus_val.parquet}
        ;;
    bigcodebench)
        DATASET_NAME="BigCodeBench"
        VALIDATION_PARQUET=${VALIDATION_PARQUET:-/data-1/dataset/code/verl_rl/online_full_bigcodebench/official_bigcodebench_val.parquet}
        ;;
    livecodebench)
        DATASET_NAME="LiveCodeBench"
        VALIDATION_PARQUET=${VALIDATION_PARQUET:-/data-1/dataset/code/verl_rl/online_full_livecodebench/official_livecodebench_val.parquet}
        ;;
    *)
        echo "ERROR: unsupported BENCHMARK=${BENCHMARK}" >&2
        exit 2
        ;;
esac

MERGED_MODEL_DIR=${MERGED_MODEL_DIR:-/data-1/model_weights/code_task/offline_eval/${MODEL_LABEL}/step150_actor}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data-1/eval_outputs/code_task/full_official}
CASE_DIR="${OUTPUT_ROOT}/${MODEL_LABEL}/${BENCHMARK}"
N_SAMPLES=${N_SAMPLES:-3}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}
MAX_TOKENS=${MAX_TOKENS:-4096}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-4}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
SEED=${SEED:-42}
ENABLE_THINKING=${ENABLE_THINKING:-}
FORCE_EVAL=${FORCE_EVAL:-0}
ALLOW_EXTRACTION_FAILURES=${ALLOW_EXTRACTION_FAILURES:-1}
CODE_EVAL_OFFICIAL_SITE=${CODE_EVAL_OFFICIAL_SITE:-/data-1/code_eval_envs/official_site}
LCB_REPO_DIR=${LCB_REPO_DIR:-/data-1/code_eval_envs/LiveCodeBench}
LCB_PYTHON=${LCB_PYTHON:-/opt/venv/bin/python}
PROJECT_CACHE_ROOT=${PROJECT_CACHE_ROOT:-/data-1/.cache}
HF_HOME=${CODE_TASK_HF_HOME:-$PROJECT_CACHE_ROOT/huggingface}
HF_DATASETS_CACHE=${CODE_TASK_HF_DATASETS_CACHE:-$HF_HOME/datasets}
HUGGINGFACE_HUB_CACHE=${CODE_TASK_HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
TRANSFORMERS_CACHE=${CODE_TASK_TRANSFORMERS_CACHE:-$HF_HOME}
XDG_CACHE_HOME=${CODE_TASK_XDG_CACHE_HOME:-$PROJECT_CACHE_ROOT}
CODE_OFFICIAL_SOURCE_ROOT=${CODE_OFFICIAL_SOURCE_ROOT:-/data-1/dataset/code/official_sources}
BIGCODEBENCH_OVERRIDE_PATH=${BIGCODEBENCH_OVERRIDE_PATH:-$CODE_OFFICIAL_SOURCE_ROOT/bigcodebench/BigCodeBench-v0.1.4.jsonl}
REPO_PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
OFFICIAL_PYTHONPATH="${REPO_ROOT}:${CODE_EVAL_OFFICIAL_SITE}:${LCB_REPO_DIR}:${PYTHONPATH:-}"
export PROJECT_CACHE_ROOT HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME
export CODE_OFFICIAL_SOURCE_ROOT BIGCODEBENCH_OVERRIDE_PATH HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

notify() {
    local title="$1" body="$2"
    if [ "${WXPUSHER_NOTIFY:-1}" = "1" ]; then
        python3 /root/agent-core/skills/wxpusher-notify/scripts/wxpusher_notify.py \
            --title "$title" \
            --body "$body" || true
    fi
}

mkdir -p "${CASE_DIR}" "${MERGED_MODEL_DIR}"
trap 'notify "Code offline eval failed" "Status: failed\nWhat happened: ${MODEL_LABEL}/${BENCHMARK} exited with error.\nEvidence: ${CASE_DIR}\nNext action: inspect case log and summary files."' ERR

echo "[code-offline-eval] MODEL_LABEL=${MODEL_LABEL}"
echo "[code-offline-eval] BENCHMARK=${BENCHMARK} DATASET_NAME=${DATASET_NAME}"
echo "[code-offline-eval] VALIDATION_PARQUET=${VALIDATION_PARQUET}"
echo "[code-offline-eval] FSDP_ACTOR_DIR=${FSDP_ACTOR_DIR}"
echo "[code-offline-eval] MERGED_MODEL_DIR=${MERGED_MODEL_DIR}"
echo "[code-offline-eval] CASE_DIR=${CASE_DIR}"
echo "[code-offline-eval] PROJECT_CACHE_ROOT=${PROJECT_CACHE_ROOT}"
echo "[code-offline-eval] HF_HOME=${HF_HOME}"
echo "[code-offline-eval] HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
echo "[code-offline-eval] HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}"
echo "[code-offline-eval] XDG_CACHE_HOME=${XDG_CACHE_HOME}"
echo "[code-offline-eval] CODE_OFFICIAL_SOURCE_ROOT=${CODE_OFFICIAL_SOURCE_ROOT}"
echo "[code-offline-eval] BIGCODEBENCH_OVERRIDE_PATH=${BIGCODEBENCH_OVERRIDE_PATH}"
echo "[code-offline-eval] LCB_REPO_DIR=${LCB_REPO_DIR}"
echo "[code-offline-eval] LCB_PYTHON=${LCB_PYTHON}"
echo "[code-offline-eval] ENABLE_THINKING=${ENABLE_THINKING:-<default>}"
echo "[code-offline-eval] ALLOW_EXTRACTION_FAILURES=${ALLOW_EXTRACTION_FAILURES}"

if [ ! -f "${MERGED_MODEL_DIR}/model.safetensors.index.json" ] && [ ! -f "${MERGED_MODEL_DIR}/model.safetensors" ]; then
    CUDA_VISIBLE_DEVICES="${MERGE_CUDA_VISIBLE_DEVICES:-0}" PYTHONPATH="${REPO_PYTHONPATH}" python3 -u -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${FSDP_ACTOR_DIR}" \
        --target_dir "${MERGED_MODEL_DIR}" \
        --trust-remote-code
else
    echo "[code-offline-eval] merged model exists; skip merge"
fi

RAW_OUTPUTS="${CASE_DIR}/raw_generations_n${N_SAMPLES}.jsonl"
CONVERTED_OUTPUT="${CASE_DIR}/${BENCHMARK}_samples_n${N_SAMPLES}"
if [ "$BENCHMARK" = "livecodebench" ]; then
    CONVERTED_OUTPUT="${CONVERTED_OUTPUT}.json"
else
    CONVERTED_OUTPUT="${CONVERTED_OUTPUT}.jsonl"
fi

if [ "${FORCE_EVAL}" = "1" ] || [ ! -s "${RAW_OUTPUTS}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        export CUDA_VISIBLE_DEVICES
    fi
    THINKING_ARGS=()
    if [ -n "${ENABLE_THINKING}" ]; then
        THINKING_ARGS+=(--enable-thinking "${ENABLE_THINKING}")
    fi
    PYTHONPATH="${REPO_PYTHONPATH}" python3 -u "${SCRIPT_DIR}/eval_code_vllm.py" \
        --model "${MERGED_MODEL_DIR}" \
        --validation-parquet "${VALIDATION_PARQUET}" \
        --output "${RAW_OUTPUTS}" \
        --summary "${CASE_DIR}/generation_summary.json" \
        --tensor-parallel "${TENSOR_PARALLEL}" \
        --n "${N_SAMPLES}" \
        --temperature "${TEMPERATURE}" \
        --top-p "${TOP_P}" \
        --max-tokens "${MAX_TOKENS}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --seed "${SEED}" \
        "${THINKING_ARGS[@]}"
else
    echo "[code-offline-eval] raw generations exist; skip generation"
fi

CONVERT_ARGS=()
if [ "${ALLOW_EXTRACTION_FAILURES}" = "1" ]; then
    CONVERT_ARGS+=(--allow-extraction-failures)
fi
PYTHONPATH="${REPO_PYTHONPATH}" python3 -u "${SCRIPT_DIR}/convert_official_outputs.py" \
    --raw-outputs "${RAW_OUTPUTS}" \
    --validation-parquet "${VALIDATION_PARQUET}" \
    --output "${CONVERTED_OUTPUT}" \
    --benchmark "${BENCHMARK}" \
    --report "${CASE_DIR}/conversion_report.json" \
    "${CONVERT_ARGS[@]}"

if [ "$BENCHMARK" = "livecodebench" ]; then
    PYTHONPATH="${OFFICIAL_PYTHONPATH}" python3 -u "${SCRIPT_DIR}/eval_code_official.py" \
        --benchmark livecodebench \
        --custom-output "${CONVERTED_OUTPUT}" \
        --output-dir "${CASE_DIR}/official" \
        --summary "${CASE_DIR}/official_summary.json" \
        --parallel "${CODE_OFFICIAL_EVAL_PARALLEL:-8}" \
        --lcb-python "${LCB_PYTHON}" \
        --overwrite
else
    EXTRA_OFFICIAL_ARGS=()
    if [ "$BENCHMARK" = "bigcodebench" ]; then
        EXTRA_OFFICIAL_ARGS+=(--bcb-override-path "${BIGCODEBENCH_OVERRIDE_PATH}")
    fi
    PYTHONPATH="${OFFICIAL_PYTHONPATH}" python3 -u "${SCRIPT_DIR}/eval_code_official.py" \
        --benchmark "${BENCHMARK}" \
        --samples "${CONVERTED_OUTPUT}" \
        --output-dir "${CASE_DIR}/official" \
        --summary "${CASE_DIR}/official_summary.json" \
        --parallel "${CODE_OFFICIAL_EVAL_PARALLEL:-8}" \
        "${EXTRA_OFFICIAL_ARGS[@]}" \
        --overwrite
fi

notify "Code offline eval completed" "Status: completed\nWhat happened: ${MODEL_LABEL}/${BENCHMARK} finished official scoring.\nEvidence: ${CASE_DIR}/official_summary.json\nNext action: review aggregate results."
echo "[code-offline-eval] complete: ${CASE_DIR}/official_summary.json"
