#!/usr/bin/env bash
# Evaluate raw Qwen3-1.7B with the same code-task settings as the cold-start
# SFT fraction sweep.
set -euo pipefail

REPO_HOST=${REPO_HOST:-/data-1/verl07/verl}
REPO_CONTAINER=${REPO_CONTAINER:-/workspace/verl}
DOCKER_IMAGE=${DOCKER_IMAGE:-verl-harness:latest}
MODEL=${MODEL:-/data-1/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data-1/eval_outputs/code_task/qwen3_1p7b_coldstart_sft_fraction/raw_base}
LOG_FILE=${LOG_FILE:-"${REPO_HOST}/recipe/on_policy_wdl_sft/code_task/qwen3_1p7b_raw_base_coldstart_comparison_eval.log"}

EVAL_BENCHMARKS=${EVAL_BENCHMARKS:-"humaneval mbpp livecodebench"}
EVAL_N_SAMPLES=${EVAL_N_SAMPLES:-1}
EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-0.2}
EVAL_TOP_P=${EVAL_TOP_P:-0.95}
EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-4096}
EVAL_TENSOR_PARALLEL=${EVAL_TENSOR_PARALLEL:-4}
EVAL_GPU_MEMORY_UTILIZATION=${EVAL_GPU_MEMORY_UTILIZATION:-0.75}
CODE_OFFICIAL_EVAL_PARALLEL=${CODE_OFFICIAL_EVAL_PARALLEL:-8}
LCB_RELEASE_VERSION=${LCB_RELEASE_VERSION:-release_v5}

if [ "${DRY_RUN:-0}" != "1" ] && [ "${ALLOW_QWEN3_1P7B_RAW_BASE_COLDSTART_COMPARISON_EVAL:-0}" != "1" ]; then
    echo "[raw-base-eval] ERROR: non-dry-run requires ALLOW_QWEN3_1P7B_RAW_BASE_COLDSTART_COMPARISON_EVAL=1" >&2
    exit 1
fi

log() {
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

run_eval() {
    local benchmark="$1"
    local validation converted_ext case_dir raw converted
    case "$benchmark" in
        humaneval)
            validation=/data-1/dataset/code/verl_rl/online_full_humaneval_plus/official_humaneval_plus_val.parquet
            converted_ext=jsonl
            ;;
        mbpp)
            validation=/data-1/dataset/code/verl_rl/online_full_mbpp_plus/official_mbpp_plus_val.parquet
            converted_ext=jsonl
            ;;
        livecodebench)
            case "$LCB_RELEASE_VERSION" in
                release_v1) validation=/data-1/dataset/code/verl_rl/online_full_livecodebench/official_livecodebench_val.parquet ;;
                release_v5) validation=/data-1/dataset/code/verl_rl/online_full_livecodebench_v5/official_livecodebench_val.parquet ;;
                *) validation="/data-1/dataset/code/verl_rl/online_full_livecodebench_${LCB_RELEASE_VERSION}/official_livecodebench_val.parquet" ;;
            esac
            converted_ext=json
            ;;
        *)
            echo "unsupported benchmark: $benchmark" >&2
            exit 2
            ;;
    esac

    case_dir="${OUTPUT_ROOT}/${benchmark}"
    raw="${case_dir}/raw_generations_n${EVAL_N_SAMPLES}.jsonl"
    converted="${case_dir}/${benchmark}_samples_n${EVAL_N_SAMPLES}.${converted_ext}"
    mkdir -p "$case_dir"

    if [ "${DRY_RUN:-0}" = "1" ]; then
        log "DRY_RUN would eval raw_base/${benchmark}: model=${MODEL} validation=${validation} output=${case_dir}"
        return
    fi

    log "eval start raw_base/${benchmark}: model=${MODEL}"
    docker run --rm --gpus all --ipc=host --network=host --shm-size=64g \
        -v /data-1:/data-1 \
        -v "$REPO_HOST":"$REPO_CONTAINER" \
        -w "$REPO_CONTAINER" \
        "$DOCKER_IMAGE" \
        bash -lc "set -euo pipefail
            export PROJECT_CACHE_ROOT=/data-1/.cache
            export HF_HOME=/data-1/.cache/huggingface
            export HF_DATASETS_CACHE=/data-1/.cache/huggingface/datasets
            export HUGGINGFACE_HUB_CACHE=/data-1/.cache/huggingface/hub
            export TRANSFORMERS_CACHE=/data-1/.cache/huggingface
            export XDG_CACHE_HOME=/data-1/.cache
            export CODE_OFFICIAL_SOURCE_ROOT=/data-1/dataset/code/official_sources
            export BIGCODEBENCH_OVERRIDE_PATH=/data-1/dataset/code/official_sources/bigcodebench/BigCodeBench-v0.1.4.jsonl
            export HF_HUB_OFFLINE=1
            export HF_DATASETS_OFFLINE=1
            PYTHONPATH='$REPO_CONTAINER':\${PYTHONPATH:-} python3 -u recipe/on_policy_wdl_sft/code_task/eval_code_vllm.py \
                --model '$MODEL' \
                --validation-parquet '$validation' \
                --output '$raw' \
                --summary '$case_dir/generation_summary.json' \
                --tensor-parallel '$EVAL_TENSOR_PARALLEL' \
                --n '$EVAL_N_SAMPLES' \
                --temperature '$EVAL_TEMPERATURE' \
                --top-p '$EVAL_TOP_P' \
                --max-tokens '$EVAL_MAX_TOKENS' \
                --gpu-memory-utilization '$EVAL_GPU_MEMORY_UTILIZATION' \
                --seed 42
            PYTHONPATH='$REPO_CONTAINER':\${PYTHONPATH:-} python3 -u recipe/on_policy_wdl_sft/code_task/convert_official_outputs.py \
                --raw-outputs '$raw' \
                --validation-parquet '$validation' \
                --output '$converted' \
                --benchmark '$benchmark' \
                --report '$case_dir/conversion_report.json' \
                --allow-extraction-failures
            if [ '$benchmark' = livecodebench ]; then
                PYTHONPATH='$REPO_CONTAINER':/data-1/code_eval_envs/official_site:/data-1/code_eval_envs/LiveCodeBench:\${PYTHONPATH:-} python3 -u recipe/on_policy_wdl_sft/code_task/eval_code_official.py \
                    --benchmark livecodebench \
                    --custom-output '$converted' \
                    --output-dir '$case_dir/official' \
                    --summary '$case_dir/official_summary.json' \
                    --parallel '$CODE_OFFICIAL_EVAL_PARALLEL' \
                    --lcb-python /opt/venv/bin/python \
                    --lcb-release-version '$LCB_RELEASE_VERSION' \
                    --overwrite
            else
                PYTHONPATH='$REPO_CONTAINER':/data-1/code_eval_envs/official_site:/data-1/code_eval_envs/LiveCodeBench:\${PYTHONPATH:-} python3 -u recipe/on_policy_wdl_sft/code_task/eval_code_official.py \
                    --benchmark '$benchmark' \
                    --samples '$converted' \
                    --output-dir '$case_dir/official' \
                    --summary '$case_dir/official_summary.json' \
                    --parallel '$CODE_OFFICIAL_EVAL_PARALLEL' \
                    --overwrite
            fi" 2>&1 | tee -a "$LOG_FILE"
    log "eval complete raw_base/${benchmark}: ${case_dir}/official_summary.json"
}

log "raw Qwen3-1.7B cold-start comparison eval start: output=${OUTPUT_ROOT}"
for benchmark in $EVAL_BENCHMARKS; do
    run_eval "$benchmark"
done
log "raw Qwen3-1.7B cold-start comparison eval complete"
