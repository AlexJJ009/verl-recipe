#!/usr/bin/env bash
# Read-only readiness check for the V2 latest unified N=3 code offline eval queue.
set -euo pipefail

REPO_HOST=${REPO_HOST:-/root/buaa/local_data1/verl07/verl}
PROJECT_CACHE_ROOT=${PROJECT_CACHE_ROOT:-/data-1/.cache}
EVALPLUS_CACHE_HOST=${EVALPLUS_CACHE_HOST:-$PROJECT_CACHE_ROOT/evalplus}
CODE_OFFICIAL_SOURCE_ROOT=${CODE_OFFICIAL_SOURCE_ROOT:-/data-1/dataset/code/official_sources}
BIGCODEBENCH_OVERRIDE_PATH=${BIGCODEBENCH_OVERRIDE_PATH:-$CODE_OFFICIAL_SOURCE_ROOT/bigcodebench/BigCodeBench-v0.1.4.jsonl}
CODE_EVAL_OFFICIAL_SITE=${CODE_EVAL_OFFICIAL_SITE:-/data-1/code_eval_envs/official_site}
LCB_REPO_DIR=${LCB_REPO_DIR:-/data-1/code_eval_envs/LiveCodeBench}
LCB_PYTHON=${LCB_PYTHON:-/opt/venv/bin/python}
DOCKER_IMAGE=${DOCKER_IMAGE:-verl-harness}
MIN_FREE_GB=${MIN_FREE_GB:-80}

paths=(
    "/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V2_1780685616/global_step_150/actor"
    "/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V2_1780736174/global_step_150/actor"
    "/data-1/dataset/code/verl_rl/online_full_humaneval_plus/official_humaneval_plus_val.parquet"
    "/data-1/dataset/code/verl_rl/online_full_mbpp_plus/official_mbpp_plus_val.parquet"
    "/data-1/dataset/code/verl_rl/online_full_bigcodebench/official_bigcodebench_val.parquet"
    "/data-1/dataset/code/verl_rl/online_full_livecodebench/official_livecodebench_val.parquet"
    "${EVALPLUS_CACHE_HOST}"
    "${BIGCODEBENCH_OVERRIDE_PATH}"
    "${CODE_EVAL_OFFICIAL_SITE}"
    "${LCB_REPO_DIR}"
)

status=0
echo "[readiness] repo=${REPO_HOST}"
echo "[readiness] min_free_gb=${MIN_FREE_GB}"
free_gb=$(df -Pk /data-1 | awk 'NR==2 {print int($4 / 1024 / 1024)}')
echo "[readiness] /data-1 free=${free_gb}G"
if [ "${free_gb}" -lt "${MIN_FREE_GB}" ]; then
    echo "[readiness] ERROR: /data-1 free space below threshold" >&2
    status=1
fi

for path in "${paths[@]}"; do
    if [ -e "${path}" ]; then
        echo "[readiness] OK ${path}"
    else
        echo "[readiness] MISSING ${path}" >&2
        status=1
    fi
done

if docker image inspect "${DOCKER_IMAGE}" >/dev/null 2>&1; then
    echo "[readiness] OK docker image ${DOCKER_IMAGE}"
else
    echo "[readiness] MISSING docker image ${DOCKER_IMAGE}" >&2
    status=1
fi

if docker run --rm \
    -v /data-1:/data-1 \
    -v "${REPO_HOST}:/workspace/verl" \
    -w /workspace/verl \
    "${DOCKER_IMAGE}" \
    bash -lc "PYTHONPATH=/workspace/verl:${CODE_EVAL_OFFICIAL_SITE}:${LCB_REPO_DIR}:\${PYTHONPATH:-} python3 -c 'import evalplus.evaluate, bigcodebench.evaluate'" \
    >/dev/null 2>&1; then
    echo "[readiness] OK container imports evalplus.evaluate and bigcodebench.evaluate"
else
    echo "[readiness] ERROR: container import probe failed for evalplus/bigcodebench" >&2
    status=1
fi

if docker run --rm \
    -v /data-1:/data-1 \
    -v "${REPO_HOST}:/workspace/verl" \
    -w /workspace/verl \
    --env LCB_REPO_DIR="${LCB_REPO_DIR}" \
    --env LCB_PYTHON="${LCB_PYTHON}" \
    "${DOCKER_IMAGE}" \
    bash -lc 'test -x "${LCB_PYTHON}" && cd "${LCB_REPO_DIR}" && PYTHONPATH="${LCB_REPO_DIR}:${PYTHONPATH:-}" "${LCB_PYTHON}" -c "import lcb_runner.runner.custom_evaluator"' \
    >/dev/null 2>&1; then
    echo "[readiness] OK LiveCodeBench custom evaluator import in container via ${LCB_PYTHON}"
else
    echo "[readiness] ERROR: LiveCodeBench import probe failed in container via ${LCB_PYTHON}" >&2
    status=1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[readiness] GPU summary:"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
else
    echo "[readiness] WARNING: nvidia-smi not found" >&2
fi

if [ "${status}" -eq 0 ]; then
    echo "[readiness] PASS"
else
    echo "[readiness] FAIL" >&2
fi
exit "${status}"
