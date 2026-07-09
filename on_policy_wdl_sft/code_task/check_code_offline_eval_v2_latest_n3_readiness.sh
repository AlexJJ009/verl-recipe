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
LCB_RELEASE_VERSION=${LCB_RELEASE_VERSION:-release_v5}
LCB_JSONL_DIR=${LCB_JSONL_DIR:-$PROJECT_CACHE_ROOT/huggingface/hub/datasets--livecodebench--code_generation_lite/snapshots/0fe84c3912ea0c4d4a78037083943e8f0c4dd505}
DOCKER_IMAGE=${DOCKER_IMAGE:-verl-harness}
MIN_FREE_GB=${MIN_FREE_GB:-80}

case "${LCB_RELEASE_VERSION}" in
    release_v1)
        LCB_VALIDATION_PARQUET=${LCB_VALIDATION_PARQUET:-/data-1/dataset/code/verl_rl/online_full_livecodebench/official_livecodebench_val.parquet}
        ;;
    release_v5)
        LCB_VALIDATION_PARQUET=${LCB_VALIDATION_PARQUET:-/data-1/dataset/code/verl_rl/online_full_livecodebench_v5/official_livecodebench_val.parquet}
        ;;
    *)
        LCB_VALIDATION_PARQUET=${LCB_VALIDATION_PARQUET:-/data-1/dataset/code/verl_rl/online_full_livecodebench_${LCB_RELEASE_VERSION}/official_livecodebench_val.parquet}
        ;;
esac

paths=(
    "/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V2_1780685616/global_step_150/actor"
    "/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V2_1780736174/global_step_150/actor"
    "/data-1/dataset/code/verl_rl/online_full_humaneval_plus/official_humaneval_plus_val.parquet"
    "/data-1/dataset/code/verl_rl/online_full_mbpp_plus/official_mbpp_plus_val.parquet"
    "/data-1/dataset/code/verl_rl/online_full_bigcodebench/official_bigcodebench_val.parquet"
    "${LCB_VALIDATION_PARQUET}"
    "${EVALPLUS_CACHE_HOST}"
    "${BIGCODEBENCH_OVERRIDE_PATH}"
    "${CODE_EVAL_OFFICIAL_SITE}"
    "${LCB_REPO_DIR}"
    "${LCB_JSONL_DIR}"
)

status=0
echo "[readiness] repo=${REPO_HOST}"
echo "[readiness] min_free_gb=${MIN_FREE_GB}"
echo "[readiness] lcb_release=${LCB_RELEASE_VERSION}"
echo "[readiness] lcb_validation_parquet=${LCB_VALIDATION_PARQUET}"
echo "[readiness] lcb_jsonl_dir=${LCB_JSONL_DIR}"
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
    --env LCB_RELEASE_VERSION="${LCB_RELEASE_VERSION}" \
    --env LCB_JSONL_DIR="${LCB_JSONL_DIR}" \
    --env PROJECT_CACHE_ROOT="${PROJECT_CACHE_ROOT}" \
    --env HF_HOME="${PROJECT_CACHE_ROOT}/huggingface" \
    --env HF_DATASETS_CACHE="${PROJECT_CACHE_ROOT}/huggingface/datasets" \
    --env HUGGINGFACE_HUB_CACHE="${PROJECT_CACHE_ROOT}/huggingface/hub" \
    --env TRANSFORMERS_CACHE="${PROJECT_CACHE_ROOT}/huggingface" \
    --env XDG_CACHE_HOME="${PROJECT_CACHE_ROOT}" \
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
rows = []
total = 0
for name in files:
    path = root / name
    assert path.is_file(), path
    with path.open(encoding="utf-8") as f:
        first_line = ""
        for line in f:
            if line.strip():
                total += 1
                if not first_line:
                    first_line = line
        assert first_line, path
        rows.append(CodeGenerationProblem(**json.loads(first_line)))
assert rows and total > 0, "empty LiveCodeBench local JSONL loader"
print(total, rows[0].question_id, rows[-1].question_id)
PY' \
    >/dev/null 2>&1; then
    echo "[readiness] OK LiveCodeBench custom evaluator and local ${LCB_RELEASE_VERSION} JSONL loader in container via ${LCB_PYTHON}"
else
    echo "[readiness] ERROR: LiveCodeBench import/local-loader probe failed in container via ${LCB_PYTHON} release=${LCB_RELEASE_VERSION}" >&2
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
