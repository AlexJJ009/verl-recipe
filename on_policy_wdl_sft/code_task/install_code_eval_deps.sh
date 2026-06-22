#!/usr/bin/env bash
# Install code-eval dependencies inside the active verl-harness Python env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${REQ_FILE:-${SCRIPT_DIR}/requirements-code-eval.txt}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data-1/pip_cache/code_eval}"
REPORT_DIR="${REPORT_DIR:-/data-1/dataset/code/verl_rl/reports}"
INSTALL_LOG="${INSTALL_LOG:-${REPORT_DIR}/code_eval_deps_install.log}"

mkdir -p "$PIP_CACHE_DIR" "$REPORT_DIR"

echo "[install_code_eval_deps] python=$(command -v python3)"
echo "[install_code_eval_deps] requirements=${REQ_FILE}"
echo "[install_code_eval_deps] pip_cache=${PIP_CACHE_DIR}"
echo "[install_code_eval_deps] log=${INSTALL_LOG}"
echo "[install_code_eval_deps] HTTP_PROXY=${HTTP_PROXY:-${http_proxy:-}}"
echo "[install_code_eval_deps] HTTPS_PROXY=${HTTPS_PROXY:-${https_proxy:-}}"

if command -v apt-get >/dev/null 2>&1; then
    echo "[install_code_eval_deps] installing KodCode official sandbox packages: firejail firejail-profiles python3-pytest"
    apt-get update
    apt-get install -y firejail firejail-profiles python3-pytest
else
    echo "[install_code_eval_deps] WARNING: apt-get unavailable; firejail must already be installed for KodCode official reward" >&2
fi

python3 -m pip install --cache-dir "$PIP_CACHE_DIR" -r "$REQ_FILE" 2>&1 | tee "$INSTALL_LOG"
