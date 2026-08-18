#!/usr/bin/env bash
# Host/container adapter for the frozen Code Stage1 step40 post-fix reevaluation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_HOST=${REPO_HOST:-"$(cd "${SCRIPT_DIR}/../../.." && pwd)"}
REPO_CONTAINER=${REPO_CONTAINER:-/workspace/verl}

if [ -x /opt/venv/bin/python ] && [ -d "${REPO_CONTAINER}" ]; then
    exec python3 "${REPO_CONTAINER}/scripts/code_wdl_stage1_postfix_reevaluation.py" "$@"
fi

export REPO_HOST REPO_CONTAINER
exec /data-1/verl07/run_train.sh \
    python3 "${REPO_CONTAINER}/scripts/code_wdl_stage1_postfix_reevaluation.py" "$@"
