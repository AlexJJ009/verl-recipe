#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data-1/code/verl}
MANIFEST=${MANIFEST:-${REPO_ROOT}/recipe/on_policy_wdl_sft/experiment_manifest/code_qwen3_1p7b_wdl_acd0_p60_beta0.yaml}
cd "$REPO_ROOT"
exec python3 scripts/code_wdl_acd0_queue.py --manifest "$MANIFEST" "$@"
