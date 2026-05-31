#!/usr/bin/env bash
# Merge a Stage 1 checkpoint into a fixed Model2 directory.

set -euo pipefail

WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WRAPPER_SCRIPT_DIR

: "${STAGE1_RUN_PREFIX:?STAGE1_RUN_PREFIX must be set}"
: "${STAGE1_CKPT_DIR:?STAGE1_CKPT_DIR must be set}"
: "${STAGE1_STEP:?STAGE1_STEP must be set}"
: "${MERGED_MODEL2_DIR:?MERGED_MODEL2_DIR must be set}"

export REQUIRE_MERGED_MODEL2_PROVENANCE=${REQUIRE_MERGED_MODEL2_PROVENANCE:-True}

# shellcheck disable=SC1091
source "${WRAPPER_SCRIPT_DIR}/_resolve_stage1_model2.sh"
