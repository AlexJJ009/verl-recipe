#!/usr/bin/env bash
# Compatibility shim for staged_v1 Stage2 common. Code-task provenance is
# checked in run_s2_code_model2_rollout_common.sh before this resolver runs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../staged_v1/_resolve_stage1_model2.sh"
