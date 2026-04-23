#!/usr/bin/env bash
# Offline eval for ABL-MINIRL-01 (2Z-SFT) step 275 — non-joint.
set -euo pipefail
export RUN_PREFIX=MINIRL-Qwen3-4B-MATH-2Z-SFT
export TIMESTAMP=1776855436
export STEP=275
export N_SAMPLES=3
bash "$(dirname "$0")/run_eval_common.sh"
