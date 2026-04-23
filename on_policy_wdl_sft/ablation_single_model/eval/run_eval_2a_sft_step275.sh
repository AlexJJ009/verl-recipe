#!/usr/bin/env bash
# Offline eval for ABL-MINIRL-02 (2A-SFT) step 275 — non-joint single backbone.
set -euo pipefail
export RUN_PREFIX=WDL-SFT-Qwen3-4B-MATH-2A-SFT
export TIMESTAMP=1776892819
export STEP=275
export N_SAMPLES=3
bash "$(dirname "$0")/run_eval_common.sh"
