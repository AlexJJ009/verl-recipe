#!/usr/bin/env bash
# Direct Meituan-worker entry for the user-approved external provenance assumption.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT=${1:-${EXPERIMENT:-}}
RLVR_SEED=${2:-${RLVR_SEED:-20260727}}

: "${ROOT:?ROOT must point to the colleague persistent Meituan root}"
# Reuse the controlled multi-root adapter so train/eval/output/cache paths match
# manifest-launched workers, while leaving platform receipt validation disabled.
export REQUIRE_PLATFORM_RECEIPTS=0
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/meituan/env.sh"
if [ -z "$EXPERIMENT" ]; then
    echo "ERROR: usage: ROOT=/path bash ${BASH_SOURCE[0]} R01|R02 [RLVR_SEED]" >&2
    exit 2
fi
case "$RLVR_SEED" in
    20260727|20260728|20260729) ;;
    *) echo "ERROR: RLVR_SEED must be one of 20260727, 20260728, 20260729" >&2; exit 2 ;;
esac

MODEL_ROOT=${MODEL_ROOT:-"${ROOT}/models/rebuttal_rlvr/init"}
R01_MODEL_PATH=${R01_MODEL_PATH:-"${MODEL_ROOT}/R01_ORDINARY_SFT_4B_AM1P4M"}
R02_MODEL_PATH=${R02_MODEL_PATH:-"${MODEL_ROOT}/R02_WDL_SFT_4B_AM1P4M"}
case "$EXPERIMENT" in
    R01) selected_model="$R01_MODEL_PATH" ;;
    R02) selected_model="$R02_MODEL_PATH" ;;
    *) echo "ERROR: direct colleague entry supports only R01 or R02" >&2; exit 2 ;;
esac

if [ ! -d "$selected_model" ]; then
    echo "ERROR: $EXPERIMENT model directory is missing: $selected_model" >&2
    exit 2
fi
if [ ! -f "$selected_model/config.json" ]; then
    echo "ERROR: $EXPERIMENT config.json is missing: $selected_model/config.json" >&2
    exit 2
fi
if ! find "$selected_model" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit | grep -q .; then
    echo "ERROR: $EXPERIMENT has no loadable weight file under $selected_model" >&2
    exit 2
fi

python3 - "$selected_model/config.json" <<'PY'
import json
import pathlib
import sys

config = json.loads(pathlib.Path(sys.argv[1]).read_text())
expected = {"model_type": "qwen3", "hidden_size": 2560, "num_hidden_layers": 36}
mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
if mismatches:
    raise SystemExit(f"expected Qwen3-4B config, mismatches={mismatches}")
PY

if [ "$EXPERIMENT" = "R02" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    if [ ! -f "$selected_model/pytorch_model.bin" ]; then
        echo "ERROR: R02 requires the pinned public pytorch_model.bin" >&2
        exit 2
    fi
    expected_bytes=8045067711
    expected_sha=3267350ca2bc2325c24b9b7d98852b776568a759c3d13242dd3cc7db1442d6b9
    actual_bytes=$(stat -c %s "$selected_model/pytorch_model.bin")
    actual_sha=$(sha256sum "$selected_model/pytorch_model.bin" | awk '{print $1}')
    if [ "$actual_bytes" != "$expected_bytes" ] || [ "$actual_sha" != "$expected_sha" ]; then
        echo "ERROR: R02 public weight identity differs from the pinned revision" >&2
        exit 2
    fi
fi

if [ "${DRY_RUN:-0}" != "1" ]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: nvidia-smi is required for a real colleague launch" >&2
        exit 2
    fi
    mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
    if [ "${#gpu_names[@]}" -ne 8 ]; then
        echo "ERROR: direct launch requires exactly 8 GPUs; found ${#gpu_names[@]}" >&2
        exit 2
    fi
    for gpu_name in "${gpu_names[@]}"; do
        if [[ "$gpu_name" != *H20* ]]; then
            echo "ERROR: direct launch requires H20 GPUs; found $gpu_name" >&2
            exit 2
        fi
    done
fi

export EXPERIMENT RLVR_SEED
export INIT_PAIR=${INIT_PAIR:-I1}
export RUN_MODE=external_checkpoint_assumption
export EXTERNAL_PROVENANCE_ASSUMPTION=user_approved_unrecoverable_public_checkpoint_metadata_20260728
export MODEL_ROOT
export ORDINARY_SFT_4B_MODEL_PATH="$R01_MODEL_PATH"
export WDL_4B_MODEL_PATH="$R02_MODEL_PATH"

echo "Launching ${EXPERIMENT} with conditional checkpoint comparison policy"
echo "model=${selected_model} seed=${RLVR_SEED} root=${ROOT}"
exec bash "${SCRIPT_DIR}/run_experiment.sh" "$EXPERIMENT"
