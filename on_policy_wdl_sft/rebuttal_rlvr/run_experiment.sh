#!/usr/bin/env bash
# Unified relative-path entrypoint. Example: bash run_experiment.sh R02

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT=${1:-${EXPERIMENT:-}}

if [ -z "$EXPERIMENT" ]; then
    echo "ERROR: pass an experiment ID: R01, R02, or R03" >&2
    exit 2
fi

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/model_paths.env"

case "$EXPERIMENT" in
    R01)
        expected_arm=sft
        run_script="${SCRIPT_DIR}/run_math_sft.sh"
        if [ "${REQUIRE_PLATFORM_RECEIPTS:-0}" != "1" ]; then
            export INIT_MODEL_PATH=${INIT_MODEL_PATH:-"$ORDINARY_SFT_4B_MODEL_PATH"}
            if [ ! -d "$INIT_MODEL_PATH" ]; then
                echo "ERROR: R01 is pre-registered but unresolved: ${ORDINARY_SFT_4B_MODEL_NAME}" >&2
                echo "EXPECTED_PATH: ${INIT_MODEL_PATH}" >&2
                echo "ACTION: place the colleague-provided ordinary-SFT 4B AM-1.4M export there or override ORDINARY_SFT_4B_MODEL_PATH." >&2
                exit 2
            fi
        fi
        ;;
    R02)
        expected_arm=wdl
        run_script="${SCRIPT_DIR}/run_math_wdl.sh"
        if [ "${REQUIRE_PLATFORM_RECEIPTS:-0}" != "1" ]; then
            export INIT_MODEL_PATH=${INIT_MODEL_PATH:-"$WDL_4B_MODEL_PATH"}
        fi
        ;;
    R03)
        expected_arm=wdl
        run_script="${SCRIPT_DIR}/run_math_wdl.sh"
        if [ "${REQUIRE_PLATFORM_RECEIPTS:-0}" != "1" ]; then
            : "${WDL_8B_MODEL_ID:?R03 is blocked until the public 8B model ID is provided}"
            : "${WDL_8B_REVISION:?R03 is blocked until the public 8B revision is provided}"
            : "${WDL_8B_MODEL_PATH:?R03 is blocked until the downloaded 8B model path is provided}"
            export INIT_MODEL_PATH="$WDL_8B_MODEL_PATH"
        fi
        ;;
    *)
        echo "ERROR: unknown EXPERIMENT=$EXPERIMENT; expected R01, R02, or R03" >&2
        exit 2
        ;;
esac

if [ -n "${ARM:-}" ] && [ "$ARM" != "$expected_arm" ]; then
    echo "ERROR: EXPERIMENT=$EXPERIMENT maps to ARM=$expected_arm, got ARM=$ARM" >&2
    exit 2
fi

export EXPERIMENT
export ARM="$expected_arm"

if [ "${REQUIRE_PLATFORM_RECEIPTS:-0}" = "1" ]; then
    : "${INIT_MODEL_PATH:?platform launch requires manifest-bound INIT_MODEL_PATH}"
fi
if [ ! -f "$run_script" ]; then
    echo "ERROR: relative training wrapper is missing: $run_script" >&2
    exit 2
fi

exec bash "$run_script"
