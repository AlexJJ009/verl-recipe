#!/usr/bin/env bash
set -euo pipefail

: "${ROOT:?ROOT must be exported by the platform shim}"
PLATFORM_MODE=${REQUIRE_PLATFORM_RECEIPTS:-0}

if [ "$PLATFORM_MODE" = "1" ]; then
    : "${DATASET_ROOT:?platform env requires manifest-bound DATASET_ROOT}"
    : "${MODEL_ROOT:?platform env requires manifest-bound MODEL_ROOT}"
    : "${STATE_ROOT:?platform env requires manifest-bound STATE_ROOT}"
    # A formal worker receives only manifest-bound controlled roots and input artifacts.
    # Do not preserve arbitrary values inherited from the Hope client/image.
    export DATASET_ROOT MODEL_ROOT STATE_ROOT
    export OUTPUT_ROOT="$STATE_ROOT/verl-exp"
    export BASE_CKPT_DIR="$OUTPUT_ROOT/checkpoints/rebuttal_rlvr"
    export EVAL_ROOT="$OUTPUT_ROOT/eval/rebuttal_rlvr"
    export LOG_DIR="$OUTPUT_ROOT/logs/rebuttal_rlvr"
    export WANDB_DIR="$OUTPUT_ROOT/wandb_runs/rebuttal_rlvr"
    export RECEIPT_ROOT="$OUTPUT_ROOT/receipts/rebuttal_rlvr"
    export REGISTRY_ROOT="$STATE_ROOT/experiment_registry"
    export EXPERIMENT_REGISTRY_DB="$REGISTRY_ROOT/experiment_registry.sqlite"
    export TRAINING_RELEASE_GATE_STATE="$REGISTRY_ROOT/training_release_gate.jsonl"
    unset RELEASE_LOG_FILE RELEASE_STATUS_FILE VERL_FILE_LOGGER_ROOT

    export HF_HOME="$OUTPUT_ROOT/cache/hf"
    export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
    export HF_DATASETS_CACHE="$OUTPUT_ROOT/cache/datasets"
    export XDG_CACHE_HOME="$OUTPUT_ROOT/cache/xdg"
    export RAY_TMPDIR="/tmp/rebuttal_rlvr/ray"
    export TMPDIR="/tmp/rebuttal_rlvr/tmp"
    export VLLM_CONFIG_ROOT="/tmp/rebuttal_rlvr/vllm"
    export VERL_ZMQ_IPC_DIR="/tmp/rebuttal_rlvr/zmq"

    export TRAIN_FILE="$DATASET_ROOT/data/math/train_rl_format.parquet"
    export MATH7_AIME_FILE="$DATASET_ROOT/data/math7/AIME-2025/aime-2025_with_system_prompt.parquet"
    export MATH7_MATH500_FILE="$DATASET_ROOT/data/math7/MATH-500/math500-test_with_system_prompt.parquet"
    export MATH7_AMC23_FILE="$DATASET_ROOT/data/math7/AMC23/amc23-test_with_system_prompt.parquet"
    export MATH7_AQUA_FILE="$DATASET_ROOT/data/math7/AQUA/aqua-test_with_system_prompt.parquet"
    export MATH7_GSM8K_FILE="$DATASET_ROOT/data/math7/gsm8k/gsm8k-test_with_system_prompt.parquet"
    export MATH7_MAWPS_FILE="$DATASET_ROOT/data/math7/MAWPS/mawps-test_with_system_prompt.parquet"
    export MATH7_SVAMP_FILE="$DATASET_ROOT/data/math7/SVAMP/svamp-test_with_system_prompt.parquet"
else
    # Local/manual use stays default-local and explicitly overridable.
    export DATASET_ROOT=${DATASET_ROOT:-"$ROOT"}
    export MODEL_ROOT=${MODEL_ROOT:-"$ROOT/models/rebuttal_rlvr/init"}
    export STATE_ROOT=${STATE_ROOT:-"$ROOT"}
    export OUTPUT_ROOT=${OUTPUT_ROOT:-"$STATE_ROOT/verl-exp"}
    export BASE_CKPT_DIR=${BASE_CKPT_DIR:-"$OUTPUT_ROOT/checkpoints/rebuttal_rlvr"}
    export EVAL_ROOT=${EVAL_ROOT:-"$OUTPUT_ROOT/eval/rebuttal_rlvr"}
    export LOG_DIR=${LOG_DIR:-"$OUTPUT_ROOT/logs/rebuttal_rlvr"}
    export WANDB_DIR=${WANDB_DIR:-"$OUTPUT_ROOT/wandb_runs/rebuttal_rlvr"}
    export RECEIPT_ROOT=${RECEIPT_ROOT:-"$OUTPUT_ROOT/receipts/rebuttal_rlvr"}

    export HF_HOME=${HF_HOME:-"$OUTPUT_ROOT/cache/hf"}
    export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-"$HF_HOME/hub"}
    export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$OUTPUT_ROOT/cache/datasets"}
    export XDG_CACHE_HOME=${XDG_CACHE_HOME:-"$OUTPUT_ROOT/cache/xdg"}
    export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/rebuttal_rlvr/ray}
    export TMPDIR=${TMPDIR:-/tmp/rebuttal_rlvr/tmp}
    export VLLM_CONFIG_ROOT=${VLLM_CONFIG_ROOT:-/tmp/rebuttal_rlvr/vllm}
    export VERL_ZMQ_IPC_DIR=${VERL_ZMQ_IPC_DIR:-/tmp/rebuttal_rlvr/zmq}

    export TRAIN_FILE=${TRAIN_FILE:-"$DATASET_ROOT/data/math/train_rl_format.parquet"}
    export MATH7_AIME_FILE=${MATH7_AIME_FILE:-"$DATASET_ROOT/data/math7/AIME-2025/aime-2025_with_system_prompt.parquet"}
    export MATH7_MATH500_FILE=${MATH7_MATH500_FILE:-"$DATASET_ROOT/data/math7/MATH-500/math500-test_with_system_prompt.parquet"}
    export MATH7_AMC23_FILE=${MATH7_AMC23_FILE:-"$DATASET_ROOT/data/math7/AMC23/amc23-test_with_system_prompt.parquet"}
    export MATH7_AQUA_FILE=${MATH7_AQUA_FILE:-"$DATASET_ROOT/data/math7/AQUA/aqua-test_with_system_prompt.parquet"}
    export MATH7_GSM8K_FILE=${MATH7_GSM8K_FILE:-"$DATASET_ROOT/data/math7/gsm8k/gsm8k-test_with_system_prompt.parquet"}
    export MATH7_MAWPS_FILE=${MATH7_MAWPS_FILE:-"$DATASET_ROOT/data/math7/MAWPS/mawps-test_with_system_prompt.parquet"}
    export MATH7_SVAMP_FILE=${MATH7_SVAMP_FILE:-"$DATASET_ROOT/data/math7/SVAMP/svamp-test_with_system_prompt.parquet"}
fi

export WANDB_MODE=offline
export REQUIRE_PLATFORM_RECEIPTS="$PLATFORM_MODE"
if [ "$PLATFORM_MODE" = "1" ]; then
    : "${INIT_MODEL_PATH:?platform env requires manifest-bound INIT_MODEL_PATH}"
    export BASE_PLACEHOLDER_MODEL_PATH="$INIT_MODEL_PATH"
else
    export ORDINARY_SFT_4B_MODEL_NAME=${ORDINARY_SFT_4B_MODEL_NAME:-R01_ORDINARY_SFT_4B_AM1P4M}
    export ORDINARY_SFT_4B_MODEL_PATH=${ORDINARY_SFT_4B_MODEL_PATH:-"$MODEL_ROOT/$ORDINARY_SFT_4B_MODEL_NAME"}
    export WDL_4B_MODEL_PATH=${WDL_4B_MODEL_PATH:-"$MODEL_ROOT/REPLACE_WITH_OFFLINE_WDL_SFT_MATH"}
    export BASE_PLACEHOLDER_MODEL_PATH=${BASE_PLACEHOLDER_MODEL_PATH:-"$ROOT/models/Qwen/Qwen3-4B-Base"}
fi

mkdir -p \
    "$BASE_CKPT_DIR" "$EVAL_ROOT" "$LOG_DIR" "$WANDB_DIR" "$RECEIPT_ROOT" \
    "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" \
    "$RAY_TMPDIR" "$TMPDIR" "$VLLM_CONFIG_ROOT" "$VERL_ZMQ_IPC_DIR"
