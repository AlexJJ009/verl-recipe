#!/usr/bin/env bash
# Common Stage2 code-task fixed-Model2 handoff launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WRAPPER_SCRIPT_DIR="$SCRIPT_DIR"

: "${RUN_PREFIX:?RUN_PREFIX must be set}"
: "${WDL_SFT_BETA:?WDL_SFT_BETA must be set}"

export STAGE2_HANDOFF_STEP=${STAGE2_HANDOFF_STEP:-${STAGE1_STEP:-}}
if [ -z "$STAGE2_HANDOFF_STEP" ]; then
    echo "[code-s2] ERROR: set STAGE2_HANDOFF_STEP/STAGE1_STEP explicitly; best handoff requires ALLOW_BEST_HANDOFF=1" >&2
    exit 1
fi
if [ "$STAGE2_HANDOFF_STEP" = "best" ] && [ "${ALLOW_BEST_HANDOFF:-0}" != "1" ]; then
    echo "[code-s2] ERROR: STAGE2_HANDOFF_STEP=best requires ALLOW_BEST_HANDOFF=1" >&2
    exit 1
fi
export STAGE1_STEP="$STAGE2_HANDOFF_STEP"

export CODE_TRAIN_FILE=${CODE_TRAIN_FILE:-/data-1/dataset/code/verl_rl/kodcode_stage2_after_s1_seed20260604_handoff.parquet}
export CODE_ONLINE_HUMANEVAL_PLUS_VAL_FILE=${CODE_ONLINE_HUMANEVAL_PLUS_VAL_FILE:-/data-1/dataset/code/verl_rl/online_full_humaneval_plus/official_humaneval_plus_val.parquet}
export CODE_ONLINE_MBPP_PLUS_VAL_FILE=${CODE_ONLINE_MBPP_PLUS_VAL_FILE:-/data-1/dataset/code/verl_rl/online_full_mbpp_plus/official_mbpp_plus_val.parquet}
export CODE_VAL_FILES=${CODE_VAL_FILES:-"['$CODE_ONLINE_HUMANEVAL_PLUS_VAL_FILE','$CODE_ONLINE_MBPP_PLUS_VAL_FILE']"}
export TRAIN_FILE=${TRAIN_FILE:-"$CODE_TRAIN_FILE"}
export TEST_FILES=${TEST_FILES:-"$CODE_VAL_FILES"}
export CUSTOM_REWARD_FN_PATH=${CUSTOM_REWARD_FN_PATH:-"${SCRIPT_DIR}/official_aligned_reward.py"}
export CUSTOM_REWARD_FN_NAME=${CUSTOM_REWARD_FN_NAME:-compute_score_code_official_aligned}
export REWARD_MANAGER=${REWARD_MANAGER:-dapo}
export CODE_EVAL_OFFICIAL_SITE=${CODE_EVAL_OFFICIAL_SITE:-/data-1/code_eval_envs/official_site}
export LCB_REPO_DIR=${LCB_REPO_DIR:-/data-1/code_eval_envs/LiveCodeBench}
export PROJECT_CACHE_ROOT=${PROJECT_CACHE_ROOT:-/data-1/.cache}
export HF_HOME=${CODE_TASK_HF_HOME:-$PROJECT_CACHE_ROOT/huggingface}
export HF_DATASETS_CACHE=${CODE_TASK_HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HUGGINGFACE_HUB_CACHE=${CODE_TASK_HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${CODE_TASK_TRANSFORMERS_CACHE:-$HF_HOME}
export XDG_CACHE_HOME=${CODE_TASK_XDG_CACHE_HOME:-$PROJECT_CACHE_ROOT}
export CODE_OFFICIAL_SOURCE_ROOT=${CODE_OFFICIAL_SOURCE_ROOT:-/data-1/dataset/code/official_sources}
export BIGCODEBENCH_OVERRIDE_PATH=${BIGCODEBENCH_OVERRIDE_PATH:-$CODE_OFFICIAL_SOURCE_ROOT/bigcodebench/BigCodeBench-v0.1.4.jsonl}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export PYTHONPATH="${CODE_EVAL_OFFICIAL_SITE}:${LCB_REPO_DIR}:${PYTHONPATH:-}"
export LOSS_MODE=${LOSS_MODE:-wdl_sft}
export LR=${LR:-5e-7}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-5}
export TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-8}
export ROLLOUT_N=${ROLLOUT_N:-2}
export TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ:-$((TRAIN_PROMPT_BSZ * ROLLOUT_N))}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
export TEST_FREQ=${TEST_FREQ:-1}
export SAVE_FREQ=${SAVE_FREQ:-1}
export VAL_N=${VAL_N:-1}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-64}
export DATA_SEED=${DATA_SEED:-20260604}
export DATA_SHUFFLE=${DATA_SHUFFLE:-False}
export WANDB_PROJECT=${WANDB_PROJECT:-OnPolicyWDLSFT-CodeTask}
export WANDB_MODE=${WANDB_MODE:-offline}
export BIGCODEBENCH_MAX_AS_LIMIT=${BIGCODEBENCH_MAX_AS_LIMIT:-131072}
export BIGCODEBENCH_MAX_DATA_LIMIT=${BIGCODEBENCH_MAX_DATA_LIMIT:-131072}
export BIGCODEBENCH_MAX_STACK_LIMIT=${BIGCODEBENCH_MAX_STACK_LIMIT:-10}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export BEST_CKPT_METRIC_KEY=${BEST_CKPT_METRIC_KEY:-"val-core/HumanEval+/acc/pass@1"}
export MAX_ACTOR_CKPTS_TO_KEEP=${MAX_ACTOR_CKPTS_TO_KEEP:-null}
export MAX_CRITIC_CKPTS_TO_KEEP=${MAX_CRITIC_CKPTS_TO_KEEP:-null}
export KEEP_BEST_CKPT=${KEEP_BEST_CKPT:-True}
export BEST_CKPT_STRIP_OPTIMIZER=${BEST_CKPT_STRIP_OPTIMIZER:-False}
export JOINT_TRAINING_ROLLOUT_SOURCE=${JOINT_TRAINING_ROLLOUT_SOURCE:-model2}
export ROLLOUT_CALCULATE_LOG_PROBS=${ROLLOUT_CALCULATE_LOG_PROBS:-True}
export CALCULATE_ENTROPY=${CALCULATE_ENTROPY:-False}
export MIN_FREE_GB_FOR_CKPT=${MIN_FREE_GB_FOR_CKPT:-30}
export REQUIRE_MERGED_MODEL2_PROVENANCE=${REQUIRE_MERGED_MODEL2_PROVENANCE:-True}

if [ ! -f "$TRAIN_FILE" ]; then
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[code-s2] WARNING: TRAIN_FILE not found during DRY_RUN: $TRAIN_FILE" >&2
    else
        echo "[code-s2] ERROR: TRAIN_FILE not found: $TRAIN_FILE" >&2
        exit 1
    fi
fi

EXTERNAL_DRY_RUN_MODEL2=0
export MERGED_MODEL2_PROVENANCE_FILE=${MERGED_MODEL2_PROVENANCE_FILE:-"${MERGED_MODEL2_DIR:-}/stage1_source.json"}
if [ -n "${MODEL2_PATH:-}" ] && [ "${ALLOW_EXTERNAL_MODEL2_FOR_DRY_RUN:-0}" = "1" ] && [ "${DRY_RUN:-0}" = "1" ]; then
    EXTERNAL_DRY_RUN_MODEL2=1
    echo "[code-s2] using explicit external MODEL2_PATH for dry-run plumbing only: $MODEL2_PATH"
else
    if [ -z "${STAGE1_SOURCE_CKPT:-${STAGE1_CKPT_DIR:-}}" ]; then
        echo "[code-s2] ERROR: set STAGE1_SOURCE_CKPT or STAGE1_CKPT_DIR for Stage2 provenance" >&2
        exit 1
    fi
    export STAGE1_CKPT_DIR=${STAGE1_CKPT_DIR:-$STAGE1_SOURCE_CKPT}
    export STAGE1_RUN_PREFIX=${STAGE1_RUN_PREFIX:-CODE-S1-SMOKE-BETA0}
    export STAGE1_MERGED_MODEL_ROOT=${STAGE1_MERGED_MODEL_ROOT:-/data-1/model_weights/code_task/stage1_model2}
    export MERGED_MODEL2_DIR=${MERGED_MODEL2_DIR:-"${STAGE1_MERGED_MODEL_ROOT}/${STAGE1_RUN_PREFIX}/step_${STAGE2_HANDOFF_STEP}"}
    export MERGED_MODEL2_PROVENANCE_FILE=${MERGED_MODEL2_PROVENANCE_FILE:-"${MERGED_MODEL2_DIR}/stage1_source.json"}
fi

has_merged_weights() {
    [ -f "${MERGED_MODEL2_DIR}/model.safetensors" ] || [ -f "${MERGED_MODEL2_DIR}/model.safetensors.index.json" ]
}

write_or_check_code_provenance() {
    python3 - "$MERGED_MODEL2_PROVENANCE_FILE" "$STAGE1_RUN_PREFIX" "$STAGE1_CKPT_DIR" "$STAGE2_HANDOFF_STEP" "${STAGE1_CKPT_DIR}/global_step_${STAGE2_HANDOFF_STEP}/actor" "$MERGED_MODEL2_DIR" "$TRAIN_FILE" <<'PY'
import json, sys
from pathlib import Path
path, run_prefix, ckpt, step, actor, target, manifest_train = sys.argv[1:]
manifest = Path(manifest_train).with_suffix(".manifest.json")
expected = {
  "stage1_run_prefix": run_prefix,
  "source_checkpoint": ckpt,
  "handoff_step": int(step) if str(step).isdigit() else step,
  "actor_dir": actor,
  "target_dir": target,
  "prompt_template_version": "code-think-answer-python-v1",
  "code_data_manifest_path": str(manifest),
}
p = Path(path)
if p.exists():
    actual = json.loads(p.read_text())
    if actual != expected:
        raise SystemExit("Stage2 provenance mismatch\nexpected=%s\nactual=%s" % (json.dumps(expected, sort_keys=True), json.dumps(actual, sort_keys=True)))
else:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
print(json.dumps(expected, sort_keys=True))
PY
}

print_or_check_code_provenance_dry_run() {
    python3 - "$MERGED_MODEL2_PROVENANCE_FILE" "$STAGE1_RUN_PREFIX" "$STAGE1_CKPT_DIR" "$STAGE2_HANDOFF_STEP" "${STAGE1_CKPT_DIR}/global_step_${STAGE2_HANDOFF_STEP}/actor" "$MERGED_MODEL2_DIR" "$TRAIN_FILE" <<'PY'
import json, sys
from pathlib import Path
path, run_prefix, ckpt, step, actor, target, manifest_train = sys.argv[1:]
manifest = Path(manifest_train).with_suffix(".manifest.json")
expected = {
  "stage1_run_prefix": run_prefix,
  "source_checkpoint": ckpt,
  "handoff_step": int(step) if str(step).isdigit() else step,
  "actor_dir": actor,
  "target_dir": target,
  "prompt_template_version": "code-think-answer-python-v1",
  "code_data_manifest_path": str(manifest),
}
p = Path(path)
if p.exists():
    actual = json.loads(p.read_text())
    if actual != expected:
        raise SystemExit("Stage2 provenance mismatch\nexpected=%s\nactual=%s" % (json.dumps(expected, sort_keys=True), json.dumps(actual, sort_keys=True)))
print(json.dumps(expected, sort_keys=True))
PY
}

prepare_model2_if_needed() {
    [ "$EXTERNAL_DRY_RUN_MODEL2" = "1" ] && return 0
    local actor_dir="${STAGE1_CKPT_DIR}/global_step_${STAGE2_HANDOFF_STEP}/actor"
    if [ ! -d "$actor_dir" ]; then
        echo "[code-s2] ERROR: Stage1 actor dir not found: $actor_dir" >&2
        exit 1
    fi
    if [ ! -f "${TRAIN_FILE%.*}.manifest.json" ] && [ ! -f "$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).with_suffix(\".manifest.json\"))' "$TRAIN_FILE")" ]; then
        echo "[code-s2] ERROR: code data manifest not found for TRAIN_FILE=$TRAIN_FILE" >&2
        exit 1
    fi
    if [ -e "$MERGED_MODEL2_DIR" ] && ! has_merged_weights; then
        if [ "${DRY_RUN:-0}" = "1" ]; then
            echo "[code-s2] DRY_RUN ignoring stale merged Model2 dir without weights: $MERGED_MODEL2_DIR" >&2
            print_or_check_code_provenance_dry_run
            return 0
        fi
        if [ "${ALLOW_OVERWRITE_MERGED_MODEL2:-0}" = "1" ]; then
            echo "[code-s2] removing stale merged Model2 dir without weights: $MERGED_MODEL2_DIR" >&2
            rm -rf "$MERGED_MODEL2_DIR"
        else
            echo "[code-s2] ERROR: stale merged Model2 dir without weights: $MERGED_MODEL2_DIR" >&2
            exit 1
        fi
    fi
    if has_merged_weights; then
        write_or_check_code_provenance
    else
        if [ "${DRY_RUN:-0}" = "1" ]; then
            print_or_check_code_provenance_dry_run
            return 0
        fi
        write_or_check_code_provenance
        mkdir -p "$(dirname "$MERGED_MODEL2_DIR")"
        CUDA_VISIBLE_DEVICES=${MERGE_CUDA_VISIBLE_DEVICES:-0} python3 -u -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$actor_dir" \
            --target_dir "$MERGED_MODEL2_DIR" \
            --trust-remote-code
        write_or_check_code_provenance
    fi
    export MODEL2_PATH="$MERGED_MODEL2_DIR"
}

print_config() {
    cat <<EOF
[CODE S2 CONFIG]
RUN_PREFIX=$RUN_PREFIX
TRAIN_FILE=$TRAIN_FILE
TEST_FILES=$TEST_FILES
CODE_ONLINE_HUMANEVAL_PLUS_VAL_FILE=$CODE_ONLINE_HUMANEVAL_PLUS_VAL_FILE
CODE_ONLINE_MBPP_PLUS_VAL_FILE=$CODE_ONLINE_MBPP_PLUS_VAL_FILE
CUSTOM_REWARD_FN_PATH=$CUSTOM_REWARD_FN_PATH
CUSTOM_REWARD_FN_NAME=$CUSTOM_REWARD_FN_NAME
CODE_EVAL_OFFICIAL_SITE=$CODE_EVAL_OFFICIAL_SITE
LCB_REPO_DIR=$LCB_REPO_DIR
PROJECT_CACHE_ROOT=$PROJECT_CACHE_ROOT
HF_HOME=$HF_HOME
HF_DATASETS_CACHE=$HF_DATASETS_CACHE
HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE
XDG_CACHE_HOME=$XDG_CACHE_HOME
CODE_OFFICIAL_SOURCE_ROOT=$CODE_OFFICIAL_SOURCE_ROOT
BIGCODEBENCH_OVERRIDE_PATH=$BIGCODEBENCH_OVERRIDE_PATH
STAGE1_CKPT_DIR=${STAGE1_CKPT_DIR:-}
STAGE2_HANDOFF_STEP=$STAGE2_HANDOFF_STEP
MERGED_MODEL2_DIR=${MERGED_MODEL2_DIR:-$MODEL2_PATH}
MERGED_MODEL2_PROVENANCE_FILE=$MERGED_MODEL2_PROVENANCE_FILE
MODEL2_PATH=${MODEL2_PATH:-}
JOINT_TRAINING_ROLLOUT_SOURCE=$JOINT_TRAINING_ROLLOUT_SOURCE
LOSS_MODE=$LOSS_MODE
WDL_SFT_BETA=$WDL_SFT_BETA
TRAIN_PROMPT_BSZ=$TRAIN_PROMPT_BSZ
ROLLOUT_N=$ROLLOUT_N
TRAIN_PROMPT_MINI_BSZ=$TRAIN_PROMPT_MINI_BSZ
NGPUS_PER_NODE=$NGPUS_PER_NODE
TOTAL_TRAINING_STEPS=$TOTAL_TRAINING_STEPS
TRAIN_MAX_SAMPLES=$TRAIN_MAX_SAMPLES
DATA_SEED=$DATA_SEED
DATA_SHUFFLE=$DATA_SHUFFLE
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}
SAVE_FREQ=$SAVE_FREQ
TEST_FREQ=$TEST_FREQ
MAX_ACTOR_CKPTS_TO_KEEP=$MAX_ACTOR_CKPTS_TO_KEEP
MAX_CRITIC_CKPTS_TO_KEEP=$MAX_CRITIC_CKPTS_TO_KEEP
KEEP_BEST_CKPT=$KEEP_BEST_CKPT
BEST_CKPT_STRIP_OPTIMIZER=$BEST_CKPT_STRIP_OPTIMIZER
BEST_CKPT_METRIC_KEY=$BEST_CKPT_METRIC_KEY
BIGCODEBENCH_MAX_AS_LIMIT=$BIGCODEBENCH_MAX_AS_LIMIT
BIGCODEBENCH_MAX_DATA_LIMIT=$BIGCODEBENCH_MAX_DATA_LIMIT
EOF
}

print_config

prepare_model2_if_needed

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[code-s2] DRY_RUN=1; exiting before merge/joint preparation/training"
    exit 0
fi

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../staged_v1/_run_stage2_model2_rollout_common.sh" "$@"
