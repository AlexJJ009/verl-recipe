#!/usr/bin/env bash
set -euo pipefail
SANDBOX_DRY_RUN=0
if [ "${1:-}" = "--sandbox-dry-run" ]; then SANDBOX_DRY_RUN=1; shift; fi
PHASE=${1:?stage1|stage2|stage3 required}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
MANIFEST=${CALIBRATION_MANIFEST:-$REPO_ROOT/recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml}
MANIFEST_TOOL=${CALIBRATION_MANIFEST_TOOL:-$REPO_ROOT/scripts/experiment_manifest.py}
if [ "$SANDBOX_DRY_RUN" = 1 ]; then
  python3 "$MANIFEST_TOOL" render "$MANIFEST" --format json | python3 -c '
import json,sys
d=json.load(sys.stdin); phase=sys.argv[1]; workload=d["calibration_workloads"][phase]
print(json.dumps({"ok":True,"phase":phase,"model_provenance_class":workload["model_provenance_class"],"model_sources":workload["model_sources"]},sort_keys=True))
' "$PHASE"
  exit 0
fi
source "$SCRIPT_DIR/qwen3_1p7b_stage123_resource_profile.sh"
: "${CALIBRATION_HUMANEVAL_PLUS_FILE:?}"
: "${CALIBRATION_MBPP_PLUS_FILE:?}"
: "${CALIBRATION_LIVE_CODE_BENCH_FILE:?}"
: "${CALIBRATION_OUTPUT_ROOT:?}"
mkdir -p "$CALIBRATION_OUTPUT_ROOT/checkpoints" "$CALIBRATION_OUTPUT_ROOT/logs"
common=(
  RUN_PREFIX="CALIBRATION-${PHASE^^}-QWEN3-1P7B-CTX8K"
  CODE_VAL_FILES="['$CALIBRATION_HUMANEVAL_PLUS_FILE','$CALIBRATION_MBPP_PLUS_FILE','$CALIBRATION_LIVE_CODE_BENCH_FILE']"
  TEST_FILES="['$CALIBRATION_HUMANEVAL_PLUS_FILE','$CALIBRATION_MBPP_PLUS_FILE','$CALIBRATION_LIVE_CODE_BENCH_FILE']" VAL_MAX_SAMPLES=-1
  VAL_BEFORE_TRAIN=True VAL_ONLY=True TOTAL_TRAINING_STEPS=1 TEST_FREQ=1 SAVE_FREQ=1000 TRAIN_MAX_SAMPLES=192
  BASE_CKPT_DIR="$CALIBRATION_OUTPUT_ROOT/checkpoints" LOG_DIR="$CALIBRATION_OUTPUT_ROOT/logs"
  WANDB_MODE=disabled KEEP_BEST_CKPT=False MAX_ACTOR_CKPTS_TO_KEEP=0 MAX_CRITIC_CKPTS_TO_KEEP=0
  CODE_REWARD_TIMEOUT=30 CODE_REWARD_MANAGER_TIMEOUT=30
)
hydra_overrides=(+data.require_source_uid=true)
case "$PHASE" in
 stage1)
  env "${common[@]}" INIT_MODEL_PATH="${STAGE1_INIT_MODEL_PATH:?}" WDL_SFT_BETA=0.1 bash "$SCRIPT_DIR/run_s1_code_base.sh" "${hydra_overrides[@]}"
  ;;
 stage2)
  : "${CALIBRATION_STAGE1_CKPT_DIR:?}" "${CALIBRATION_STAGE1_MODEL2:?}"
  env "${common[@]}" BASE_MODEL_PATH="${QWEN3_1P7B_MODEL_PATH:?}" SUBMODEL_KL_MODEL1_REF_PATH="$QWEN3_1P7B_MODEL_PATH" MODEL_PATH="$CALIBRATION_OUTPUT_ROOT/joint_model" MODEL2_CACHE_TAG=calibration-s2-1p7b STAGE1_CKPT_DIR="$CALIBRATION_STAGE1_CKPT_DIR" STAGE1_RUN_PREFIX="${CALIBRATION_STAGE1_RUN_PREFIX:?}" STAGE2_HANDOFF_STEP="${CALIBRATION_STAGE1_HANDOFF_STEP:?}" MERGED_MODEL2_DIR="$CALIBRATION_STAGE1_MODEL2" MODEL2_PATH="$CALIBRATION_STAGE1_MODEL2" REQUIRE_MERGED_MODEL2_PROVENANCE=False CODE_TRAIN_FILE="${CALIBRATION_TRAIN_FILE:?}" TRAIN_FILE="$CALIBRATION_TRAIN_FILE" WDL_SFT_BETA=0.1 EXPECTED_STAGE1_BETA=0.1 FUSION_LAMBDA=0.8 bash "$SCRIPT_DIR/run_s2_code_model2_rollout_common.sh" "${hydra_overrides[@]}"
  ;;
 stage3)
  env "${common[@]}" INIT_MODEL_PATH="${CALIBRATION_STAGE3_MODEL_PATH:?}" WDL_SFT_BETA=0.1 DATA_SHUFFLE=False bash "$SCRIPT_DIR/run_s1_code_base.sh" "${hydra_overrides[@]}"
  ;;
 *) echo "invalid phase: $PHASE" >&2; exit 2;;
esac
