#!/usr/bin/env bash
set -euo pipefail
RAY_STARTED=0
cleanup() {
  [ "$RAY_STARTED" = 0 ] || python3 - "$RAY_TMPDIR" <<'PY' || true
import os, signal, sys, time
root = sys.argv[1]
owned = []
for entry in os.scandir('/proc'):
    if not entry.name.isdigit() or int(entry.name) == os.getpid():
        continue
    try:
        command = open(f'/proc/{entry.name}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='replace')
    except OSError:
        continue
    if root in command:
        owned.append(int(entry.name))
for pid in owned:
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
deadline = time.monotonic() + 10
while time.monotonic() < deadline and any(os.path.exists(f'/proc/{pid}') for pid in owned):
    time.sleep(0.1)
for pid in owned:
    if os.path.exists(f'/proc/{pid}'):
        try: os.kill(pid, signal.SIGKILL)
        except ProcessLookupError: pass
PY
}
trap cleanup EXIT INT TERM
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
: "${REPO_PYTHONPATH_ROOT:=/workspace/verl}"
: "${CODE_EVAL_OFFICIAL_SITE:=/data-1/code_eval_envs/official_site}"
: "${LCB_REPO_DIR:=/data-1/code_eval_envs/LiveCodeBench}"
: "${LCB_INPUT_OUTPUT_INDEX:=/data-2/evaluator_assets/livecodebench_cache/index/release_v5_input_output.sqlite}"
export CALIBRATION_SCORER_PYTHONPATH="$REPO_PYTHONPATH_ROOT:$CODE_EVAL_OFFICIAL_SITE:$LCB_REPO_DIR"
PYTHONPATH="$CALIBRATION_SCORER_PYTHONPATH" LCB_INPUT_OUTPUT_INDEX="$LCB_INPUT_OUTPUT_INDEX" \
  python3 "$SCRIPT_DIR/check_official_scorer_dependencies.py" >/dev/null
: "${CALIBRATION_RAY_WORKER_PORT_MIN:=21000}"
: "${CALIBRATION_RAY_WORKER_PORT_MAX:=21999}"
: "${CALIBRATION_RAY_HEAD_PORT:=22000}"
: "${CALIBRATION_TCPSTORE_PORT_MIN:=35000}"
: "${CALIBRATION_TCPSTORE_PORT_MAX:=35999}"
: "${CALIBRATION_OUTPUT_ROOT:?}"
python3 - "$CALIBRATION_RAY_WORKER_PORT_MIN" "$CALIBRATION_RAY_WORKER_PORT_MAX" "$CALIBRATION_TCPSTORE_PORT_MIN" "$CALIBRATION_TCPSTORE_PORT_MAX" <<'PY'
import sys

ray_min, ray_max, store_min, store_max = map(int, sys.argv[1:])
for name, low, high in (("Ray worker", ray_min, ray_max), ("TCPStore", store_min, store_max)):
    if not 1024 <= low < high <= 65535:
        raise SystemExit(f"invalid {name} port range: {low}-{high}")
if max(ray_min, store_min) <= min(ray_max, store_max):
    raise SystemExit("calibration Ray worker and TCPStore port ranges overlap")
PY
: "${CALIBRATION_RAY_TMPDIR:=$CALIBRATION_OUTPUT_ROOT/ray}"
mkdir -p "$CALIBRATION_RAY_TMPDIR"
export RAY_TMPDIR="$CALIBRATION_RAY_TMPDIR"
ray start --head --port="$CALIBRATION_RAY_HEAD_PORT" --temp-dir="$RAY_TMPDIR" \
  --min-worker-port="$CALIBRATION_RAY_WORKER_PORT_MIN" \
  --max-worker-port="$CALIBRATION_RAY_WORKER_PORT_MAX" \
  --include-dashboard=false --disable-usage-stats >/dev/null
RAY_STARTED=1
export RAY_ADDRESS="127.0.0.1:$CALIBRATION_RAY_HEAD_PORT"
: "${CALIBRATION_HUMANEVAL_PLUS_FILE:?}"
: "${CALIBRATION_MBPP_PLUS_FILE:?}"
: "${CALIBRATION_LIVE_CODE_BENCH_FILE:?}"
: "${CALIBRATION_TOTAL_TRAINING_STEPS:=0}"
: "${CALIBRATION_OPTIMIZER_ENABLED:=false}"
if [ "$CALIBRATION_TOTAL_TRAINING_STEPS" != 0 ] || [ "$CALIBRATION_OPTIMIZER_ENABLED" != false ]; then
  echo '{"code":"calibration_training_enabled","message":"calibration must remain zero-step with optimizer disabled","context":{}}' >&2
  exit 2
fi
: "${VERL_FILE_LOGGER_ROOT:=$CALIBRATION_OUTPUT_ROOT/logs/metrics}"
export VERL_FILE_LOGGER_ROOT
mkdir -p "$CALIBRATION_OUTPUT_ROOT/checkpoints" "$CALIBRATION_OUTPUT_ROOT/logs" "$VERL_FILE_LOGGER_ROOT"
common=(
  RUN_PREFIX="CALIBRATION-${PHASE^^}-QWEN3-1P7B-CTX8K"
  CODE_VAL_FILES="['$CALIBRATION_HUMANEVAL_PLUS_FILE','$CALIBRATION_MBPP_PLUS_FILE','$CALIBRATION_LIVE_CODE_BENCH_FILE']"
  TEST_FILES="['$CALIBRATION_HUMANEVAL_PLUS_FILE','$CALIBRATION_MBPP_PLUS_FILE','$CALIBRATION_LIVE_CODE_BENCH_FILE']" VAL_MAX_SAMPLES=-1
  VAL_BEFORE_TRAIN=True VAL_ONLY=True TOTAL_TRAINING_STEPS="$CALIBRATION_TOTAL_TRAINING_STEPS" TEST_FREQ=1 SAVE_FREQ=1000 TRAIN_MAX_SAMPLES=192
  BASE_CKPT_DIR="$CALIBRATION_OUTPUT_ROOT/checkpoints" LOG_DIR="$CALIBRATION_OUTPUT_ROOT/logs"
  WANDB_MODE=disabled KEEP_BEST_CKPT=False MAX_ACTOR_CKPTS_TO_KEEP=0 MAX_CRITIC_CKPTS_TO_KEEP=0
  CODE_REWARD_TIMEOUT=30 CODE_REWARD_MANAGER_TIMEOUT=30
)
hydra_overrides=(
  +data.require_source_uid=true
  +trainer.ray_master_port_range="[$CALIBRATION_TCPSTORE_PORT_MIN,$CALIBRATION_TCPSTORE_PORT_MAX]"
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT="'$VERL_FILE_LOGGER_ROOT'"
  trainer.val_only=true
  trainer.save_freq=-1
  'trainer.logger=["file"]'
)
case "$PHASE" in
 stage1)
  python3 - "${STAGE1_INIT_MODEL_PATH:?}" "${STAGE1_INIT_PROVENANCE_PATH:?}" <<'PY'
import json,sys
from pathlib import Path
model=Path(sys.argv[1]).resolve(); provenance=Path(sys.argv[2])
data=json.loads(provenance.read_text())
if Path(data.get("target_dir", "")).resolve() != model:
    raise SystemExit("Stage1 calibration provenance target mismatch")
PY
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
