#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

die() { echo "Lambda matrix Slurm admission failed: $*" >&2; exit 64; }
require_env() { [ -n "${!1:-}" ] || die "$1 is required"; }

for name in SLURM_JOB_ID SLURMD_NODENAME CUDA_VISIBLE_DEVICES LAMBDA_ARM \
    FUSION_LAMBDA TRAINING_LR LAMBDA_MATRIX_PARENT_SHA LAMBDA_MATRIX_RECIPE_SHA \
    LAMBDA_MATRIX_IMAGE_ID \
    LAMBDA_MATRIX_EVIDENCE_RELAY_HOST LAMBDA_MATRIX_NODE_ROOT_MAP \
    LAMBDA_MATRIX_STAGE_REL; do
    require_env "$name"
done

case "$FUSION_LAMBDA" in
    0.5) lambda_tag=lambda05 ;;
    0.7) lambda_tag=lambda07 ;;
    0.8) lambda_tag=lambda08 ;;
    0.9) lambda_tag=lambda09 ;;
    *) die "unsupported fusion lambda" ;;
esac
case "$TRAINING_LR" in
    1e-6) lr_tag=lr1e6 ;;
    5e-7) lr_tag=lr5e7 ;;
    *) die "unsupported training lr" ;;
esac
case "$LAMBDA_ARM:$FUSION_LAMBDA:$TRAINING_LR" in
    fixed-m1:0.7:1e-6|fixed-m1:0.9:1e-6)
        arm_tag=FIXED-M1; artifact_arm=fixed-m1; expected_gradient=zero ;;
    d0:0.7:1e-6|d0:0.9:1e-6)
        arm_tag=D0; artifact_arm=d0; expected_gradient=zero ;;
    standard-c:0.5:5e-7|standard-c:0.8:5e-7)
        arm_tag=C; artifact_arm=standard-c; expected_gradient=nonzero ;;
    *) die "unauthorized arm/lambda/lr triple" ;;
esac
run_prefix="MATH-WDL-${lambda_tag^^}-ARM-${arm_tag}-${lr_tag^^}-P60-QWEN3-1P7B"
family="$run_prefix"

[[ "$LAMBDA_MATRIX_PARENT_SHA" =~ ^[0-9a-f]{40}$ ]] || die "invalid parent SHA"
[[ "$LAMBDA_MATRIX_RECIPE_SHA" =~ ^[0-9a-f]{40}$ ]] || die "invalid recipe SHA"
[[ "$LAMBDA_MATRIX_STAGE_REL" =~ ^workspace/jobs/[A-Za-z0-9._-]+$ ]] || die "invalid stage path"
[[ "$LAMBDA_MATRIX_EVIDENCE_RELAY_HOST" =~ ^[A-Za-z0-9._@:-]+$ ]] || die "invalid relay host"

node_root=""
IFS=';' read -r -a node_roots <<<"$LAMBDA_MATRIX_NODE_ROOT_MAP"
for entry in "${node_roots[@]}"; do
    if [ "${entry%%=*}" = "$SLURMD_NODENAME" ]; then node_root="${entry#*=}"; break; fi
done
[ -n "$node_root" ] || die "node root is not mapped"
node_root="$(realpath -e "$node_root")" || die "node root unavailable"
[ "$node_root" != / ] || die "node root cannot be /"
workspace="$(realpath -e "${node_root}/${LAMBDA_MATRIX_STAGE_REL}")" || die "stage unavailable"
case "$workspace" in "$node_root"/workspace/jobs/*) ;; *) die "stage escapes node root" ;; esac
repo_host="$(realpath -e "$workspace/repo")"
recipe_host="$repo_host/recipe"
launch_receipt="$workspace/.launch-receipt.json"
data1_host="$(realpath -e "$workspace/runtime/data-1")"
data2_host="$(realpath -e "$workspace/runtime/data-2")"
test "$(tr -d '\n' <"$workspace/.candidate-parent-sha")" = "$LAMBDA_MATRIX_PARENT_SHA" || die "parent marker mismatch"
test "$(tr -d '\n' <"$workspace/.candidate-recipe-sha")" = "$LAMBDA_MATRIX_RECIPE_SHA" || die "recipe marker mismatch"
test "$(git -C "$repo_host" rev-parse HEAD)" = "$LAMBDA_MATRIX_PARENT_SHA" || die "parent checkout mismatch"
test "$(git -C "$recipe_host" rev-parse HEAD)" = "$LAMBDA_MATRIX_RECIPE_SHA" || die "recipe checkout mismatch"
test "$(git -C "$repo_host" rev-parse HEAD:recipe)" = "$LAMBDA_MATRIX_RECIPE_SHA" || die "gitlink mismatch"
test -z "$(git -C "$repo_host" status --porcelain)" || die "parent checkout dirty"
test -z "$(git -C "$recipe_host" status --porcelain)" || die "recipe checkout dirty"

python3 - "$launch_receipt" "$LAMBDA_ARM" "$FUSION_LAMBDA" "$TRAINING_LR" \
    "$LAMBDA_MATRIX_PARENT_SHA" "$LAMBDA_MATRIX_RECIPE_SHA" "$LAMBDA_MATRIX_IMAGE_ID" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
arm, lam, lr, parent, recipe, image = sys.argv[2:]
receipt = json.loads(path.read_text())
expected = {
    "status": "authorized",
    "experiment_id": "math_qwen3_1p7b_wdl_lambda_followup_p60",
    "max_training_steps": 60,
    "parent_candidate_sha": parent,
    "recipe_candidate_sha": recipe,
    "image_id": image,
}
for key, value in expected.items():
    if receipt.get(key) != value:
        raise SystemExit(f"launch receipt mismatch: {key}")
run_id = f"lambda{lam.replace('.', '')}-{'fixed' if arm == 'fixed-m1' else arm.replace('standard-', '')}-lr{lr.replace('-', '')}"
authorized = {(run["id"], str(run["fusion_lambda"]), str(run["lr"])) for run in receipt.get("runs", [])}
if (run_id, lam, lr) not in authorized:
    raise SystemExit(f"run is not authorized: {run_id}")
PY

local_image_ref="$(docker image inspect verl-harness:latest --format '{{.Id}}')" || die "image missing"
image_id="$(python3 "$repo_host/scripts/l40s/resolve_image_config_digest.py" "$local_image_ref")"
test "$image_id" = "$LAMBDA_MATRIX_IMAGE_ID" || die "image identity mismatch"
foreign_gpu_processes="$(nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name --format=csv,noheader,nounits 2>/dev/null || true)"
test -z "$foreign_gpu_processes" || die "foreign GPU process present"
mem_available_gib="$(awk '/MemAvailable:/ {print int($2 / 1024 / 1024)}' /proc/meminfo)"
test "$mem_available_gib" -ge 300 || die "less than 300 GiB memory available"

artifact_root="${data2_host}/model_weights/math_task/qwen3_1p7b_wdl_lambda_followup/${lambda_tag}/${artifact_arm}-${lr_tag}-p60"
job_root="${artifact_root}/slurm/${SLURM_JOB_ID}"
test ! -e "$job_root" || die "job root already exists"
mkdir -p "$job_root"
relay_job_root="/data-1/code/_artifacts/verl-v0.7/math-lambda-followup/${LAMBDA_MATRIX_PARENT_SHA}/${SLURM_JOB_ID}"
container_name="lambda-followup-${SLURM_JOB_ID}-${artifact_arm}-${lambda_tag}-${lr_tag}"
training_pid="" gate_pid="" run_name="" completion_recorded=false

discover_run_name() {
    [ -n "$run_name" ] && return 0
    local metrics_dir="${artifact_root}/logs/metrics/OnPolicyWDLSFT-Math-1P7B-Lambda-Followup-P60"
    local -a candidates=()
    [ -d "$metrics_dir" ] || return 1
    mapfile -t candidates < <(find "$metrics_dir" -maxdepth 1 -type f -name "${run_prefix}_*.jsonl" | sort)
    [ "${#candidates[@]}" -eq 1 ] || return 1
    run_name="$(basename "${candidates[0]}" .jsonl)"
}

relay_files() {
    local -a files=(); local name
    for name in "$@"; do [ ! -f "$job_root/$name" ] || files+=("$job_root/$name"); done
    [ "${#files[@]}" -eq 0 ] || rsync --archive --mkpath --protect-args \
        -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10' \
        "${files[@]}" "${LAMBDA_MATRIX_EVIDENCE_RELAY_HOST}:${relay_job_root}/"
}

record_gate() {
    local status=$1 checkpoint=${2:-} metrics=${3:-} observed=${4:-0}
    [ -n "$run_name" ] || return 0
    [[ "$run_name" =~ ^[A-Z0-9._-]+$ ]] || return 1
    ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
        "$LAMBDA_MATRIX_EVIDENCE_RELAY_HOST" \
        python3 /data-1/code/verl/scripts/training_result_release_gate.py record \
        --run-name "$run_name" --family "$family" --status "$status" \
        --source slurm:lambda-matrix --checkpoint "$checkpoint" --metrics "$metrics" \
        --final-step 60 --observed-step "$observed"
    if [ "$status" = success_complete ]; then
        ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
            "$LAMBDA_MATRIX_EVIDENCE_RELAY_HOST" \
            python3 /data-1/code/verl/scripts/training_result_release_gate.py check \
            --run-name "$run_name" --family "$family"
    fi
}

cleanup() {
    local rc=$?
    local terminal_rc=$rc evidence_relayed=true
    trap - EXIT TERM INT
    set +e
    [ -z "$training_pid" ] || { kill "$training_pid" >/dev/null 2>&1; wait "$training_pid" >/dev/null 2>&1; }
    [ -z "$gate_pid" ] || { kill "$gate_pid" >/dev/null 2>&1; wait "$gate_pid" >/dev/null 2>&1; }
    docker rm --force "$container_name" >/dev/null 2>&1 || true
    tail -n 200 "$job_root/stdout.log" >"$job_root/stdout.tail.log" 2>/dev/null || true
    tail -n 200 "$job_root/stderr.log" >"$job_root/stderr.tail.log" 2>/dev/null || true
    discover_run_name || true
    if [ "$rc" -ne 0 ] && [ "$completion_recorded" != true ] && [ -n "$run_name" ]; then
        record_gate failed "" "" 0 >"$job_root/release-gate.log" 2>&1 || true
    fi
    relay_files admission.json first-step.json first-step.log completion.json release-gate.log stdout.tail.log stderr.tail.log || { evidence_relayed=false; rc=70; }
    printf '{"job_id":"%s","node":"%s","arm":"%s","fusion_lambda":%s,"training_lr":"%s","training_exit_code":%s,"completion_recorded":%s,"evidence_relayed":%s,"parent_sha":"%s","recipe_sha":"%s","image_id":"%s"}\n' \
        "$SLURM_JOB_ID" "$SLURMD_NODENAME" "$LAMBDA_ARM" "$FUSION_LAMBDA" "$TRAINING_LR" "$terminal_rc" \
        "$completion_recorded" "$evidence_relayed" "$LAMBDA_MATRIX_PARENT_SHA" \
        "$LAMBDA_MATRIX_RECIPE_SHA" "$LAMBDA_MATRIX_IMAGE_ID" >"$job_root/terminal.json"
    relay_files terminal.json || rc=70
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

printf '{"job_id":"%s","node":"%s","arm":"%s","fusion_lambda":%s,"training_lr":"%s","parent_sha":"%s","recipe_sha":"%s","image_id":"%s","job_root":"%s","relay_root":"%s"}\n' \
    "$SLURM_JOB_ID" "$SLURMD_NODENAME" "$LAMBDA_ARM" "$FUSION_LAMBDA" "$TRAINING_LR" \
    "$LAMBDA_MATRIX_PARENT_SHA" "$LAMBDA_MATRIX_RECIPE_SHA" "$LAMBDA_MATRIX_IMAGE_ID" \
    "$job_root" "$relay_job_root" >"$job_root/admission.json"
relay_files admission.json

export REPO_HOST="$repo_host" REPO_CONTAINER=/workspace/verl DOCKER_IMAGE="$local_image_ref"
export DOCKER_CONTAINER_NAME="$container_name" DATA1_HOST="$data1_host" DATA2_HOST="$data2_host"
export REPO_MOUNT_MODE=ro PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline
unset ROCR_VISIBLE_DEVICES

bash "$repo_host/scripts/l40s/run_train.sh" \
    bash recipe/on_policy_wdl_sft/math_task/run_math_qwen3_1p7b_wdl_lambda_matrix_p60.sh \
    >"$job_root/stdout.log" 2>"$job_root/stderr.log" &
training_pid=$!

python3 "$repo_host/scripts/math_wdl_first_step_gate.py" \
    --metrics-root "${artifact_root}/logs/metrics" \
    --project OnPolicyWDLSFT-Math-1P7B-Lambda-Followup-P60 \
    --run-prefix "$run_prefix" --expected-model1-gradient "$expected_gradient" \
    --slurm-job-id "$SLURM_JOB_ID" --timeout-seconds 43200 \
    --receipt "$job_root/first-step.json" >"$job_root/first-step.log" 2>&1 &
gate_pid=$!

while kill -0 "$training_pid" 2>/dev/null && kill -0 "$gate_pid" 2>/dev/null; do sleep 5; done
if ! kill -0 "$training_pid" 2>/dev/null; then
    set +e; wait "$training_pid"; training_rc=$?; set -e
    training_pid=""; kill "$gate_pid" 2>/dev/null || true; wait "$gate_pid" 2>/dev/null || true; gate_pid=""
    [ "$training_rc" -ne 0 ] || exit 66
    exit "$training_rc"
fi
wait "$gate_pid" || { gate_pid=""; kill "$training_pid" 2>/dev/null || true; wait "$training_pid" 2>/dev/null || true; training_pid=""; exit 65; }
gate_pid=""; relay_files first-step.json first-step.log || exit 70
wait "$training_pid"; training_pid=""

metrics_dir="${artifact_root}/logs/metrics/OnPolicyWDLSFT-Math-1P7B-Lambda-Followup-P60"
mapfile -t metrics_files < <(find "$metrics_dir" -maxdepth 1 -type f -name "${run_prefix}_*.jsonl" | sort)
[ "${#metrics_files[@]}" -eq 1 ] || die "expected exactly one metrics file"
metrics_path="${metrics_files[0]}"; run_name="$(basename "$metrics_path" .jsonl)"
checkpoint_root="${data1_host}/checkpoints/${run_name}"
python3 - "$metrics_path" "$checkpoint_root" "$job_root/completion.json" <<'PY'
import json, sys
from pathlib import Path
metrics, ckpt, receipt = map(Path, sys.argv[1:])
rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
terminal = [row for row in rows if int(row.get("step", -1)) == 60]
if not terminal or "val-core/model2/math7_macro/acc/mean@3" not in terminal[-1].get("data", {}):
    raise SystemExit("terminal validation metric missing")
latest = int((ckpt / "latest_checkpointed_iteration.txt").read_text().strip())
if latest != 60 or not (ckpt / "global_step_60").is_dir():
    raise SystemExit("final checkpoint missing")
receipt.write_text(json.dumps({"status": "success_complete", "run_name": ckpt.name, "observed_step": 60,
    "checkpoint": str(ckpt), "metrics": str(metrics)}, sort_keys=True) + "\n")
PY
record_gate success_complete "$checkpoint_root" "$metrics_path" 60 >"$job_root/release-gate.log" 2>&1
completion_recorded=true
relay_files completion.json release-gate.log
