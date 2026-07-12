#!/usr/bin/env bash
# Approval-gated Stage1 plateau -> short Stage2 -> Stage1-like Stage3 queue.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/qwen3_1p7b_stage123_resource_profile.sh"
source "${SCRIPT_DIR}/stage123_manifest_gate.sh"

export STAGE123_MANIFEST=${STAGE123_MANIFEST:-${REPO_ROOT}/recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml}
export STAGE123_MANIFEST_TOOL=${STAGE123_MANIFEST_TOOL:-${REPO_ROOT}/scripts/experiment_manifest.py}
export STAGE123_MANIFEST_PYTHON=${STAGE123_MANIFEST_PYTHON:-python3}
export DRY_RUN=${DRY_RUN:-1}
export STAGE123_SCRATCH_ROOT=${STAGE123_SCRATCH_ROOT:-/data-1/tmp/verl_agent_scratch/qwen3_1p7b_stage123}
mkdir -p "$STAGE123_SCRATCH_ROOT"
export STAGE123_NORMALIZED_MANIFEST=${STAGE123_NORMALIZED_MANIFEST:-${STAGE123_SCRATCH_ROOT}/stage123.normalized.json}
"$STAGE123_MANIFEST_PYTHON" "$STAGE123_MANIFEST_TOOL" render "$STAGE123_MANIFEST" --format json > "$STAGE123_NORMALIZED_MANIFEST"
manifest_get() { python3 - "$STAGE123_NORMALIZED_MANIFEST" "$1" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); v=d
for key in sys.argv[2].split('.'): v=v[key]
print(v)
PY
}
manifest_hash=$(manifest_get manifest_sha256)
export STAGE123_EXPECTED_PROFILE_HASH=$(manifest_get resource_profile.sha256)
export STAGE123_RECEIPT_MAX_AGE_SECONDS=$(manifest_get preflight.receipt_max_age_seconds)
export STAGE123_PREFLIGHT_POLICY=${STAGE123_PREFLIGHT_POLICY:-${REPO_ROOT}/$(manifest_get preflight.policy)}
export STAGE123_FORMAL_QUEUE_ID=${STAGE123_FORMAL_QUEUE_ID:-$(manifest_get experiment_id)}
if [ "$DRY_RUN" != 1 ]; then
    : "${STAGE123_PREFLIGHT_REPORT:?STAGE123_PREFLIGHT_REPORT required}"
    : "${STAGE123_PREFLIGHT_RECEIPT:?STAGE123_PREFLIGHT_RECEIPT required}"
    : "${STAGE123_DEPLOYABILITY_RECEIPT:?STAGE123_DEPLOYABILITY_RECEIPT required}"
    : "${STAGE123_CALIBRATION_REPORT:?STAGE123_CALIBRATION_REPORT required}"
    : "${STAGE123_CALIBRATION_POLICY:?STAGE123_CALIBRATION_POLICY required}"
    : "${STAGE123_CALIBRATION_HISTORY_INDEX:?STAGE123_CALIBRATION_HISTORY_INDEX required}"
    : "${STAGE123_CALIBRATION_PREDICTION_CONTRACT:?STAGE123_CALIBRATION_PREDICTION_CONTRACT required}"
fi

export BASE_CKPT_DIR=${BASE_CKPT_DIR:-/data-1/checkpoints}
[ "$(readlink -f "$BASE_CKPT_DIR")" = /data-2/checkpoints ] || { echo "ERROR: checkpoints must physically resolve to /data-2/checkpoints" >&2; exit 1; }
export FUSION_LAMBDA=${FUSION_LAMBDA:-0.8}
export QUEUE_STATUS_FILE=${QUEUE_STATUS_FILE:-${SCRIPT_DIR}/run_code_task_qwen3_1p7b_stage123_status.tsv}
export ARTIFACT_ROOT=${ARTIFACT_ROOT:-$(manifest_get paths.artifact_root)}
export STAGE2_SOURCE_TRAIN_FILE=${STAGE2_SOURCE_TRAIN_FILE:-$(manifest_get paths.source_train_file)}
export QWEN3_1P7B_MODEL_PATH=${QWEN3_1P7B_MODEL_PATH:-$(manifest_get paths.base_model)}
export BASE_MODEL_PATH="$QWEN3_1P7B_MODEL_PATH"

if [ "$DRY_RUN" != 1 ] && [ "${ALLOW_QWEN3_1P7B_STAGE123_TRAINING:-0}" != 1 ]; then
    echo "ERROR: formal execution requires ALLOW_QWEN3_1P7B_STAGE123_TRAINING=1" >&2; exit 1
fi
profile_hash=$(stage123_profile_hash)
[ "$profile_hash" = "$STAGE123_EXPECTED_PROFILE_HASH" ] || { echo "ERROR: manifest/profile hash mismatch" >&2; exit 1; }
profile_serialization=$(stage123_profile_snapshot)
export STAGE123_EXPECTED_PROFILE_SERIALIZATION="$profile_serialization"
if [ "$DRY_RUN" = 1 ]; then
    mkdir -p "$STAGE123_SCRATCH_ROOT"
    QUEUE_STATUS_FILE="$STAGE123_SCRATCH_ROOT/status.tsv"
fi
printf 'timestamp\tchain\tphase\tstatus\tdetail\n' > "$QUEUE_STATUS_FILE"
record() { printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" "$3" "$4" >> "$QUEUE_STATUS_FILE"; }

write_run_provenance() {
    local output=$1 run_id=$2 run_prefix=$3 train_file=$4 release_eligible=$5 source_json=$6
    local receipt_path=${STAGE123_PREFLIGHT_RECEIPT:-}
    python3 - "$output" "$STAGE123_NORMALIZED_MANIFEST" "$receipt_path" "$run_id" "$run_prefix" "$train_file" "$release_eligible" "$source_json" <<'PY'
import hashlib,json,sys
from pathlib import Path
out,manifest_path,receipt_path,run_id,prefix,train_file,eligible,source_json=sys.argv[1:]
m=json.load(open(manifest_path)); run=next(x for x in m['runs'] if x['id']==run_id)
receipt_hash=hashlib.sha256(Path(receipt_path).read_bytes()).hexdigest() if receipt_path and Path(receipt_path).is_file() else 'dry-run'
d={'schema_version':1,'run_id':run_id,'run_prefix':prefix,'manifest_sha256':m['manifest_sha256'],'profile_sha256':m['resource_profile']['sha256'],'train_file':train_file,'train_file_sha256':run['train_file_sha256'],'preflight_receipt_sha256':receipt_hash,'release_eligible':eligible=='true','source':json.loads(source_json)}
Path(out).parent.mkdir(parents=True,exist_ok=True); Path(out).write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
PY
}
json_object() {
    "$STAGE123_MANIFEST_PYTHON" - "$@" <<'PY'
import json,sys
items=sys.argv[1:]; data={}
for key,value,kind in zip(items[0::3],items[1::3],items[2::3]):
    data[key]=int(value) if kind=='int' else value
print(json.dumps(data,separators=(',',':')))
PY
}

find_s1() {
    local fraction=$1
    local tag=${fraction^^}
    find "$(readlink -f "$BASE_CKPT_DIR")" -maxdepth 1 -type d -name "ONPOLICY-SFT-Qwen3-1P7B-COLDSTART-${tag}-CODE-KODCODE-CTX8K-S1-BETA01-V1_*" | sort | tail -1
}

run_phase_dry() {
    local phase=$1; shift
    local output
    output=$(env DRY_RUN=1 "$@" 2>&1)
    printf '%s\n' "$output"
    grep -Fq "sha256=${profile_hash}" <<<"$output" || { echo "ERROR: ${phase} resource profile hash mismatch" >&2; exit 1; }
}

prepare_nonoverlap_shard() {
    local consumed_steps=$1 selected_steps=$2 output=$3
    local args=(
        --source "$STAGE2_SOURCE_TRAIN_FILE"
        --output "$output"
        --model-path "$QWEN3_1P7B_MODEL_PATH"
        --seed 20260604
        --stage1-steps "$consumed_steps"
        --stage1-train-batch-size "$TRAIN_PROMPT_BSZ"
        --stage2-steps "$selected_steps"
        --stage2-train-batch-size "$TRAIN_PROMPT_BSZ"
        --max-prompt-length "$MAX_PROMPT_LENGTH"
    )
    if [ -f "$output" ] && [ -f "${output%.parquet}.manifest.json" ]; then
        if [ "$DRY_RUN" = 1 ]; then
            "$STAGE123_MANIFEST_PYTHON" - "$output" "$STAGE2_SOURCE_TRAIN_FILE" "$consumed_steps" "$selected_steps" "$TRAIN_PROMPT_BSZ" "$MAX_PROMPT_LENGTH" <<'PY'
import hashlib,json,sys
from pathlib import Path
output,source,consumed_steps,selected_steps,batch_size,max_prompt=sys.argv[1:]
output=Path(output); manifest=Path(str(output).removesuffix('.parquet')+'.manifest.json')
d=json.loads(manifest.read_text()); sha=hashlib.sha256(output.read_bytes()).hexdigest()
expected_rows=int(selected_steps)*int(batch_size); expected_consumed=int(consumed_steps)*int(batch_size)
checks={
 'output path': d.get('output_path')==str(output), 'source path': d.get('source_path')==source,
 'output hash': d.get('output_sha256')==sha, 'selected rows': d.get('selected_row_count')==expected_rows,
 'stage1 consumed': d.get('stage1',{}).get('consumed_rows')==expected_consumed,
 'stage2 steps': d.get('stage2',{}).get('steps')==int(selected_steps),
 'stage2 batch': d.get('stage2',{}).get('train_batch_size')==int(batch_size),
 'max prompt': d.get('filter_settings',{}).get('max_prompt_length')==int(max_prompt),
}
failed=[name for name,ok in checks.items() if not ok]
if failed: raise SystemExit('dry-run shard manifest mismatch: '+', '.join(failed))
print(json.dumps({'status':'PASS','mode':'content-addressed-dry-run','output':str(output),'sha256':sha,'row_count':expected_rows},sort_keys=True))
PY
        else
            /data-1/verl07/run_train.sh /opt/venv/bin/python \
                recipe/on_policy_wdl_sft/code_task/create_code_stage2_nonoverlap_shard.py \
                "${args[@]}" --verify-only
        fi
    elif [ "$DRY_RUN" = 1 ]; then
        echo "ERROR: dry-run requires a prebuilt exact-budget Stage2 shard: $output" >&2
        return 1
    else
        /data-1/verl07/run_train.sh /opt/venv/bin/python \
            recipe/on_policy_wdl_sft/code_task/create_code_stage2_nonoverlap_shard.py \
            "${args[@]}"
    fi
}

mapfile -t stage2_rows < <("$STAGE123_MANIFEST_PYTHON" "$STAGE123_MANIFEST_TOOL" render "$STAGE123_MANIFEST" --format tsv | awk -F '\t' 'NR>1 && $4=="stage2"')
if [ "$DRY_RUN" != 1 ]; then
    for row in "${stage2_rows[@]}"; do
        IFS=$'\t' read -r stage2_id chain _fraction _phase _order _stage2_prefix _steps _tmux _train_file _sha _chain_root _provenance <<<"$row"
        IFS=$'\t' read -r stage3_id < <("$STAGE123_MANIFEST_PYTHON" - "$STAGE123_NORMALIZED_MANIFEST" "$chain" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); r=next(x for x in d['runs'] if x['chain']==sys.argv[2] and x['phase']=='stage3')
print(r['id'])
PY
)
        stage123_require_formal_admission "$stage2_id"
        stage123_require_formal_admission "$stage3_id"
    done
    stage123_check_machine
fi
for row in "${stage2_rows[@]}"; do
    IFS=$'\t' read -r stage2_id chain fraction _phase _order stage2_prefix STAGE2_STEPS stage2_tmux stage2_train_file stage2_sha chain_root provenance <<<"$row"
    IFS=$'\t' read -r stage3_id stage3_prefix STAGE3_STEPS stage3_tmux stage3_train_file stage3_provenance < <("$STAGE123_MANIFEST_PYTHON" - "$STAGE123_NORMALIZED_MANIFEST" "$chain" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); r=next(x for x in d['runs'] if x['chain']==sys.argv[2] and x['phase']=='stage3')
print('\t'.join(str(r[key]) for key in ('id','run_prefix','final_step','tmux_name','train_file','provenance_file')))
PY
)
    trigger=$(python3 - "$STAGE123_NORMALIZED_MANIFEST" "$stage2_id" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); print(next(x for x in d['runs'] if x['id']==sys.argv[2])['source']['handoff_step'])
PY
)
    s1_dir=$(find_s1 "$fraction")
    [ -n "$s1_dir" ] || { echo "ERROR: missing Stage1 checkpoint for $fraction" >&2; exit 1; }
    [ -d "$s1_dir/global_step_${trigger}/actor" ] || { echo "ERROR: missing ${s1_dir}/global_step_${trigger}/actor" >&2; exit 1; }
    s1_merged="${ARTIFACT_ROOT}/${chain}/beta01/stage1_model2"
    stage2_model2="${ARTIFACT_ROOT}/${chain}/stage2_final_model2"
    [ "$DRY_RUN" = 1 ] && chain_root="${STAGE123_SCRATCH_ROOT}/${chain}"
    [ "$DRY_RUN" = 1 ] && provenance="${chain_root}/${stage2_id}.provenance.json"
    [ "$DRY_RUN" = 1 ] && stage3_provenance="${chain_root}/${stage3_id}.provenance.json"
    stage3_offset=$((trigger + STAGE2_STEPS))
    prepare_nonoverlap_shard "$trigger" "$STAGE2_STEPS" "$stage2_train_file"
    prepare_nonoverlap_shard "$stage3_offset" "$STAGE3_STEPS" "$stage3_train_file"
    mkdir -p "$(dirname "$provenance")"
    write_run_provenance "$provenance" "$stage2_id" "$stage2_prefix" "$stage2_train_file" false \
      "$(json_object type stage1_checkpoint str checkpoint "$s1_dir/global_step_$trigger" str handoff_step "$trigger" int)"

    echo "[STAGE123 CHAIN] $chain"
    record "$chain" stage1 source_verified "checkpoint=$s1_dir/global_step_$trigger"
    if [ "$DRY_RUN" = 1 ]; then
      run_phase_dry STAGE1 env STAGE123_RUN_ID="$stage2_id" RUN_PREFIX="CODE-S1-QWEN3-1P7B-STAGE123-${chain^^}-SOURCE-CHECK" \
        INIT_MODEL_PATH="$QWEN3_1P7B_MODEL_PATH" TOTAL_TRAINING_STEPS="$trigger" WDL_SFT_BETA=0.1 \
        bash "${SCRIPT_DIR}/run_s1_code_qwen3_1p7b_stage123_common.sh"
      run_phase_dry STAGE2 env STAGE123_RUN_ID="$stage2_id" RUN_PREFIX="$stage2_prefix" STAGE1_RUN_PREFIX="$(basename "$s1_dir" | sed -E 's/_[0-9]+$//')" \
        EXPECTED_STAGE1_RUN_PREFIX="$(basename "$s1_dir" | sed -E 's/_[0-9]+$//')" STAGE1_CKPT_DIR="$s1_dir" STAGE2_HANDOFF_STEP="$trigger" \
        WDL_SFT_BETA=0.1 EXPECTED_STAGE1_BETA=0.1 CODE_TRAIN_FILE="$stage2_train_file" TRAIN_FILE="$stage2_train_file" \
        TOTAL_TRAINING_STEPS="$STAGE2_STEPS" MERGED_MODEL2_DIR="$s1_merged" MODEL2_CACHE_TAG="$chain" \
        bash "${SCRIPT_DIR}/run_s2_code_qwen3_1p7b_stage123_common.sh"
      dry_stage2_model2="${chain_root}/stage2_final_model2"
      mkdir -p "$dry_stage2_model2"; touch "$dry_stage2_model2/config.json" "$dry_stage2_model2/model.safetensors"
      write_run_provenance "$provenance" "$stage2_id" "$stage2_prefix" "$stage2_train_file" false \
        "$(json_object type stage1_checkpoint str checkpoint "$s1_dir/global_step_$trigger" str extracted_model2 "$dry_stage2_model2" str)"
      write_run_provenance "$stage3_provenance" "$stage3_id" "$stage3_prefix" "$stage3_train_file" false \
        "$(json_object type stage2_model2 str run_id "$stage2_id" str model2 "$dry_stage2_model2" str)"
      run_phase_dry STAGE3 env STAGE123_RUN_ID="$stage3_id" RUN_PREFIX="$stage3_prefix" STAGE2_MODEL2_PATH="$dry_stage2_model2" STAGE2_PROVENANCE_FILE="$provenance" \
        CODE_TRAIN_FILE="$stage3_train_file" TRAIN_FILE="$stage3_train_file" DATA_SHUFFLE=False \
        TOTAL_TRAINING_STEPS="$STAGE3_STEPS" bash "${SCRIPT_DIR}/run_s3_code_qwen3_1p7b_stage123_common.sh"
      record "$chain" all dry_run_pass "profile_hash=$profile_hash"
      continue
    fi
    launch_and_wait() {
        local tmux_name=$1 prefix=$2 final_step=$3 wrapper=$4; shift 4
        local log_file="${SCRIPT_DIR}/${prefix}.log"
        local deadline_root="${STAGE123_DEADLINE_ROOT:-/data-2/experiment_registry/validation_deadlines}"
        local ownership_file="${deadline_root}/${prefix}.ownership.json"
        local deadline_report="${deadline_root}/${prefix}.deadline.json"
        local validation_ready_epoch=""
        local container_name="stage123-${tmux_name//_/-}"
        local env_cmd="" item
        for item in "$@"; do printf -v env_cmd '%s %q' "$env_cmd" "$item"; done
        tmux has-session -t "$tmux_name" 2>/dev/null && { echo "ERROR: tmux already exists: $tmux_name" >&2; return 1; }
        tmux new-session -d -s "$tmux_name" \
            "cd '$REPO_ROOT' && env DOCKER_CONTAINER_NAME='$container_name' $env_cmd /data-1/verl07/run_train.sh bash '$wrapper' 2>&1 | tee -a '$log_file'"
        mkdir -p "$deadline_root"
        while tmux has-session -t "$tmux_name" 2>/dev/null; do
            if [ -z "$validation_ready_epoch" ] && grep -Eq 'validation batch [0-9]+/[0-9]+ start:' "$log_file" 2>/dev/null; then
                validation_ready_epoch=$(date +%s)
                local pane_pid container_pid descendants gpu_pids
                pane_pid=$(tmux list-panes -t "$tmux_name" -F '#{pane_pid}' | head -1)
                container_pid=$(docker inspect -f '{{.State.Pid}}' "$container_name" 2>/dev/null || true)
                [ -n "$container_pid" ] && [ "$container_pid" != 0 ] || {
                    echo "ERROR: cannot establish Docker ownership for $prefix container=$container_name" >&2
                    record "$prefix" validation blocked "docker_ownership_unproven container=$container_name"
                    return 125
                }
                descendants=$(python3 - "$container_pid" <<'PY'
import subprocess,sys
root=int(sys.argv[1]); seen=set(); frontier=[root]
while frontier:
    parent=frontier.pop()
    out=subprocess.run(['pgrep','-P',str(parent)],text=True,capture_output=True).stdout.split()
    for value in out:
        pid=int(value)
        if pid not in seen: seen.add(pid); frontier.append(pid)
seen.add(root); print(','.join(map(str,sorted(seen))))
PY
)
                gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9\n' | paste -sd, - || true)
                python3 - "$ownership_file" "$prefix" "$validation_ready_epoch" "$tmux_name" "$pane_pid" "$container_name" "$container_pid" "$descendants" "$gpu_pids" <<'PY'
import json,sys
out,run_id,ready,tmux_name,pane,container,container_pid,desc,gpu=sys.argv[1:]
descendants={int(x) for x in desc.split(',') if x}; descendants.add(int(pane))
gpu_set={int(x) for x in gpu.split(',') if x}
data={'schema_version':1,'run_id':run_id,'validation_ready_epoch_s':int(ready),'deadline_seconds':1800,'first_training_step':0,'complete_validation_metrics':False,'tmux_sessions':[tmux_name],'process_group_id':0,'container_init_pid':int(container_pid),'descendant_pids':sorted(descendants),'gpu_pids':sorted(descendants & gpu_set),'docker_containers':[container]}
open(out,'w').write(json.dumps(data,indent=2,sort_keys=True)+'\n')
PY
                record "$prefix" validation deadline_started "ownership=$ownership_file deadline_seconds=1800"
            fi
            if [ -n "$validation_ready_epoch" ] && [ $(( $(date +%s) - validation_ready_epoch )) -ge 1800 ]; then
                python3 "$REPO_ROOT/scripts/validation_deadline_controller.py" --ownership "$ownership_file" --report "$deadline_report" --grace-seconds "${STAGE123_VALIDATION_GRACE_SECONDS:-30}" || true
                record "$prefix" validation blocked "deadline_report=$deadline_report"
                echo "ERROR: $prefix validation exceeded 30-minute hard wall; evidence=$deadline_report" >&2
                return 124
            fi
            sleep "${QUEUE_POLL_SEC:-30}"
        done
        local ckpt step metrics
        ckpt=$(find "$(readlink -f "$BASE_CKPT_DIR")" -maxdepth 1 -type d -name "${prefix}_*" | sort | tail -1)
        [ -n "$ckpt" ] || { echo "ERROR: no checkpoint for $prefix" >&2; return 1; }
        step=$(tr -dc '0-9' < "$ckpt/latest_checkpointed_iteration.txt" 2>/dev/null || true)
        metrics=$(find "$REPO_ROOT/recipe/on_policy_wdl_sft" -path "*/metrics/OnPolicyWDLSFT-CodeTask/$(basename "$ckpt").jsonl" -print | head -1)
        [ "${step:-0}" -ge "$final_step" ] && [ -n "$metrics" ] || {
            echo "ERROR: $prefix stopped early step=${step:-none} metrics=${metrics:-none}" >&2; return 1; }
        printf '%s\n' "$ckpt"
    }

    stage123_require_formal_admission "$stage2_id"
    stage2_ckpt=$(launch_and_wait "$stage2_tmux" "$stage2_prefix" "$STAGE2_STEPS" \
        /workspace/verl/recipe/on_policy_wdl_sft/code_task/run_s2_code_qwen3_1p7b_stage123_common.sh \
        RUN_PREFIX="$stage2_prefix" STAGE1_RUN_PREFIX="$(basename "$s1_dir" | sed -E 's/_[0-9]+$//')" \
        EXPECTED_STAGE1_RUN_PREFIX="$(basename "$s1_dir" | sed -E 's/_[0-9]+$//')" STAGE1_CKPT_DIR="$s1_dir" \
        STAGE2_HANDOFF_STEP="$trigger" WDL_SFT_BETA=0.1 EXPECTED_STAGE1_BETA=0.1 \
        CODE_TRAIN_FILE="$stage2_train_file" TRAIN_FILE="$stage2_train_file" TOTAL_TRAINING_STEPS="$STAGE2_STEPS" \
        MERGED_MODEL2_DIR="$s1_merged" MODEL2_CACHE_TAG="$chain" FUSION_LAMBDA="$FUSION_LAMBDA" \
        STAGE123_RUN_ID="$stage2_id" STAGE123_MANIFEST="$STAGE123_MANIFEST" STAGE123_NORMALIZED_MANIFEST="$STAGE123_NORMALIZED_MANIFEST" STAGE123_PREFLIGHT_REPORT="$STAGE123_PREFLIGHT_REPORT" STAGE123_PREFLIGHT_RECEIPT="$STAGE123_PREFLIGHT_RECEIPT" STAGE123_PREFLIGHT_POLICY="$STAGE123_PREFLIGHT_POLICY" STAGE123_RECEIPT_MAX_AGE_SECONDS="$STAGE123_RECEIPT_MAX_AGE_SECONDS" STAGE123_EXPECTED_PROFILE_HASH="$profile_hash" \
        STAGE123_DEPLOYABILITY_RECEIPT="$STAGE123_DEPLOYABILITY_RECEIPT" STAGE123_FORMAL_QUEUE_ID="$STAGE123_FORMAL_QUEUE_ID" STAGE123_CALIBRATION_REPORT="$STAGE123_CALIBRATION_REPORT" STAGE123_CALIBRATION_POLICY="$STAGE123_CALIBRATION_POLICY" STAGE123_CALIBRATION_HISTORY_INDEX="$STAGE123_CALIBRATION_HISTORY_INDEX" STAGE123_CALIBRATION_PREDICTION_CONTRACT="$STAGE123_CALIBRATION_PREDICTION_CONTRACT" STAGE123_CALIBRATION_SEMANTIC_CONTRACT="${STAGE123_CALIBRATION_SEMANTIC_CONTRACT:-}" STAGE123_DEPLOYABILITY_RECEIPT_MAX_AGE_SECONDS="${STAGE123_DEPLOYABILITY_RECEIPT_MAX_AGE_SECONDS:-86400}" STAGE123_DEPLOYABILITY_RECEIPT_FUTURE_SKEW_SECONDS="${STAGE123_DEPLOYABILITY_RECEIPT_FUTURE_SKEW_SECONDS:-300}")

    joint_dir="${chain_root}/stage2_final_joint"
    rm -rf "$joint_dir" "$stage2_model2"
    /data-1/verl07/run_train.sh /opt/venv/bin/python -m verl.model_merger merge --backend fsdp \
        --local_dir "$stage2_ckpt/global_step_${STAGE2_STEPS}/actor" --target_dir "$joint_dir" --trust-remote-code
    /data-1/verl07/run_train.sh /opt/venv/bin/python recipe/joint_training/extract_sub_model.py \
        --joint_model_path "$joint_dir" --output_path "$stage2_model2" --sub_model_index 1
    write_run_provenance "$provenance" "$stage2_id" "$stage2_prefix" "$stage2_train_file" true \
      "$(json_object type stage2_complete str checkpoint "$stage2_ckpt" str extracted_model2 "$stage2_model2" str)"
    write_run_provenance "$stage3_provenance" "$stage3_id" "$stage3_prefix" "$stage3_train_file" false \
      "$(json_object type stage2_model2 str run_id "$stage2_id" str model2 "$stage2_model2" str)"

    stage123_require_formal_admission "$stage3_id"
    launch_and_wait "$stage3_tmux" "$stage3_prefix" "$STAGE3_STEPS" \
        /workspace/verl/recipe/on_policy_wdl_sft/code_task/run_s3_code_qwen3_1p7b_stage123_common.sh \
        RUN_PREFIX="$stage3_prefix" STAGE2_MODEL2_PATH="$stage2_model2" STAGE2_PROVENANCE_FILE="$provenance" \
        CODE_TRAIN_FILE="$stage3_train_file" TRAIN_FILE="$stage3_train_file" DATA_SHUFFLE=False TOTAL_TRAINING_STEPS="$STAGE3_STEPS" \
        STAGE123_RUN_ID="$stage3_id" STAGE123_MANIFEST="$STAGE123_MANIFEST" STAGE123_NORMALIZED_MANIFEST="$STAGE123_NORMALIZED_MANIFEST" STAGE123_PREFLIGHT_REPORT="$STAGE123_PREFLIGHT_REPORT" STAGE123_PREFLIGHT_RECEIPT="$STAGE123_PREFLIGHT_RECEIPT" STAGE123_PREFLIGHT_POLICY="$STAGE123_PREFLIGHT_POLICY" STAGE123_RECEIPT_MAX_AGE_SECONDS="$STAGE123_RECEIPT_MAX_AGE_SECONDS" STAGE123_EXPECTED_PROFILE_HASH="$profile_hash" \
        STAGE123_DEPLOYABILITY_RECEIPT="$STAGE123_DEPLOYABILITY_RECEIPT" STAGE123_FORMAL_QUEUE_ID="$STAGE123_FORMAL_QUEUE_ID" STAGE123_CALIBRATION_REPORT="$STAGE123_CALIBRATION_REPORT" STAGE123_CALIBRATION_POLICY="$STAGE123_CALIBRATION_POLICY" STAGE123_CALIBRATION_HISTORY_INDEX="$STAGE123_CALIBRATION_HISTORY_INDEX" STAGE123_CALIBRATION_PREDICTION_CONTRACT="$STAGE123_CALIBRATION_PREDICTION_CONTRACT" STAGE123_CALIBRATION_SEMANTIC_CONTRACT="${STAGE123_CALIBRATION_SEMANTIC_CONTRACT:-}" STAGE123_DEPLOYABILITY_RECEIPT_MAX_AGE_SECONDS="${STAGE123_DEPLOYABILITY_RECEIPT_MAX_AGE_SECONDS:-86400}" STAGE123_DEPLOYABILITY_RECEIPT_FUTURE_SKEW_SECONDS="${STAGE123_DEPLOYABILITY_RECEIPT_FUTURE_SKEW_SECONDS:-300}" >/dev/null
    write_run_provenance "$stage3_provenance" "$stage3_id" "$stage3_prefix" "$stage3_train_file" true \
      "$(json_object type stage3_complete str source_run_id "$stage2_id" str init_model2 "$stage2_model2" str)"
    record "$chain" all completed "profile_hash=$profile_hash"
done
echo "[STAGE123 QUEUE] DRY_RUN PASS manifest_hash=$manifest_hash profile_hash=$profile_hash status=$QUEUE_STATUS_FILE"
