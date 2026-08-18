#!/usr/bin/env bash
# Host-side Qwen3-1.7B code format cold-start SFT fraction queue.
#
# Builds 25/50/100 percent KodCode format-SFT datasets, trains one SFT model per
# fraction, merges the final checkpoint to HF format, then evaluates the merged
# model as a Stage1-start candidate on official code validation sets.
set -euo pipefail

REPO_HOST=${REPO_HOST:-/data-1/verl07/verl}
REPO_CONTAINER=${REPO_CONTAINER:-/workspace/verl}
DOCKER_IMAGE=${DOCKER_IMAGE:-verl-harness:latest}
CKPT_ROOT=${CKPT_ROOT:-/data-1/checkpoints/format_cold_start_fraction_cot_v3}
MERGED_ROOT=${MERGED_ROOT:-/data-1/model_weights/format_cold_start_fraction_cot_v3}
DATA_ROOT=${DATA_ROOT:-/data-1/dataset/code/format_cold_start_fraction_cot_v3}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data-1/eval_outputs/code_task/qwen3_1p7b_coldstart_sft_fraction_cot_v3}
LOG_FILE=${LOG_FILE:-"${REPO_HOST}/recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_coldstart_sft_fraction_queue.log"}
QUEUE_STATUS_FILE=${QUEUE_STATUS_FILE:-"${LOG_FILE%.log}_status.tsv"}
QUEUE_TMUX=${QUEUE_TMUX:-code_task_qwen3_1p7b_coldstart_sft_fraction_queue}
QUEUE_POLL_SEC=${QUEUE_POLL_SEC:-60}
START_INDEX=${START_INDEX:-0}
END_INDEX=${END_INDEX:-2}
ALLOW_RESUME=${ALLOW_RESUME:-0}
ALLOW_OVERWRITE_MERGED=${ALLOW_OVERWRITE_MERGED:-0}
ALLOW_OVERWRITE_DATA=${ALLOW_OVERWRITE_DATA:-0}
ALLOW_EVAL_RERUN=${ALLOW_EVAL_RERUN:-0}
MIN_FREE_GB=${MIN_FREE_GB:-250}
WXPUSHER_NOTIFY=${WXPUSHER_NOTIFY:-1}
WXPUSHER_SCRIPT=${WXPUSHER_SCRIPT:-/root/agent-core/skills/wxpusher-notify/scripts/wxpusher_notify.py}

MODEL_PATH=${MODEL_PATH:-/data-1/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
BASE_TOTAL_SAMPLES=${BASE_TOTAL_SAMPLES:-10000}
BASE_TOTAL_STEPS=${BASE_TOTAL_STEPS:-120}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
LR=${LR:-5e-6}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-5}
SAVE_FREQ=${SAVE_FREQ:-0}
MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP:-1}
DATA_SEED=${DATA_SEED:-20260706}
TRAIN_SEED=${TRAIN_SEED:-$DATA_SEED}
DATA_SHUFFLE=${DATA_SHUFFLE:-False}

EVAL_BENCHMARKS=${EVAL_BENCHMARKS:-"humaneval mbpp livecodebench"}
EVAL_N_SAMPLES=${EVAL_N_SAMPLES:-3}
EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-0.2}
EVAL_TOP_P=${EVAL_TOP_P:-0.95}
EVAL_TOP_K=${EVAL_TOP_K:--1}
EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-8192}
EVAL_TENSOR_PARALLEL=${EVAL_TENSOR_PARALLEL:-4}
EVAL_GPU_MEMORY_UTILIZATION=${EVAL_GPU_MEMORY_UTILIZATION:-0.75}
CODE_OFFICIAL_EVAL_PARALLEL=${CODE_OFFICIAL_EVAL_PARALLEL:-8}
LCB_RELEASE_VERSION=${LCB_RELEASE_VERSION:-release_v5}

RUN_LABELS=("frac25" "frac50" "frac100")
RUN_FRACTIONS=("0.25" "0.50" "1.00")
RUN_SAMPLES=(
  "${FRAC25_SAMPLES:-2500}"
  "${FRAC50_SAMPLES:-5000}"
  "${FRAC100_SAMPLES:--1}"
)
RUN_STEPS=(
  "${FRAC25_STEPS:-30}"
  "${FRAC50_STEPS:-60}"
  "${FRAC100_STEPS:-120}"
)
RUN_PREFIXES=(
  "${FRAC25_RUN_PREFIX:-SFT-FORMAT-COLDSTART-Qwen3-1P7B-CODE-KODCODE-FRAC25-COT-V3}"
  "${FRAC50_RUN_PREFIX:-SFT-FORMAT-COLDSTART-Qwen3-1P7B-CODE-KODCODE-FRAC50-COT-V3}"
  "${FRAC100_RUN_PREFIX:-SFT-FORMAT-COLDSTART-Qwen3-1P7B-CODE-KODCODE-FRAC100-COT-V3}"
)
TMUX_NAMES=(
  "format_cold_start_sft_q17b_code_frac25"
  "format_cold_start_sft_q17b_code_frac50"
  "format_cold_start_sft_q17b_code_frac100"
)

if [ "${DRY_RUN:-0}" != "1" ] && [ "${ALLOW_QWEN3_1P7B_COLDSTART_SFT_FRACTION_QUEUE:-0}" != "1" ]; then
    echo "[coldstart-sft-fraction] ERROR: non-dry-run requires ALLOW_QWEN3_1P7B_COLDSTART_SFT_FRACTION_QUEUE=1" >&2
    exit 1
fi

log() { mkdir -p "$(dirname "$LOG_FILE")"; echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE" >&2; }

record_status() {
    mkdir -p "$(dirname "$QUEUE_STATUS_FILE")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" "$2" "$3" "$4" "$5" >>"$QUEUE_STATUS_FILE"
}

notify() {
    local title="$1" body="$2"
    if [ "$WXPUSHER_NOTIFY" != "1" ] || [ "${DRY_RUN:-0}" = "1" ]; then
        return
    fi
    [ -f "$WXPUSHER_SCRIPT" ] && python3 "$WXPUSHER_SCRIPT" --title "$title" --body "$body" >>"$LOG_FILE" 2>&1 || true
}

get_df_target() {
    local path="$1"
    while [ ! -e "$path" ] && [ "$path" != "/" ]; do
        path=$(dirname "$path")
    done
    printf '%s\n' "$path"
}

free_gb() {
    local target
    target=$(get_df_target "$1")
    df -Pk "$target" | awk 'NR==2 {print int($4 / 1024 / 1024)}'
}

disk_gate() {
    [ "${DRY_RUN:-0}" = "1" ] && return 0
    local free
    free=$(free_gb "$CKPT_ROOT")
    if [ "$free" -lt "$MIN_FREE_GB" ]; then
        log "ERROR disk gate failed: target=${CKPT_ROOT} free=${free}G need>=${MIN_FREE_GB}G"
        record_status "${CURRENT_IDX:-NA}" "${CURRENT_LABEL:-NA}" "${CURRENT_PREFIX:-NA}" "blocked_disk" "free=${free}G need>=${MIN_FREE_GB}G"
        notify "Qwen3-1.7B coldstart SFT fraction blocked" "Disk gate failed: free=${free}G need>=${MIN_FREE_GB}G"
        exit 1
    fi
}

latest_run_dir() {
    [ -d "$CKPT_ROOT" ] || return 0
    find "$CKPT_ROOT" -maxdepth 1 -type d -name "$1*" 2>/dev/null | sort | tail -1
}

latest_step() {
    local run_dir="$1"
    [ -n "$run_dir" ] && [ -d "$run_dir" ] || { echo 0; return; }
    if [ -f "${run_dir}/latest_checkpointed_iteration.txt" ]; then
        tr -dc '0-9' <"${run_dir}/latest_checkpointed_iteration.txt"
        return
    fi
    find "$run_dir" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1 | awk '{print $1 + 0}'
}

merged_ready() {
    [ -f "$1/model.safetensors" ] || [ -f "$1/model.safetensors.index.json" ] || [ -f "$1/pytorch_model.bin" ] || [ -f "$1/pytorch_model.bin.index.json" ]
}

prepare_dataset() {
    local idx="$1"
    local label="${RUN_LABELS[$idx]}"
    local samples="${RUN_SAMPLES[$idx]}"
    local output="${DATA_ROOT}/kodcode_light_sft_messages_${label}.parquet"
    local container_script="${REPO_CONTAINER}/recipe/on_policy_wdl_sft/format_cold_start/prepare_code_kodcode_sft_dataset.py"
    if [ -f "$output" ] && [ "$ALLOW_OVERWRITE_DATA" != "1" ]; then
        if ! rg -q '"format_cold_start": "code-cot-python-answer-v3"' "${output}.manifest.json" 2>/dev/null; then
            log "ERROR refusing stale or unfiltered dataset: ${output}; set ALLOW_OVERWRITE_DATA=1 to regenerate CoT-v3 data"
            exit 1
        fi
        log "CoT-v3 dataset exists; verify ${label}: ${output}"
    elif [ "${DRY_RUN:-0}" = "1" ]; then
        log "DRY_RUN would prepare dataset ${label}: samples=${samples} output=${output}"
    else
        docker run --rm --gpus all --ipc=host --network=host --shm-size=64g \
            -v /data-1:/data-1 -v /data-2:/data-2 -v "$REPO_HOST":"$REPO_CONTAINER" \
            -w "$REPO_CONTAINER" "$DOCKER_IMAGE" \
            python "$container_script" --output "$output" --max-samples "$samples" --seed "$DATA_SEED" >&2
    fi
    if [ "${DRY_RUN:-0}" != "1" ] || [ -f "$output" ]; then
        docker run --rm --gpus all --ipc=host --network=host --shm-size=64g \
            -v /data-1:/data-1 -v /data-2:/data-2 -v "$REPO_HOST":"$REPO_CONTAINER" \
            -w "$REPO_CONTAINER" "$DOCKER_IMAGE" \
            python "$container_script" --output "$output" --verify-only >&2
    fi
    printf '%s\n' "$output"
}

launch_train() {
    local idx="$1"
    local label="${RUN_LABELS[$idx]}"
    local prefix="${RUN_PREFIXES[$idx]}"
    local tmux_name="${TMUX_NAMES[$idx]}"
    local train_file="$2"
    local final="${RUN_STEPS[$idx]}"
    local host_script="${REPO_HOST}/recipe/on_policy_wdl_sft/format_cold_start/run_sft_code_qwen3_1p7b_kodcode_format.sh"
    local container_script="${REPO_CONTAINER}/recipe/on_policy_wdl_sft/format_cold_start/run_sft_code_qwen3_1p7b_kodcode_format.sh"
    local run_dir step save_freq
    run_dir=$(latest_run_dir "$prefix")
    step=$(latest_step "$run_dir")
    if [ -n "$run_dir" ] && [ "$step" -ge "$final" ]; then
        log "training already complete ${label}: step=${step} final=${final} run_dir=${run_dir}"
        return
    fi
    if [ -n "$run_dir" ] && [ "$ALLOW_RESUME" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
        log "ERROR partial checkpoint exists and ALLOW_RESUME=0: ${run_dir}"
        record_status "$idx" "$label" "$prefix" "blocked_partial_checkpoint" "run_dir=${run_dir};step=${step};final=${final}"
        exit 1
    fi
    if [ "${DRY_RUN:-0}" = "1" ]; then
        log "DRY_RUN validate launcher ${label}: steps=${final} train_file=${train_file}"
        DRY_RUN=1 TRAIN_FILE="$train_file" RUN_PREFIX="$prefix" MODEL_PATH="$MODEL_PATH" CKPT_ROOT="$CKPT_ROOT" TOTAL_TRAINING_STEPS="$final" TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" LR="$LR" LR_WARMUP_STEPS="$LR_WARMUP_STEPS" TRAIN_SEED="$TRAIN_SEED" DATA_SHUFFLE="$DATA_SHUFFLE" SAVE_FREQ="$final" MAX_CKPT_TO_KEEP="$MAX_CKPT_TO_KEEP" bash "$host_script"
        return
    fi
    if tmux has-session -t "$tmux_name" 2>/dev/null; then
        log "adopt active training ${label}: ${tmux_name}"
        return
    fi
    save_freq="$SAVE_FREQ"
    [ "$save_freq" = "0" ] && save_freq="$final"
    log "launch training ${label}: tmux=${tmux_name} steps=${final} samples=${RUN_SAMPLES[$idx]} train_file=${train_file}"
    tmux new-session -d -s "$tmux_name" \
        "docker run --rm --gpus all --ipc=host --network=host --shm-size=64g -v /data-1:/data-1 -v /data-2:/data-2 -v '$REPO_HOST':'$REPO_CONTAINER' -w '$REPO_CONTAINER' '$DOCKER_IMAGE' bash -lc \"ALLOW_FORMAT_COLD_START_SFT=1 MODEL_PATH='$MODEL_PATH' TRAIN_FILE='$train_file' RUN_PREFIX='$prefix' CKPT_ROOT='$CKPT_ROOT' TOTAL_TRAINING_STEPS='$final' TRAIN_BATCH_SIZE='$TRAIN_BATCH_SIZE' LR='$LR' LR_WARMUP_STEPS='$LR_WARMUP_STEPS' TRAIN_SEED='$TRAIN_SEED' DATA_SHUFFLE='$DATA_SHUFFLE' SAVE_FREQ='$save_freq' MAX_CKPT_TO_KEEP='$MAX_CKPT_TO_KEEP' bash '$container_script'\" 2>&1 | tee -a '$LOG_FILE'"
}

wait_train() {
    local idx="$1"
    local label="${RUN_LABELS[$idx]}"
    local prefix="${RUN_PREFIXES[$idx]}"
    local tmux_name="${TMUX_NAMES[$idx]}"
    local final="${RUN_STEPS[$idx]}"
    [ "${DRY_RUN:-0}" = "1" ] && return 0
    while tmux has-session -t "$tmux_name" 2>/dev/null; do
        local run_dir step
        run_dir=$(latest_run_dir "$prefix")
        step=$(latest_step "$run_dir")
        log "waiting training ${label}: step=${step} final=${final} run_dir=${run_dir:-none}"
        sleep "$QUEUE_POLL_SEC"
    done
    local run_dir step
    run_dir=$(latest_run_dir "$prefix")
    step=$(latest_step "$run_dir")
    if [ -z "$run_dir" ] || [ "$step" -lt "$final" ]; then
        log "ERROR training stopped before final checkpoint: ${label} step=${step} final=${final} run_dir=${run_dir:-none}"
        record_status "$idx" "$label" "$prefix" "failed_training" "step=${step};final=${final};run_dir=${run_dir:-none}"
        exit 1
    fi
    record_status "$idx" "$label" "$prefix" "trained" "step=${step};run_dir=${run_dir}"
}

merge_model() {
    local idx="$1"
    local label="${RUN_LABELS[$idx]}"
    local prefix="${RUN_PREFIXES[$idx]}"
    local target="${MERGED_ROOT}/qwen3-1p7b-kodcode-format-sft-${label}"
    local run_dir
    if [ "${DRY_RUN:-0}" = "1" ]; then
        log "DRY_RUN would merge ${label} -> ${target}"
        printf '%s\n' "$target"
        return
    fi
    if merged_ready "$target" && [ "$ALLOW_OVERWRITE_MERGED" != "1" ]; then
        log "merged model exists; reuse ${label}: ${target}"
        printf '%s\n' "$target"
        return
    fi
    run_dir=$(latest_run_dir "$prefix")
    [ -n "$run_dir" ] || { log "ERROR missing run dir for merge: $prefix"; exit 1; }
    local overwrite=()
    [ "$ALLOW_OVERWRITE_MERGED" = "1" ] && overwrite+=(--overwrite)
    docker run --rm --ipc=host --network=host --shm-size=64g \
        -v /data-1:/data-1 \
        -v /data-2:/data-2 \
        -v "$REPO_HOST":"$REPO_CONTAINER" \
        -w "$REPO_CONTAINER" \
        "$DOCKER_IMAGE" \
        bash "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/format_cold_start/merge_sft_checkpoint.sh" \
            --checkpoint-dir "$run_dir" \
            --target-dir "$target" \
            "${overwrite[@]}" >&2
    record_status "$idx" "$label" "$prefix" "merged" "target=${target}"
    printf '%s\n' "$target"
}

eval_model() {
    local idx="$1"
    local label="${RUN_LABELS[$idx]}"
    local merged="$2"
    local benchmark case_dir summary
    if [ "${DRY_RUN:-0}" = "1" ]; then
        for benchmark in $EVAL_BENCHMARKS; do
            log "DRY_RUN would eval ${label}: benchmark=${benchmark} merged=${merged} output=${OUTPUT_ROOT}/${label}/${benchmark}"
        done
        return
    fi
    merged_ready "$merged" || { log "ERROR merged model is not ready: $merged"; exit 1; }
    for benchmark in $EVAL_BENCHMARKS; do
        case_dir="${OUTPUT_ROOT}/${label}/${benchmark}"
        summary="${case_dir}/official_summary.json"
        if [ -s "$summary" ] && [ "$ALLOW_EVAL_RERUN" != "1" ]; then
            log "eval exists; reuse ${label}/${benchmark}: ${summary}"
            record_status "$idx" "$label" "$benchmark" "eval_reused" "$summary"
            continue
        fi
        disk_gate
        log "eval start ${label}/${benchmark}: merged=${merged}"
        docker run --rm --gpus all --ipc=host --network=host --shm-size=64g \
            -v /data-1:/data-1 \
            -v /data-2:/data-2 \
            -v "$REPO_HOST":"$REPO_CONTAINER" \
            -w "$REPO_CONTAINER" \
            "$DOCKER_IMAGE" \
            bash -lc "set -euo pipefail
                export PROJECT_CACHE_ROOT=/data-1/.cache
                export HF_HOME=/data-1/.cache/huggingface
                export HF_DATASETS_CACHE=/data-1/.cache/huggingface/datasets
                export HUGGINGFACE_HUB_CACHE=/data-1/.cache/huggingface/hub
                export TRANSFORMERS_CACHE=/data-1/.cache/huggingface
                export XDG_CACHE_HOME=/data-1/.cache
                export CODE_OFFICIAL_SOURCE_ROOT=/data-1/dataset/code/official_sources
                export BIGCODEBENCH_OVERRIDE_PATH=/data-1/dataset/code/official_sources/bigcodebench/BigCodeBench-v0.1.4.jsonl
                export HF_HUB_OFFLINE=1
                export HF_DATASETS_OFFLINE=1
                case '$benchmark' in
                    humaneval) validation=/data-1/dataset/code/verl_rl/online_full_humaneval_plus/official_humaneval_plus_val.parquet; converted_ext=jsonl ;;
                    mbpp) validation=/data-1/dataset/code/verl_rl/online_full_mbpp_plus/official_mbpp_plus_val.parquet; converted_ext=jsonl ;;
                    livecodebench)
                        case '${LCB_RELEASE_VERSION}' in
                            release_v1) validation=/data-1/dataset/code/verl_rl/online_full_livecodebench/official_livecodebench_val.parquet ;;
                            release_v5) validation=/data-1/dataset/code/verl_rl/online_full_livecodebench_v5/official_livecodebench_val.parquet ;;
                            *) validation=/data-1/dataset/code/verl_rl/online_full_livecodebench_${LCB_RELEASE_VERSION}/official_livecodebench_val.parquet ;;
                        esac
                        converted_ext=json
                        ;;
                    *) echo 'unsupported benchmark: $benchmark' >&2; exit 2 ;;
                esac
                mkdir -p '$case_dir'
                raw='$case_dir/raw_generations_n${EVAL_N_SAMPLES}.jsonl'
                converted='$case_dir/${benchmark}_samples_n${EVAL_N_SAMPLES}.'\"\$converted_ext\"
                PYTHONPATH='$REPO_CONTAINER:\${PYTHONPATH:-}' python3 -u recipe/on_policy_wdl_sft/code_task/eval_code_vllm.py \
                    --model '$merged' \
                    --validation-parquet \"\$validation\" \
                    --output \"\$raw\" \
                    --summary '$case_dir/generation_summary.json' \
                    --tensor-parallel '${EVAL_TENSOR_PARALLEL}' \
                    --n '${EVAL_N_SAMPLES}' \
                    --temperature '${EVAL_TEMPERATURE}' \
                    --top-p '${EVAL_TOP_P}' \
                    --top-k '${EVAL_TOP_K}' \
                    --max-tokens '${EVAL_MAX_TOKENS}' \
                    --gpu-memory-utilization '${EVAL_GPU_MEMORY_UTILIZATION}' \
                    --seed 42
                PYTHONPATH='$REPO_CONTAINER:\${PYTHONPATH:-}' python3 -u recipe/on_policy_wdl_sft/code_task/convert_official_outputs.py \
                    --raw-outputs \"\$raw\" \
                    --validation-parquet \"\$validation\" \
                    --output \"\$converted\" \
                    --benchmark '$benchmark' \
                    --report '$case_dir/conversion_report.json' \
                    --allow-extraction-failures
                if [ '$benchmark' = livecodebench ]; then
                    PYTHONPATH='$REPO_CONTAINER:/data-1/code_eval_envs/official_site:/data-1/code_eval_envs/LiveCodeBench:\${PYTHONPATH:-}' python3 -u recipe/on_policy_wdl_sft/code_task/eval_code_official.py \
                        --benchmark livecodebench \
                        --custom-output \"\$converted\" \
                        --output-dir '$case_dir/official' \
                        --summary '$case_dir/official_summary.json' \
                        --parallel '${CODE_OFFICIAL_EVAL_PARALLEL}' \
                        --lcb-python /opt/venv/bin/python \
                        --lcb-release-version '${LCB_RELEASE_VERSION}' \
                        --overwrite
                else
                    PYTHONPATH='$REPO_CONTAINER:/data-1/code_eval_envs/official_site:/data-1/code_eval_envs/LiveCodeBench:\${PYTHONPATH:-}' python3 -u recipe/on_policy_wdl_sft/code_task/eval_code_official.py \
                        --benchmark '$benchmark' \
                        --samples \"\$converted\" \
                        --output-dir '$case_dir/official' \
                        --summary '$case_dir/official_summary.json' \
                        --parallel '${CODE_OFFICIAL_EVAL_PARALLEL}' \
                        --overwrite
                fi" \
            2>&1 | tee -a "$LOG_FILE"
        record_status "$idx" "$label" "$benchmark" "eval_complete" "$summary"
    done
}

write_curve_csv() {
    local csv="${OUTPUT_ROOT}/coldstart_sft_fraction_curve.csv"
    [ "${DRY_RUN:-0}" = "1" ] && { log "DRY_RUN would write curve csv: ${csv}"; return; }
    python3 - "$OUTPUT_ROOT" "$csv" <<'PY'
import csv, json, sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    label = label_dir.name
    for bench_dir in sorted(p for p in label_dir.iterdir() if p.is_dir()):
        conv = bench_dir / "conversion_report.json"
        gen = bench_dir / "generation_summary.json"
        official = bench_dir / "official_summary.json"
        row = {"fraction_label": label, "benchmark": bench_dir.name}
        if conv.exists():
            data = json.loads(conv.read_text())
            counts = data.get("extraction_status_counts", {})
            total = sum(counts.values()) or data.get("total", 0) or 0
            row["extraction_fail"] = counts.get("extraction_fail", 0)
            row["extraction_fail_rate"] = (counts.get("extraction_fail", 0) / total) if total else ""
            row["conversion_ok"] = data.get("ok", "")
        if gen.exists():
            data = json.loads(gen.read_text())
            row["n_samples"] = data.get("n", data.get("n_samples", ""))
            row["num_prompts"] = data.get("num_prompts", "")
        if official.exists():
            data = json.loads(official.read_text())
            for key in ("pass@1", "mean@1", "plus_pass@1", "base_pass@1", "official_aligned"):
                if key in data:
                    row[key] = data[key]
            metrics = data.get("metrics")
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if "pass@1" in key or key in {"pass@1", "mean@1"}:
                        row[key] = value
        rows.append(row)
fields = sorted({k for r in rows for k in r})
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(out)
PY
    log "curve csv updated: ${csv}"
}

log "Qwen3-1.7B coldstart SFT fraction queue start DRY_RUN=${DRY_RUN:-0} range=${START_INDEX}-${END_INDEX} CKPT_ROOT=${CKPT_ROOT} MERGED_ROOT=${MERGED_ROOT} OUTPUT_ROOT=${OUTPUT_ROOT}"
notify "Qwen3-1.7B coldstart SFT fraction queue started" "range=${START_INDEX}-${END_INDEX}; output=${OUTPUT_ROOT}"

for idx in "${!RUN_LABELS[@]}"; do
    [ "$idx" -lt "$START_INDEX" ] && continue
    [ "$idx" -gt "$END_INDEX" ] && continue
    CURRENT_IDX="$idx"
    CURRENT_LABEL="${RUN_LABELS[$idx]}"
    CURRENT_PREFIX="${RUN_PREFIXES[$idx]}"
    disk_gate
    train_file=$(prepare_dataset "$idx")
    launch_train "$idx" "$train_file"
    wait_train "$idx"
    merged=$(merge_model "$idx")
    eval_model "$idx" "$merged"
done

write_curve_csv
log "Qwen3-1.7B coldstart SFT fraction queue complete"
notify "Qwen3-1.7B coldstart SFT fraction queue complete" "results=${OUTPUT_ROOT}; csv=${OUTPUT_ROOT}/coldstart_sft_fraction_curve.csv"
