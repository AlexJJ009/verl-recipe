#!/usr/bin/env bash
# Shared L40S resource profile for every phase of the Qwen3-1.7B Stage123 chain.
set -euo pipefail

export STAGE123_RESOURCE_PROFILE_NAME=${STAGE123_RESOURCE_PROFILE_NAME:-l40s8x46g_ram582g_cpu176_ctx9k_v3}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-9216}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-9216}
export REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-9216}
export ACTOR_PPO_MAX_TOKEN_LEN=${ACTOR_PPO_MAX_TOKEN_LEN:-9216}
export GENERATION_MICRO_BATCH_SIZE=${GENERATION_MICRO_BATCH_SIZE:-32}
export LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-8}
export REF_LOG_PROB_MICRO_BATCH_SIZE=${REF_LOG_PROB_MICRO_BATCH_SIZE:-1}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.40}
export ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-False}
export ROLLOUT_ENABLE_SLEEP_MODE=${ROLLOUT_ENABLE_SLEEP_MODE:-False}
export REF_FSDP_OFFLOAD=${REF_FSDP_OFFLOAD:-True}
export FSDP_OFFLOAD=${FSDP_OFFLOAD:-True}
export FSDP_OPTIMIZER_OFFLOAD=${FSDP_OPTIMIZER_OFFLOAD:-True}
export TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-64}
export ROLLOUT_N=${ROLLOUT_N:-8}
export TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ:-$((TRAIN_PROMPT_BSZ * ROLLOUT_N))}
export TEST_FREQ=${TEST_FREQ:-5}
export SAVE_FREQ=${SAVE_FREQ:-5}
export TEMPERATURE=${TEMPERATURE:-1.0}
export TOP_P=${TOP_P:-1.0}
export ROLLOUT_DO_SAMPLE=${ROLLOUT_DO_SAMPLE:-True}
export VAL_N=${VAL_N:-1}
export VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.2}
export VAL_TOP_P=${VAL_TOP_P:-0.95}
export VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
export CODE_REWARD_NUM_WORKERS=${CODE_REWARD_NUM_WORKERS:-8}
export CODE_REWARD_MAX_CONCURRENCY_PER_WORKER=${CODE_REWARD_MAX_CONCURRENCY_PER_WORKER:-4}
export ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-64}
export CODE_REWARD_TIMEOUT=${CODE_REWARD_TIMEOUT:-30}
export CODE_REWARD_MANAGER_TIMEOUT=${CODE_REWARD_MANAGER_TIMEOUT:-30}
export CODE_REWARD_STDIN_CASE_TIMEOUT=${CODE_REWARD_STDIN_CASE_TIMEOUT:-2}
export CODE_REWARD_EXEC_MAX_AS_MB=${CODE_REWARD_EXEC_MAX_AS_MB:-4096}
export BIGCODEBENCH_MAX_AS_LIMIT=${BIGCODEBENCH_MAX_AS_LIMIT:-131072}
export BIGCODEBENCH_MAX_DATA_LIMIT=${BIGCODEBENCH_MAX_DATA_LIMIT:-131072}
export BIGCODEBENCH_MAX_STACK_LIMIT=${BIGCODEBENCH_MAX_STACK_LIMIT:-10}
export LCB_INPUT_OUTPUT_INDEX=${LCB_INPUT_OUTPUT_INDEX:-/data-2/evaluator_assets/livecodebench_cache/index/release_v5_input_output.sqlite}
export LCB_INPUT_OUTPUT_INDEX_SHA256=${LCB_INPUT_OUTPUT_INDEX_SHA256:-2f049e91c20f55b3967655c2828f4188cef4bc13108fd3a6d0407046375954b4}
export LCB_INPUT_OUTPUT_INDEX_RECEIPT=${LCB_INPUT_OUTPUT_INDEX_RECEIPT:-/data-2/evaluator_assets/livecodebench_cache/index/release_v5_input_output.receipt.json}
export LCB_SUBPROCESS_TIMEOUT=${LCB_SUBPROCESS_TIMEOUT:-25}
export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.90}
export RAY_memory_monitor_refresh_ms=${RAY_memory_monitor_refresh_ms:-1000}
export RAY_object_spilling_directory=${RAY_object_spilling_directory:-/data-2/ray_spill}
export RAY_TMPDIR=${STAGE123_RAY_TMPDIR:-/data-2/ray_tmp}
export TMPDIR=${STAGE123_TMPDIR:-/data-2/tmp}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-1}

stage123_profile_fields() {
    printf '%s\n' \
        STAGE123_RESOURCE_PROFILE_NAME MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH \
        ROLLOUT_MAX_MODEL_LEN ROLLOUT_MAX_NUM_BATCHED_TOKENS \
        LOG_PROB_MAX_TOKEN_LEN_PER_GPU REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU ACTOR_PPO_MAX_TOKEN_LEN \
        GENERATION_MICRO_BATCH_SIZE LOG_PROB_MICRO_BATCH_SIZE REF_LOG_PROB_MICRO_BATCH_SIZE \
        ROLLOUT_GPU_MEMORY_UTILIZATION ROLLOUT_FREE_CACHE_ENGINE \
        ROLLOUT_ENABLE_SLEEP_MODE REF_FSDP_OFFLOAD FSDP_OFFLOAD FSDP_OPTIMIZER_OFFLOAD TRAIN_PROMPT_BSZ ROLLOUT_N \
        TRAIN_PROMPT_MINI_BSZ TEST_FREQ SAVE_FREQ TEMPERATURE TOP_P ROLLOUT_DO_SAMPLE \
        VAL_N VAL_TEMPERATURE VAL_TOP_P VAL_DO_SAMPLE VAL_BEFORE_TRAIN \
        TRAIN_MAX_SAMPLES CODE_REWARD_NUM_WORKERS CODE_REWARD_MAX_CONCURRENCY_PER_WORKER \
        ROLLOUT_AGENT_NUM_WORKERS CODE_REWARD_TIMEOUT \
        CODE_REWARD_MANAGER_TIMEOUT CODE_REWARD_STDIN_CASE_TIMEOUT \
        CODE_REWARD_EXEC_MAX_AS_MB BIGCODEBENCH_MAX_AS_LIMIT \
        BIGCODEBENCH_MAX_DATA_LIMIT BIGCODEBENCH_MAX_STACK_LIMIT \
        LCB_INPUT_OUTPUT_INDEX LCB_INPUT_OUTPUT_INDEX_SHA256 LCB_INPUT_OUTPUT_INDEX_RECEIPT \
        LCB_SUBPROCESS_TIMEOUT \
        RAY_memory_usage_threshold RAY_memory_monitor_refresh_ms \
        RAY_object_spilling_directory RAY_TMPDIR TMPDIR OMP_NUM_THREADS \
        NGPUS_PER_NODE ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE
}

stage123_profile_snapshot() {
    local field
    while read -r field; do printf '%s=%s\n' "$field" "${!field}"; done < <(stage123_profile_fields)
}

stage123_profile_hash() { stage123_profile_snapshot | sha256sum | awk '{print $1}'; }

stage123_assert_expected_profile() {
    local phase=${1:?phase required}
    local actual_file expected_file
    actual_file=$(mktemp)
    expected_file=$(mktemp)
    stage123_profile_snapshot > "$actual_file"
    if [ -n "${STAGE123_EXPECTED_PROFILE_SERIALIZATION:-}" ]; then
        printf '%s\n' "$STAGE123_EXPECTED_PROFILE_SERIALIZATION" > "$expected_file"
        if ! cmp -s "$expected_file" "$actual_file"; then
            echo "ERROR: ${phase} resource profile field mismatch" >&2
            diff -u "$expected_file" "$actual_file" >&2 || true
            rm -f "$actual_file" "$expected_file"
            return 1
        fi
    fi
    local actual_hash
    actual_hash=$(sha256sum "$actual_file" | awk '{print $1}')
    if [ -n "${STAGE123_EXPECTED_PROFILE_HASH:-}" ] && [ "$actual_hash" != "$STAGE123_EXPECTED_PROFILE_HASH" ]; then
        echo "ERROR: ${phase} profile hash mismatch expected=${STAGE123_EXPECTED_PROFILE_HASH} actual=${actual_hash}" >&2
        rm -f "$actual_file" "$expected_file"
        return 1
    fi
    echo "[STAGE123 CANONICAL PROFILE phase=${phase} sha256=${actual_hash}]"
    cat "$actual_file"
    rm -f "$actual_file" "$expected_file"
}

stage123_validate_profile() {
    [ "$MAX_RESPONSE_LENGTH" = 8192 ] || { echo "ERROR: MAX_RESPONSE_LENGTH must equal 8192" >&2; return 1; }
    [ "$ROLLOUT_MAX_MODEL_LEN" -ge $((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) ] || { echo "ERROR: model length is smaller than prompt+response" >&2; return 1; }
    [ "$ROLLOUT_MAX_NUM_BATCHED_TOKENS" -ge "$ROLLOUT_MAX_MODEL_LEN" ] || {
        echo "ERROR: rollout token batching must cover at least one full model context" >&2
        return 1
    }
    [ "$ROLLOUT_MAX_NUM_BATCHED_TOKENS" -ge 16384 ] || {
        echo "ERROR: rollout token batching remains safety-only; require at least 16384" >&2
        return 1
    }
    python3 - "$ROLLOUT_GPU_MEMORY_UTILIZATION" <<'PY' || return 1
import sys

utilization = float(sys.argv[1])
if not 0.4 <= utilization < 1.0:
    raise SystemExit("rollout GPU memory utilization must be throughput-qualified in [0.4, 1.0)")
PY
    [ "$ROLLOUT_FREE_CACHE_ENGINE" = False ] || {
        echo "ERROR: vLLM cache unloading is not admitted on this async stack" >&2
        return 1
    }
    [ "$ROLLOUT_ENABLE_SLEEP_MODE" = False ] || {
        echo "ERROR: vLLM sleep mode is not admitted on this async stack" >&2
        return 1
    }
    [ "$REF_FSDP_OFFLOAD" = True ] || {
        echo "ERROR: Stage123 requires on-demand CPU offload for the KL reference model" >&2
        return 1
    }
    [ "$FSDP_OPTIMIZER_OFFLOAD" = True ] || {
        echo "ERROR: Stage123 requires CPU offload for optimizer state" >&2
        return 1
    }
    [ "$FSDP_OFFLOAD" = True ] || {
        echo "ERROR: Stage123 requires actor parameter offload between rollout and training phases" >&2
        return 1
    }
    [ "$REF_LOG_PROB_MICRO_BATCH_SIZE" = 1 ] || {
        echo "ERROR: Stage123 KL reference log-prob micro-batch must equal calibrated value 1" >&2
        return 1
    }
    [ "$REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU" = 9216 ] || {
        echo "ERROR: Stage123 KL reference dynamic token budget must cover the full 9216-token context" >&2
        return 1
    }
    [ "$LOG_PROB_MAX_TOKEN_LEN_PER_GPU" = "$ROLLOUT_MAX_MODEL_LEN" ] || return 1
    [ "$ACTOR_PPO_MAX_TOKEN_LEN" = "$ROLLOUT_MAX_MODEL_LEN" ] || return 1
    [ "$TRAIN_PROMPT_MINI_BSZ" = $((TRAIN_PROMPT_BSZ * ROLLOUT_N)) ] || return 1
    [ "$TEMPERATURE" = 1.0 ] || { echo "ERROR: Stage123 rollout temperature must equal 1.0" >&2; return 1; }
    [ "$TOP_P" = 1.0 ] || { echo "ERROR: Stage123 rollout top_p must equal 1.0" >&2; return 1; }
    [ "$ROLLOUT_DO_SAMPLE" = True ] || { echo "ERROR: Stage123 rollout do_sample must equal True" >&2; return 1; }
    [ "$VAL_TEMPERATURE" = 0.2 ] || { echo "ERROR: Stage123 validation temperature must equal 0.2" >&2; return 1; }
    [ "$VAL_TOP_P" = 0.95 ] || { echo "ERROR: Stage123 validation top_p must equal 0.95" >&2; return 1; }
    [ "$VAL_DO_SAMPLE" = True ] || { echo "ERROR: Stage123 validation do_sample must equal True" >&2; return 1; }
    [ "$VAL_N" = "${STAGE123_EXPECTED_VAL_N:-1}" ] || {
        echo "ERROR: Stage123 validation n must equal ${STAGE123_EXPECTED_VAL_N:-1}" >&2
        return 1
    }
    [ "$CODE_REWARD_NUM_WORKERS" -eq 8 ] || { echo "ERROR: Stage123 reward workers must equal 8" >&2; return 1; }
    [ "$CODE_REWARD_MAX_CONCURRENCY_PER_WORKER" -eq 4 ] || { echo "ERROR: Stage123 per-worker reward concurrency must equal 4" >&2; return 1; }
    [ "$ROLLOUT_AGENT_NUM_WORKERS" -eq 64 ] || { echo "ERROR: Stage123 full-validation profile requires 64 agent workers" >&2; return 1; }
    [ "$NGPUS_PER_NODE" = 8 ] || { echo "ERROR: NGPUS_PER_NODE must equal 8" >&2; return 1; }
    [ "$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE" = 1 ] || { echo "ERROR: Qwen3-1.7B Stage123 TP must equal 1" >&2; return 1; }
    [ -f "$LCB_INPUT_OUTPUT_INDEX" ] || { echo "ERROR: LiveCodeBench index missing: $LCB_INPUT_OUTPUT_INDEX" >&2; return 1; }
    [ -f "$LCB_INPUT_OUTPUT_INDEX_RECEIPT" ] || { echo "ERROR: LiveCodeBench index receipt missing" >&2; return 1; }
    python3 - "$LCB_INPUT_OUTPUT_INDEX" "$LCB_INPUT_OUTPUT_INDEX_RECEIPT" "$LCB_INPUT_OUTPUT_INDEX_SHA256" <<'PY' || return 1
import json,os,sys
index,receipt_path,expected_sha=sys.argv[1:]
receipt=json.load(open(receipt_path,encoding='utf-8'))
assert receipt.get('schema_version') == 1
assert receipt.get('release_version') == 'release_v5'
assert receipt.get('row_count') == 880
assert receipt.get('sha256') == expected_sha
assert receipt.get('size_bytes') == os.path.getsize(index)
PY
    mkdir -p "$RAY_object_spilling_directory" "$RAY_TMPDIR" "$TMPDIR"
}

stage123_check_machine() {
    [ "$(nproc --all)" -ge 176 ] || { echo "ERROR: expected at least 176 online CPUs" >&2; return 1; }
    [ "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA L40S$')" -eq 8 ] || { echo "ERROR: expected 8 NVIDIA L40S GPUs" >&2; return 1; }
    local mem_kb
    mem_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
    [ "$mem_kb" -ge 590000000 ] || { echo "ERROR: expected approximately 582 GiB host RAM" >&2; return 1; }
}

stage123_print_profile() { stage123_assert_expected_profile "${1:?phase required}"; }

stage123_validate_profile
