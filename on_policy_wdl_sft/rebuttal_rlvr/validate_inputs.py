#!/usr/bin/env python3
"""Fail-closed input and resolved-config checks for rebuttal RLVR."""

from __future__ import annotations

import argparse
import ast
import configparser
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

import jsonschema
import yaml


HERE = Path(__file__).resolve().parent
TRAIN_RECEIPT_SCHEMA = HERE / "train_receipt.schema.json"
MATH7_RECEIPT_SCHEMA = HERE / "math7_receipt.schema.json"
GRADER_RECEIPT_SCHEMA = HERE / "grader_receipt.schema.json"
H20_PROFILE_SCHEMA = HERE / "h20_profile.schema.json"
H20_CALIBRATION_SCHEMA = HERE / "h20_calibration_receipt.schema.json"
H20_TERMINAL_SCHEMA = HERE / "h20_calibration_terminal.schema.json"
H20_WORKER_EVIDENCE_SCHEMA = HERE / "h20_worker_evidence.schema.json"
PATH_OVERRIDE_RECEIPT_SCHEMA = HERE / "path_override_receipt.schema.json"
MATH7_KEYS = ("aime_2025", "math_500", "amc23", "aqua", "gsm8k", "mawps", "svamp")
RUNTIME_VERSION_FIELDS = (
    "nvidia_driver",
    "cuda_driver",
    "cuda_runtime",
    "pytorch",
    "vllm",
    "flashinfer",
)

MATCHED_FIELDS = (
    "base_model_revision",
    "architecture",
    "tokenizer_hash",
    "math_source_receipt",
    "prompt_template_hash",
    "target_supervised_tokens",
    "optimizer_updates",
    "optimizer",
    "learning_rate_schedule",
    "initialization_seed",
    "checkpoint_selection_rule",
)

CLASSIFIER_FOR_ARM = {"sft": "ordinary_sft", "wdl": "offline_wdl_sft"}
FORBIDDEN_OFFLINE_WDL_MARKERS = (
    "on_policy",
    "on-policy",
    "wdl_sft_is",
    "minirl",
    "group_adv",
    "group-adv",
    "grpo",
    "ppo",
    "dpo",
)


class ValidationError(ValueError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_projection_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def validate_runtime_versions_projection(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(RUNTIME_VERSION_FIELDS):
        raise ValidationError(f"{label} must contain exactly {RUNTIME_VERSION_FIELDS}")
    numeric_fields = {"nvidia_driver", "cuda_driver", "cuda_runtime"}
    for field in RUNTIME_VERSION_FIELDS:
        version = value[field]
        pattern = r"[0-9]+(?:\.[0-9]+)+" if field in numeric_fields else r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,127}"
        if not isinstance(version, str) or re.fullmatch(pattern, version) is None:
            raise ValidationError(f"{label} has an invalid {field} version")
    return value


def probe_runtime_versions(expected_gpu_count: int) -> dict[str, str]:
    try:
        driver_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        summary_result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValidationError("cannot execute the live NVIDIA runtime probe") from exc
    if driver_result.returncode != 0 or driver_result.stderr.strip():
        raise ValidationError("live NVIDIA driver probe failed or wrote stderr")
    if summary_result.returncode != 0 or summary_result.stderr.strip():
        raise ValidationError("live NVIDIA CUDA-driver probe failed or wrote stderr")
    driver_versions = [line.strip() for line in driver_result.stdout.splitlines() if line.strip()]
    if len(driver_versions) != expected_gpu_count or len(set(driver_versions)) != 1:
        raise ValidationError(
            f"live NVIDIA driver probe must report one version on exactly {expected_gpu_count} GPUs"
        )
    cuda_driver_match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)+)", summary_result.stdout)
    if cuda_driver_match is None:
        raise ValidationError("cannot parse the live CUDA driver API version from nvidia-smi")

    try:
        torch = importlib.import_module("torch")
        vllm = importlib.import_module("vllm")
        flashinfer = importlib.import_module("flashinfer")
    except Exception as exc:
        raise ValidationError("cannot import torch, vllm, and flashinfer for the live runtime probe") from exc
    try:
        cuda_runtime = torch.version.cuda
        if not torch.cuda.is_available() or torch.cuda.device_count() != expected_gpu_count:
            raise ValidationError(
                f"PyTorch must see exactly {expected_gpu_count} CUDA devices during the live runtime probe"
            )
        versions = {
            "nvidia_driver": driver_versions[0],
            "cuda_driver": cuda_driver_match.group(1),
            "cuda_runtime": str(cuda_runtime) if cuda_runtime is not None else "",
            "pytorch": str(torch.__version__),
            "vllm": str(vllm.__version__),
            "flashinfer": str(flashinfer.__version__),
        }
    except (AttributeError, TypeError) as exc:
        raise ValidationError("one or more imported runtime packages do not expose a version") from exc
    return validate_runtime_versions_projection(versions, "live runtime_versions")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"JSON root must be an object: {path}")
    return value


def validate_self_hash(value: dict[str, Any], field: str, label: str) -> None:
    expected = value.get(field)
    actual = hashlib.sha256(canonical_json({k: v for k, v in value.items() if k != field})).hexdigest()
    if expected != actual:
        raise ValidationError(f"{label} {field} mismatch")


def validate_json_schema(value: dict[str, Any], schema_path: Path, label: str) -> None:
    try:
        jsonschema.validate(value, json.loads(schema_path.read_text()))
    except (OSError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
        detail = exc.message if isinstance(exc, jsonschema.ValidationError) else str(exc)
        raise ValidationError(f"{label} schema validation failed: {detail}") from exc


def parquet_row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValidationError("pyarrow is required to verify parquet row counts") from exc
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception as exc:  # pyarrow raises several format- and IO-specific exception types
        raise ValidationError(f"cannot read parquet row count: {path}") from exc


def parquet_prompt_projection(path: Path) -> tuple[list[Any], list[Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValidationError("pyarrow is required to verify parquet prompt projections") from exc
    try:
        table = pq.read_table(path, columns=["prompt", "extra_info"])
        prompts = table.column("prompt").to_pylist()
        extra_info = table.column("extra_info").to_pylist()
    except Exception as exc:
        raise ValidationError(f"cannot read prompt/extra_info columns: {path}") from exc
    prompt_ids = []
    for index, value in enumerate(extra_info):
        if not isinstance(value, dict) or "index" not in value:
            raise ValidationError(f"parquet row {index} lacks extra_info.index: {path}")
        prompt_ids.append(value["index"])
    return prompts, prompt_ids


def filtered_train_projection(path: Path, tokenizer_model: Path, max_prompt_length: int = 500) -> tuple[int, str, str]:
    prompts, _ = parquet_prompt_projection(path)
    try:
        from transformers import AutoTokenizer
        from verl.utils.chat_template import normalize_chat_template_token_ids

        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_model),
            local_files_only=True,
            trust_remote_code=True,
        )
    except Exception as exc:
        raise ValidationError(f"cannot load the bound init tokenizer: {tokenizer_model}") from exc
    eligible = []
    for index, prompt in enumerate(prompts):
        try:
            token_ids = normalize_chat_template_token_ids(
                tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
            )
        except Exception as exc:
            raise ValidationError(f"cannot apply the frozen chat template to train row {index}") from exc
        if len(token_ids) <= max_prompt_length:
            eligible.append(index)
    return (
        len(eligible),
        canonical_projection_hash(eligible),
        canonical_projection_hash(prompts),
    )


def validate_bound_parquet(entry: dict[str, Any], expected_path: Path, label: str) -> None:
    receipt_path = Path(str(entry.get("path", "")))
    if receipt_path.resolve() != expected_path.resolve():
        raise ValidationError(f"{label} receipt path does not match the consumed parquet")
    if not expected_path.is_file():
        raise ValidationError(f"{label} parquet is missing: {expected_path}")
    if sha256_file(expected_path) != entry.get("sha256"):
        raise ValidationError(f"{label} live parquet hash mismatch")
    if parquet_row_count(expected_path) != entry.get("row_count"):
        raise ValidationError(f"{label} live parquet row count mismatch")


def validate_train_receipt(receipt: dict[str, Any], expected_path: Path, tokenizer_model: Path) -> None:
    validate_json_schema(receipt, TRAIN_RECEIPT_SCHEMA, "train receipt")
    validate_self_hash(receipt, "receipt_sha256", "train receipt")
    validate_bound_parquet(
        {**receipt, "row_count": receipt["source_row_count"]},
        expected_path,
        "train",
    )
    filtered_count, filtered_ids_hash, prompt_hash = filtered_train_projection(expected_path, tokenizer_model)
    if filtered_count != receipt["filtered_row_count"]:
        raise ValidationError("train filtered row count mismatch")
    if filtered_ids_hash != receipt["filtered_row_ids_sha256"]:
        raise ValidationError("train filtered row-ID hash mismatch")
    if prompt_hash != receipt["prompt_template_sha256"]:
        raise ValidationError("train prompt/template projection hash mismatch")


def validate_math7_receipt(
    receipt: dict[str, Any],
    expected_paths: dict[str, Path],
    grader_receipt_path: Path,
) -> None:
    validate_json_schema(receipt, MATH7_RECEIPT_SCHEMA, "Math-7 receipt")
    validate_self_hash(receipt, "receipt_sha256", "Math-7 receipt")
    if set(expected_paths) != set(MATH7_KEYS):
        raise ValidationError("internal Math-7 expected-path map is incomplete")
    for key in MATH7_KEYS:
        validate_bound_parquet(receipt["datasets"][key], expected_paths[key], f"Math-7 {key}")
        prompts, prompt_ids = parquet_prompt_projection(expected_paths[key])
        if canonical_projection_hash(prompt_ids) != receipt["datasets"][key]["ordered_prompt_ids_sha256"]:
            raise ValidationError(f"Math-7 {key} ordered prompt-ID hash mismatch")
        if canonical_projection_hash(prompts) != receipt["datasets"][key]["prompt_template_sha256"]:
            raise ValidationError(f"Math-7 {key} prompt/template projection hash mismatch")
    if receipt["grader_receipt_sha256"] != sha256_file(grader_receipt_path):
        raise ValidationError("Math-7 receipt is not bound to the supplied grader receipt")


def validate_grader_receipt(
    receipt: dict[str, Any],
    expected_path: Path,
    repo_root: Path,
    image_digest: str,
) -> None:
    validate_json_schema(receipt, GRADER_RECEIPT_SCHEMA, "grader receipt")
    validate_self_hash(receipt, "receipt_sha256", "grader receipt")
    if Path(receipt["path"]).resolve() != expected_path.resolve():
        raise ValidationError("grader receipt path does not match the executed grader")
    if not expected_path.is_file() or sha256_file(expected_path) != receipt["sha256"]:
        raise ValidationError("grader live file hash mismatch")
    if receipt["image_digest"] != image_digest:
        raise ValidationError("grader receipt image digest does not match the manifest")
    try:
        recipe_commit = subprocess.run(
            ["git", "-C", str(repo_root / "recipe"), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.SubprocessError as exc:
        raise ValidationError("cannot resolve the live recipe submodule commit") from exc
    if receipt["recipe_commit"] != recipe_commit:
        raise ValidationError("grader receipt recipe commit mismatch")
    try:
        tree = ast.parse(expected_path.read_text(), filename=str(expected_path))
    except (OSError, SyntaxError) as exc:
        raise ValidationError("grader source cannot be parsed") from exc
    functions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if receipt["function_name"] not in functions:
        raise ValidationError("grader function is absent from the bound source")


def new_ini_parser() -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None, strict=True, empty_lines_in_values=False)
    parser.optionxform = str
    return parser


def h20_resources_from_ini(path: Path) -> tuple[dict[str, Any], str]:
    parser = new_ini_parser()
    try:
        with path.open(encoding="utf-8") as handle:
            parser.read_file(handle)
    except (OSError, configparser.Error) as exc:
        raise ValidationError(f"cannot strictly parse staged H20 run.hope: {path}") from exc
    required_keys = {
        "resource": {"usergroup", "queue"},
        "roles": {"workers", "worker.memory", "worker.vcore", "worker.gcoresh20-141g", "worker.script"},
        "failover": {"afo.app.support.engine.failover"},
    }
    for section, keys in required_keys.items():
        if section not in parser or set(parser[section]) != keys:
            raise ValidationError(f"staged H20 run.hope has an unexpected [{section}] resource surface")
    for key in (
        "afo.app.env.YARN_CONTAINER_RUNTIME_DOCKER_SHM_SIZE_BYTES",
        "afo.role.worker.task.attempt.max.retry",
    ):
        if "others" not in parser or key not in parser["others"]:
            raise ValidationError(f"staged H20 run.hope is missing [others] {key}")
    if "docker" not in parser or set(parser["docker"]) != {"afo.docker.image.name"}:
        raise ValidationError("staged H20 run.hope has an unexpected [docker] surface")

    def integer(section: str, key: str) -> int:
        raw = parser[section][key]
        if not re.fullmatch(r"0|[1-9][0-9]*", raw):
            raise ValidationError(f"staged H20 run.hope has a non-canonical integer: [{section}] {key}")
        return int(raw)

    failover = parser["failover"]["afo.app.support.engine.failover"]
    if failover not in {"true", "false"}:
        raise ValidationError("staged H20 failover value is not canonical boolean text")
    return (
        {
            "usergroup": parser["resource"]["usergroup"],
            "queue": parser["resource"]["queue"],
            "workers": integer("roles", "workers"),
            "worker_memory_mb": integer("roles", "worker.memory"),
            "worker_vcore": integer("roles", "worker.vcore"),
            "gpu_resource_key": "worker.gcoresh20-141g",
            "gpu_count": integer("roles", "worker.gcoresh20-141g"),
            "worker_script": parser["roles"]["worker.script"],
            "shm_size_bytes": integer("others", "afo.app.env.YARN_CONTAINER_RUNTIME_DOCKER_SHM_SIZE_BYTES"),
            "max_retry": integer("others", "afo.role.worker.task.attempt.max.retry"),
            "failover": failover == "true",
        },
        parser["docker"]["afo.docker.image.name"],
    )


def validate_bound_evidence(binding: dict[str, Any], root: Path, label: str) -> Path:
    path = Path(str(binding.get("path", "")))
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValidationError(f"{label} is outside ROOT") from exc
    if path.resolve() == root.resolve() or not path.is_file():
        raise ValidationError(f"{label} is missing")
    if sha256_file(path) != binding.get("sha256"):
        raise ValidationError(f"{label} hash mismatch")
    return path


def load_reviewer_key(path: Path, key_id: str) -> dict[str, str]:
    value = load_json(path)
    if set(value) != {"schema_version", "reviewers"} or value["schema_version"] != 1:
        raise ValidationError("independent reviewer-key allowlist has an invalid root schema")
    reviewers = value["reviewers"]
    if not isinstance(reviewers, list):
        raise ValidationError("independent reviewer-key allowlist reviewers must be a list")
    matches = []
    seen: set[str] = set()
    for reviewer in reviewers:
        required = {"key_id", "principal", "owner_identity", "public_key"}
        if not isinstance(reviewer, dict) or set(reviewer) != required:
            raise ValidationError("independent reviewer-key allowlist entry has an invalid schema")
        if reviewer["key_id"] in seen:
            raise ValidationError("independent reviewer-key allowlist contains a duplicate key_id")
        seen.add(reviewer["key_id"])
        if reviewer["key_id"] == key_id:
            matches.append(reviewer)
    if len(matches) != 1:
        raise ValidationError(f"H20 calibration reviewer key is not independently allowlisted: {key_id}")
    reviewer = matches[0]
    if not re.fullmatch(r"[A-Za-z0-9_.@+-]+", reviewer["principal"]):
        raise ValidationError("H20 calibration reviewer principal is unsafe")
    if not re.fullmatch(r"ssh-ed25519 [A-Za-z0-9+/=]+(?: [^\r\n]+)?", reviewer["public_key"]):
        raise ValidationError("H20 calibration reviewer key must be one SSH Ed25519 line")
    return reviewer


def verify_reviewer_signature(payload: Path, signature: Path, reviewer: dict[str, str]) -> None:
    with tempfile.TemporaryDirectory(prefix="h20-g4-verify-") as temporary:
        allowed = Path(temporary) / "allowed_signers"
        allowed.write_text(f"{reviewer['principal']} {reviewer['public_key']}\n")
        try:
            result = subprocess.run(
                [
                    "ssh-keygen",
                    "-Y",
                    "verify",
                    "-f",
                    str(allowed),
                    "-I",
                    reviewer["principal"],
                    "-n",
                    "rebuttal-rlvr-g4",
                    "-s",
                    str(signature),
                ],
                input=payload.read_bytes(),
                capture_output=True,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValidationError("cannot execute H20 calibration signature verification") from exc
    if result.returncode != 0:
        raise ValidationError("H20 calibration detached reviewer signature is invalid")


def validate_h20_profile(
    profile: dict[str, Any],
    profile_path: Path,
    calibration_receipt_path: Path | None,
    reviewer_allowlist_path: Path,
    rendered_hope_path: Path,
    root: Path,
    image_digest: str,
    run_mode: str,
    env_output: Path,
) -> None:
    validate_json_schema(profile, H20_PROFILE_SCHEMA, "H20 profile")
    validate_self_hash(profile, "receipt_sha256", "H20 profile")
    if profile["image_digest"] != image_digest:
        raise ValidationError("H20 profile image digest does not match the manifest")
    expected_runtime_versions = validate_runtime_versions_projection(
        profile["runtime_versions"], "H20 profile runtime_versions"
    )
    live_runtime_versions = probe_runtime_versions(profile["gpu_count"])
    runtime_mismatches = [
        field
        for field in RUNTIME_VERSION_FIELDS
        if live_runtime_versions[field] != expected_runtime_versions[field]
    ]
    if runtime_mismatches:
        raise ValidationError(f"live runtime_versions differ from the H20 profile: {runtime_mismatches}")
    rendered_resources, rendered_image = h20_resources_from_ini(rendered_hope_path)
    if rendered_resources != profile["platform_resources"]:
        raise ValidationError("actual staged run.hope resources differ from the H20 profile")
    if not rendered_image.endswith("@" + image_digest):
        raise ValidationError("actual staged run.hope image differs from the manifest digest")
    if run_mode == "smoke":
        if profile["profile_status"] != "smoke_candidate":
            raise ValidationError("smoke runs require profile_status=smoke_candidate")
    else:
        if profile["profile_status"] != "formal_frozen":
            raise ValidationError("formal runs require a calibrated formal_frozen H20 profile")
        if calibration_receipt_path is None:
            raise ValidationError("formal runs require a signed H20 calibration admission")
        admission = load_json(calibration_receipt_path)
        validate_json_schema(admission, H20_CALIBRATION_SCHEMA, "H20 calibration admission")
        validate_self_hash(admission, "receipt_sha256", "H20 calibration admission")
        if Path(admission["h20_profile_path"]).resolve() != profile_path.resolve():
            raise ValidationError("H20 calibration admission points at a different profile")
        if admission["h20_profile_sha256"] != sha256_file(profile_path):
            raise ValidationError("H20 calibration admission profile hash mismatch")

        terminals: dict[str, tuple[Path, dict[str, Any]]] = {}
        for arm in ("sft", "wdl"):
            terminal_path = validate_bound_evidence(admission["terminal_receipts"][arm], root, f"H20 {arm} terminal receipt")
            terminal = load_json(terminal_path)
            validate_json_schema(terminal, H20_TERMINAL_SCHEMA, f"H20 {arm} terminal receipt")
            validate_self_hash(terminal, "receipt_sha256", f"H20 {arm} terminal receipt")
            if terminal["arm"] != arm or terminal["calibration_id"] != admission["calibration_id"]:
                raise ValidationError(f"H20 {arm} terminal receipt identity mismatch")
            for field in ("image_digest", "runtime_versions", "platform_resources", "selected", "fixed"):
                if terminal[field] != profile[field]:
                    raise ValidationError(f"H20 {arm} terminal receipt differs on {field}")
            if any(terminal["metrics"][key] != value for key, value in profile["arm_metrics"][arm].items()):
                raise ValidationError(f"H20 {arm} terminal metrics differ from the frozen profile")
            staged = validate_bound_evidence(terminal["staged_run_hope"], root, f"H20 {arm} staged run.hope")
            resources, staged_image = h20_resources_from_ini(staged)
            if resources != profile["platform_resources"] or not staged_image.endswith("@" + image_digest):
                raise ValidationError(f"H20 {arm} staged run.hope differs from the frozen profile")
            validate_bound_evidence(terminal["status_evidence"], root, f"H20 {arm} status evidence")
            worker_path = validate_bound_evidence(
                terminal["worker_evidence"], root, f"H20 {arm} worker evidence"
            )
            worker = load_json(worker_path)
            validate_json_schema(worker, H20_WORKER_EVIDENCE_SCHEMA, f"H20 {arm} worker evidence")
            validate_self_hash(worker, "receipt_sha256", f"H20 {arm} worker evidence")
            expected_worker_identity = {
                "calibration_id": admission["calibration_id"],
                "arm": arm,
                "job_id": terminal["job_id"],
                "image_digest": image_digest,
            }
            worker_identity_mismatches = [
                key for key, wanted in expected_worker_identity.items() if worker.get(key) != wanted
            ]
            if worker_identity_mismatches:
                raise ValidationError(
                    f"H20 {arm} worker evidence identity mismatch: {worker_identity_mismatches}"
                )
            if worker["runtime_versions"] != terminal["runtime_versions"]:
                raise ValidationError(f"H20 {arm} worker runtime differs from the terminal receipt")
            if worker["runtime_versions_sha256"] != canonical_projection_hash(terminal["runtime_versions"]):
                raise ValidationError(f"H20 {arm} worker runtime_versions hash mismatch")
            if worker["metrics"] != terminal["metrics"]:
                raise ValidationError(f"H20 {arm} worker metrics differ from the terminal receipt")
            terminals[arm] = (terminal_path, terminal)
        if terminals["sft"][1]["job_id"] == terminals["wdl"][1]["job_id"]:
            raise ValidationError("H20 calibration arms cannot reuse one platform job")

        payload = validate_bound_evidence(
            {"path": admission["attestation_payload_path"], "sha256": admission["attestation_payload_sha256"]},
            root,
            "H20 calibration attestation payload",
        )
        signature = validate_bound_evidence(
            {"path": admission["attestation_signature_path"], "sha256": admission["attestation_signature_sha256"]},
            root,
            "H20 calibration attestation signature",
        )
        attestation = load_json(payload)
        expected_fields = {
            "schema_version", "gate", "status", "approval_scope", "calibration_id",
            "h20_profile_path", "h20_profile_sha256", "sft_terminal_receipt_path",
            "sft_terminal_receipt_sha256", "wdl_terminal_receipt_path",
            "wdl_terminal_receipt_sha256", "image_digest", "platform_resources_sha256",
            "runtime_versions_sha256", "selected_system_knobs_sha256", "fixed_system_knobs_sha256",
            "selection_policy_version", "calibration_submitter_identity", "reviewer_key_id",
            "review_evidence_path", "review_evidence_sha256",
        }
        if set(attestation) != expected_fields:
            raise ValidationError("H20 calibration attestation has unexpected or missing fields")
        expected_values = {
            "schema_version": 1,
            "gate": "G4",
            "status": "passed",
            "approval_scope": "rebuttal-h20-common-v1",
            "calibration_id": admission["calibration_id"],
            "h20_profile_path": str(profile_path),
            "h20_profile_sha256": sha256_file(profile_path),
            "sft_terminal_receipt_path": str(terminals["sft"][0]),
            "sft_terminal_receipt_sha256": sha256_file(terminals["sft"][0]),
            "wdl_terminal_receipt_path": str(terminals["wdl"][0]),
            "wdl_terminal_receipt_sha256": sha256_file(terminals["wdl"][0]),
            "image_digest": image_digest,
            "platform_resources_sha256": canonical_projection_hash(profile["platform_resources"]),
            "runtime_versions_sha256": canonical_projection_hash(profile["runtime_versions"]),
            "selected_system_knobs_sha256": canonical_projection_hash(profile["selected"]),
            "fixed_system_knobs_sha256": canonical_projection_hash(profile["fixed"]),
            "selection_policy_version": "rebuttal-h20-common-selection-v1",
            "reviewer_key_id": admission["reviewer_key_id"],
        }
        mismatches = [key for key, wanted in expected_values.items() if attestation.get(key) != wanted]
        if mismatches:
            raise ValidationError(f"H20 calibration attestation binding mismatch: {mismatches}")
        review_evidence = validate_bound_evidence(
            {"path": attestation["review_evidence_path"], "sha256": attestation["review_evidence_sha256"]},
            root,
            "H20 calibration review evidence",
        )
        if not review_evidence.read_bytes():
            raise ValidationError("H20 calibration review evidence is empty")
        reviewer = load_reviewer_key(reviewer_allowlist_path, admission["reviewer_key_id"])
        if reviewer["owner_identity"] == attestation["calibration_submitter_identity"]:
            raise ValidationError("H20 calibration reviewer key owner must differ from the calibration submitter")
        verify_reviewer_signature(payload, signature, reviewer)

    selected = profile["selected"]
    fixed = profile["fixed"]
    exports = {
        "ROLLOUT_GPU_MEMORY_UTILIZATION": selected["rollout_gpu_memory_utilization"],
        "GENERATION_MICRO_BATCH_SIZE": selected["generation_micro_batch_size"],
        "LOG_PROB_MICRO_BATCH_SIZE": selected["log_prob_micro_batch_size"],
        "ACTOR_PPO_MAX_TOKEN_LEN": selected["actor_ppo_max_token_len"],
        "ROLLOUT_TP_SIZE": fixed["tensor_parallel_size"],
        "ROLLOUT_AGENT_NUM_WORKERS": fixed["rollout_agent_num_workers"],
        "ROLLOUT_MAX_NUM_SEQS": fixed["rollout_max_num_seqs"],
        "ROLLOUT_ENFORCE_EAGER": str(fixed["rollout_enforce_eager"]).lower(),
        "ROLLOUT_ENABLE_CHUNKED_PREFILL": str(fixed["rollout_enable_chunked_prefill"]).lower(),
        "ROLLOUT_MAX_MODEL_LEN": fixed["rollout_max_model_len"],
        "LOG_PROB_MAX_TOKEN_LEN_PER_GPU": fixed["log_prob_max_token_len_per_gpu"],
        "ACTOR_PARAM_OFFLOAD": str(fixed["actor_param_offload"]).lower(),
        "ACTOR_OPTIMIZER_OFFLOAD": str(fixed["actor_optimizer_offload"]).lower(),
    }
    env_output.write_text("".join(f"export {key}='{value}'\n" for key, value in exports.items()))
    env_output.chmod(0o600)


def validate_path_override_receipt(
    receipt: dict[str, Any],
    root: Path,
    dataset_root: Path,
    model_root: Path,
    state_root: Path,
    repo_subpath: str,
    init_model_path: Path,
    run_mode: str,
    runtime_paths: dict[str, Path],
) -> None:
    validate_json_schema(receipt, PATH_OVERRIDE_RECEIPT_SCHEMA, "path-override receipt")
    validate_self_hash(receipt, "receipt_sha256", "path-override receipt")
    subpath = Path(repo_subpath)
    if (
        not root.is_absolute()
        or any(not path.is_absolute() for path in (dataset_root, model_root, state_root))
        or subpath.is_absolute()
        or any(part in {"", ".", ".."} for part in subpath.parts)
    ):
        raise ValidationError("path-override validator received an unsafe root/repo_subpath")
    controlled_roots = {
        "dataset_root": dataset_root,
        "model_root": model_root,
        "state_root": state_root,
    }
    resolved_root = root.resolve()
    outside_roots = []
    for name, path in controlled_roots.items():
        try:
            relative = path.resolve().relative_to(resolved_root)
        except ValueError:
            outside_roots.append(name)
            continue
        if not relative.parts:
            outside_roots.append(name)
    if outside_roots:
        raise ValidationError(
            "dataset_root, model_root, and state_root must be strict children of root: "
            f"{outside_roots}"
        )
    output_root = state_root / "verl-exp"
    expected = {
        "root": str(root),
        "dataset_root": str(dataset_root),
        "state_root": str(state_root),
        "repo_subpath": repo_subpath,
        "repo_root": str(root / repo_subpath),
        "model_root": str(model_root),
        "init_model_path": str(init_model_path),
        "train_file": str(dataset_root / "data/math/train_rl_format.parquet"),
        "math7_root": str(dataset_root / "data/math7"),
        "output_root": str(output_root),
        "checkpoint_root": str(output_root / "checkpoints/rebuttal_rlvr"),
        "eval_root": str(output_root / "eval/rebuttal_rlvr"),
        "log_root": str(output_root / "logs/rebuttal_rlvr"),
        "wandb_root": str(output_root / "wandb_runs/rebuttal_rlvr"),
        "receipt_root": str(output_root / "receipts/rebuttal_rlvr"),
        "hf_home": str(output_root / "cache/hf"),
        "huggingface_hub_cache": str(output_root / "cache/hf/hub"),
        "hf_datasets_cache": str(output_root / "cache/datasets"),
        "xdg_cache_home": str(output_root / "cache/xdg"),
        "ray_tmpdir": "/tmp/rebuttal_rlvr/ray",
        "tmpdir": "/tmp/rebuttal_rlvr/tmp",
        "vllm_config_root": "/tmp/rebuttal_rlvr/vllm",
        "zmq_ipc_dir": "/tmp/rebuttal_rlvr/zmq",
    }
    mismatches = [key for key, wanted in expected.items() if receipt.get(key) != wanted]
    if mismatches:
        raise ValidationError(f"path-override receipt differs from the controlled multi-root layout: {mismatches}")
    runtime_mismatches = [key for key, wanted in expected.items() if key in runtime_paths and str(runtime_paths[key]) != wanted]
    if runtime_mismatches:
        raise ValidationError(f"live platform paths differ from the controlled multi-root layout: {runtime_mismatches}")
    if run_mode == "formal":
        resolved_model_root = model_root.resolve()
        resolved_model = init_model_path.resolve()
        try:
            relative = resolved_model.relative_to(resolved_model_root)
        except ValueError as exc:
            raise ValidationError("formal init model must be below the receipt model_root") from exc
        if not relative.parts:
            raise ValidationError("formal init model must name a concrete directory below model_root")


def validate_hf_model(path: Path) -> set[str]:
    if not path.is_dir():
        raise ValidationError(f"model directory does not exist: {path}")
    if not (path / "config.json").is_file():
        raise ValidationError(f"model config.json is missing: {path}")
    tokenizer_names = [name for name in ("tokenizer.json", "tokenizer.model", "vocab.json") if (path / name).is_file()]
    if not tokenizer_names:
        raise ValidationError(f"tokenizer payload is missing: {path}")

    candidates = (
        ("model.safetensors", "model.safetensors.index.json"),
        ("pytorch_model.bin", "pytorch_model.bin.index.json"),
    )
    selected = next(
        ((single_name, index_name) for single_name, index_name in candidates if (path / single_name).is_file() or (path / index_name).is_file()),
        None,
    )
    if selected is None:
        raise ValidationError(f"model weights or index are missing: {path}")
    single_name, index_name = selected
    single = path / single_name
    index = path / index_name
    if single.is_file():
        return {"config.json", tokenizer_names[0], single_name}
    try:
        index_data = json.loads(index.read_text())
        shards = sorted(set(index_data["weight_map"].values()))
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValidationError(f"invalid model weight index: {index}") from exc
    if not shards or any(not (path / shard).is_file() for shard in shards):
        raise ValidationError(f"one or more indexed model shards are missing: {path}")
    return {"config.json", tokenizer_names[0], index_name, *shards}


def validate_checkpoint_receipt(receipt: dict[str, Any], receipt_path: Path, arm: str, model: Path) -> None:
    validate_self_hash(receipt, "receipt_sha256", "checkpoint receipt")
    expected_classifier = CLASSIFIER_FOR_ARM[arm]
    if receipt.get("classifier") != expected_classifier:
        raise ValidationError(f"{arm} receipt classifier must be {expected_classifier}")
    if Path(str(receipt.get("model_path", ""))).resolve() != model.resolve():
        raise ValidationError("checkpoint receipt model_path does not match launch model")
    if receipt.get("post_checkpoint_rl") is not False:
        raise ValidationError("checkpoint receipt must prove post_checkpoint_rl=false")

    required = (
        "base_model_revision",
        "architecture",
        "tokenizer_hash",
        "dataset_receipt_sha256",
        "prompt_template_hash",
        "initialization_seed",
        "optimizer",
        "optimizer_updates",
        "target_supervised_tokens",
        "learning_rate_schedule",
        "checkpoint_selection_rule",
        "training_code_commit",
    )
    missing = [field for field in required if receipt.get(field) in (None, "")]
    if missing:
        raise ValidationError(f"checkpoint receipt is missing provenance fields: {missing}")

    required_model_files = validate_hf_model(model)
    files = receipt.get("files")
    if not isinstance(files, dict) or not files:
        raise ValidationError("checkpoint receipt requires a non-empty files hash map")
    missing_files = sorted(required_model_files - set(files))
    if missing_files:
        raise ValidationError(f"checkpoint receipt omits required model files: {missing_files}")
    for relative, expected_hash in files.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValidationError(f"checkpoint receipt has unsafe relative file: {relative}")
        candidate = model / relative_path
        if not candidate.is_file() or sha256_file(candidate) != expected_hash:
            raise ValidationError(f"checkpoint model-file hash mismatch: {relative}")

    if arm == "wdl":
        provenance_text = canonical_json(receipt).decode("utf-8").lower()
        bad = [marker for marker in FORBIDDEN_OFFLINE_WDL_MARKERS if marker in provenance_text]
        if bad:
            raise ValidationError(f"offline WDL receipt contains post-training markers: {bad}")
        if receipt.get("offline_wdl_stage") in (None, "") or receipt.get("single_model_extraction_rule") in (None, ""):
            raise ValidationError("offline WDL receipt lacks paper stage or extraction rule")

    if sha256_file(receipt_path) == "0" * 64:  # defensive impossibility check
        raise ValidationError("invalid checkpoint receipt hash")


def validate_pair_manifest(pair: dict[str, Any], pair_path: Path, arm: str, receipt_path: Path) -> None:
    validate_self_hash(pair, "manifest_sha256", "paired-init manifest")
    if pair.get("schema_version") != 1 or pair.get("status") != "admitted":
        raise ValidationError("paired-init manifest is not admitted")
    if pair.get("admission_scope") not in {"fixed_pair_pilot", "method_matrix"}:
        raise ValidationError("paired-init manifest has invalid admission_scope")
    matched = pair.get("matched_fields")
    if not isinstance(matched, dict):
        raise ValidationError("paired-init matched_fields must be an object")
    unequal = [field for field in MATCHED_FIELDS if matched.get(field) is not True]
    if unequal:
        raise ValidationError(f"paired-init matching failed: {unequal}")

    binding_name = "ordinary_sft_receipt" if arm == "sft" else "offline_wdl_sft_receipt"
    binding = pair.get(binding_name)
    if not isinstance(binding, dict):
        raise ValidationError(f"paired-init manifest lacks {binding_name}")
    if Path(str(binding.get("path", ""))).resolve() != receipt_path.resolve():
        raise ValidationError(f"paired-init {binding_name} path mismatch")
    if binding.get("sha256") != sha256_file(receipt_path):
        raise ValidationError(f"paired-init {binding_name} file hash mismatch")
    if not pair_path.is_file():
        raise ValidationError("paired-init manifest disappeared during validation")


def nested_get(value: dict[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValidationError(f"resolved config is missing {dotted}")
        current = current[part]
    return current


def expected_resolved_values(
    seed: int | None,
    model: str | None,
    total_training_steps: int = 115,
    save_freq: int = 5,
) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "algorithm.adv_estimator": "grpo",
        "algorithm.norm_adv_by_std_in_grpo": True,
        "algorithm.use_kl_in_reward": False,
        "algorithm.kl_ctrl.kl_coef": 0.0,
        "algorithm.rollout_correction.rollout_is": None,
        "algorithm.rollout_correction.rollout_is_threshold": None,
        "algorithm.rollout_correction.rollout_is_batch_normalize": False,
        "algorithm.rollout_correction.rollout_rs": None,
        "algorithm.rollout_correction.rollout_rs_threshold": None,
        "algorithm.rollout_correction.bypass_mode": False,
        "algorithm.rollout_correction.loss_type": "ppo_clip",
        "actor_rollout_ref.actor.policy_loss.loss_mode": "vanilla",
        "actor_rollout_ref.actor.policy_loss.all_correct_sft_fallback": False,
        "actor_rollout_ref.actor.clip_ratio": 0.2,
        "actor_rollout_ref.actor.clip_ratio_low": 0.2,
        "actor_rollout_ref.actor.clip_ratio_high": 0.2,
        "actor_rollout_ref.actor.clip_ratio_c": 3.0,
        "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 8,
        "actor_rollout_ref.actor.optim.optimizer": "AdamW",
        "actor_rollout_ref.actor.optim.optimizer_impl": "torch.optim",
        "actor_rollout_ref.actor.optim.lr": 5e-7,
        "actor_rollout_ref.actor.optim.weight_decay": 0.1,
        "actor_rollout_ref.actor.optim.lr_warmup_steps": 5,
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "constant",
        "actor_rollout_ref.actor.optim.betas": [0.9, 0.999],
        "actor_rollout_ref.actor.optim.zero_indexed_step": True,
        "actor_rollout_ref.actor.optim.override_optimizer_config": {"eps": 1e-8},
        "actor_rollout_ref.actor.grad_clip": 1.0,
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.calculate_entropy": True,
        "actor_rollout_ref.actor.use_kl_loss": True,
        "actor_rollout_ref.actor.kl_loss_coef": 0.001,
        "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
        "actor_rollout_ref.actor.track_joint_submodel_losses": False,
        "actor_rollout_ref.actor.submodel_kl.enabled": False,
        "actor_rollout_ref.actor.shuffle": False,
        "actor_rollout_ref.model.joint_training": False,
        "actor_rollout_ref.rollout.n": 8,
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.top_k": -1,
        "actor_rollout_ref.rollout.do_sample": True,
        "actor_rollout_ref.rollout.response_length": 4096,
        "actor_rollout_ref.rollout.calculate_log_probs": True,
        "actor_rollout_ref.rollout.val_kwargs.temperature": 1.0,
        "actor_rollout_ref.rollout.val_kwargs.top_p": 0.95,
        "actor_rollout_ref.rollout.val_kwargs.top_k": -1,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": True,
        "actor_rollout_ref.rollout.val_kwargs.n": 3,
        "data.train_batch_size": 64,
        "data.max_prompt_length": 500,
        "data.max_response_length": 4096,
        "data.shuffle": False,
        "reward_model.reward_manager": "naive",
        "trainer.val_before_train": True,
        "trainer.test_freq": 5,
        "trainer.total_epochs": 1,
        "trainer.total_training_steps": total_training_steps,
        "trainer.save_freq": save_freq,
        "trainer.max_actor_ckpt_to_keep": 1,
        "trainer.keep_best_ckpt": True,
        "trainer.best_ckpt_metric_key": "val-core/HuggingFaceH4/MATH-500/acc/mean@3",
        "trainer.best_ckpt_metric_mode": "max",
        "trainer.best_ckpt_strip_optimizer": True,
        "trainer.resume_mode": "disable",
    }
    if seed is not None:
        expected.update(
            {
                "data.seed": seed,
                "actor_rollout_ref.actor.data_loader_seed": seed,
                "actor_rollout_ref.actor.fsdp_config.seed": seed,
                "actor_rollout_ref.ref.fsdp_config.seed": seed,
                "actor_rollout_ref.rollout.seed": seed,
            }
        )
    if model is not None:
        expected["actor_rollout_ref.model.path"] = model
    return expected


def validate_resolved_config(
    path: Path,
    seed: int | None,
    model: str | None,
    total_training_steps: int = 115,
    save_freq: int = 5,
) -> None:
    try:
        value = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ValidationError(f"cannot read resolved Hydra config: {path}") from exc
    if not isinstance(value, dict):
        raise ValidationError("resolved Hydra config is not a mapping")

    mismatches = []
    for key, wanted in expected_resolved_values(seed, model, total_training_steps, save_freq).items():
        actual = nested_get(value, key)
        if actual != wanted:
            mismatches.append(f"{key}: expected {wanted!r}, got {actual!r}")
    if mismatches:
        raise ValidationError("resolved config violates frozen GRPO v1:\n" + "\n".join(mismatches))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    launch = sub.add_parser("launch")
    launch.add_argument("--arm", choices=sorted(CLASSIFIER_FOR_ARM), required=True)
    launch.add_argument("--classifier", required=True)
    launch.add_argument("--model", type=Path, required=True)
    launch.add_argument("--pair-manifest", type=Path, required=True)
    launch.add_argument("--checkpoint-receipt", type=Path, required=True)

    resolved = sub.add_parser("resolved-config")
    resolved.add_argument("--config", type=Path, required=True)
    resolved.add_argument("--seed", type=int)
    resolved.add_argument("--model")
    resolved.add_argument("--total-training-steps", type=int, default=115)
    resolved.add_argument("--save-freq", type=int, default=5)

    artifacts = sub.add_parser("platform-artifacts")
    artifacts.add_argument("--train-receipt", type=Path, required=True)
    artifacts.add_argument("--train-file", type=Path, required=True)
    artifacts.add_argument("--math7-receipt", type=Path, required=True)
    for key in MATH7_KEYS:
        artifacts.add_argument(f"--math7-{key.replace('_', '-')}-file", dest=f"math7_{key}", type=Path, required=True)
    artifacts.add_argument("--grader-receipt", type=Path, required=True)
    artifacts.add_argument("--grader-path", type=Path, required=True)
    artifacts.add_argument("--h20-profile", type=Path, required=True)
    artifacts.add_argument("--h20-calibration-receipt", type=Path)
    artifacts.add_argument("--reviewer-allowlist", type=Path, required=True)
    artifacts.add_argument("--rendered-hope", type=Path, required=True)
    artifacts.add_argument("--path-override-receipt", type=Path, required=True)
    artifacts.add_argument("--image-digest", required=True)
    artifacts.add_argument("--repo-root", type=Path, required=True)
    artifacts.add_argument("--root", type=Path, required=True)
    artifacts.add_argument("--dataset-root", type=Path, required=True)
    artifacts.add_argument("--model-root", type=Path, required=True)
    artifacts.add_argument("--state-root", type=Path, required=True)
    artifacts.add_argument("--repo-subpath", required=True)
    artifacts.add_argument("--init-model-path", type=Path, required=True)
    for name in (
        "output-root",
        "checkpoint-root",
        "eval-root",
        "log-root",
        "wandb-root",
        "receipt-root",
        "hf-home",
        "huggingface-hub-cache",
        "hf-datasets-cache",
        "xdg-cache-home",
        "ray-tmpdir",
        "tmpdir",
        "vllm-config-root",
        "zmq-ipc-dir",
    ):
        artifacts.add_argument(f"--{name}", dest=name.replace("-", "_"), type=Path, required=True)
    artifacts.add_argument("--run-mode", choices=("formal", "smoke"), required=True)
    artifacts.add_argument("--h20-env-output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "launch":
            if args.classifier != CLASSIFIER_FOR_ARM[args.arm]:
                raise ValidationError("wrapper classifier does not match arm")
            receipt = load_json(args.checkpoint_receipt)
            pair = load_json(args.pair_manifest)
            validate_checkpoint_receipt(receipt, args.checkpoint_receipt, args.arm, args.model)
            validate_pair_manifest(pair, args.pair_manifest, args.arm, args.checkpoint_receipt)
        elif args.command == "resolved-config":
            validate_resolved_config(
                args.config,
                args.seed,
                args.model,
                args.total_training_steps,
                args.save_freq,
            )
        else:
            train_receipt = load_json(args.train_receipt)
            math7_receipt = load_json(args.math7_receipt)
            grader_receipt = load_json(args.grader_receipt)
            h20_profile = load_json(args.h20_profile)
            path_override_receipt = load_json(args.path_override_receipt)
            validate_train_receipt(train_receipt, args.train_file, args.init_model_path)
            validate_grader_receipt(
                grader_receipt,
                args.grader_path,
                args.repo_root,
                args.image_digest,
            )
            validate_math7_receipt(
                math7_receipt,
                {key: getattr(args, f"math7_{key}") for key in MATH7_KEYS},
                args.grader_receipt,
            )
            validate_h20_profile(
                h20_profile,
                args.h20_profile,
                args.h20_calibration_receipt,
                args.reviewer_allowlist,
                args.rendered_hope,
                args.root,
                args.image_digest,
                args.run_mode,
                args.h20_env_output,
            )
            validate_path_override_receipt(
                path_override_receipt,
                args.root,
                args.dataset_root,
                args.model_root,
                args.state_root,
                args.repo_subpath,
                args.init_model_path,
                args.run_mode,
                {
                    "output_root": args.output_root,
                    "checkpoint_root": args.checkpoint_root,
                    "eval_root": args.eval_root,
                    "log_root": args.log_root,
                    "wandb_root": args.wandb_root,
                    "receipt_root": args.receipt_root,
                    "hf_home": args.hf_home,
                    "huggingface_hub_cache": args.huggingface_hub_cache,
                    "hf_datasets_cache": args.hf_datasets_cache,
                    "xdg_cache_home": args.xdg_cache_home,
                    "ray_tmpdir": args.ray_tmpdir,
                    "tmpdir": args.tmpdir,
                    "vllm_config_root": args.vllm_config_root,
                    "zmq_ipc_dir": args.zmq_ipc_dir,
                },
            )
    except ValidationError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps({"ok": True, "command": args.command}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
