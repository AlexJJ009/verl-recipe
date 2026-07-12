#!/usr/bin/env python3
"""Build and verify outcome-schema-v2 calibration workload descriptors."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat


HASH_ALGORITHM = "sorted_relative_path_content_sha256_v1"
PARAMETER_COUNTER_VERSION = "hf_qwen3_config_parameter_count_v1"
OUTCOME_SCHEMA_VERSION = 2


def _load_eligibility_compute():
    path = Path(__file__).with_name("calibration_validation_eligibility.py")
    spec = importlib.util.spec_from_file_location("calibration_validation_eligibility", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load validation eligibility calculator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hf_blob_root(path: Path) -> Path | None:
    for parent in (path, *path.parents):
        if parent.name.startswith("models--"):
            root = parent / "blobs"
            return root.resolve() if root.is_dir() else None
    return None


def artifact_sha256(path: Path) -> str:
    if path.is_file() and not path.is_symlink():
        return file_sha256(path)
    if not path.is_dir():
        raise ValueError(f"artifact is not a file or directory: {path}")
    blob_root = _hf_blob_root(path)
    digest = hashlib.sha256()
    entries = sorted((item for item in path.rglob("*") if not item.is_dir()), key=lambda item: item.relative_to(path).as_posix())
    if not entries:
        raise ValueError(f"artifact directory is empty: {path}")
    for item in entries:
        rel = item.relative_to(path).as_posix()
        info = item.lstat()
        content_path = item
        if stat.S_ISLNK(info.st_mode):
            target_text = os.readlink(item)
            if os.path.isabs(target_text) or blob_root is None:
                raise ValueError(f"unsafe artifact symlink: {item}")
            immediate = item.parent / target_text
            if immediate.is_symlink():
                raise ValueError(f"artifact symlink chain is forbidden: {item}")
            resolved = immediate.resolve(strict=True)
            try:
                resolved.relative_to(blob_root)
            except ValueError as exc:
                raise ValueError(f"artifact symlink escapes HF blob root: {item}") from exc
            if not resolved.is_file():
                raise ValueError(f"artifact symlink target is not a regular file: {item}")
            content_path = resolved
        elif not stat.S_ISREG(info.st_mode):
            raise ValueError(f"special artifact entry is forbidden: {item}")
        rel_bytes = rel.encode()
        digest.update(len(rel_bytes).to_bytes(8, "big"))
        digest.update(rel_bytes)
        digest.update(bytes.fromhex(file_sha256(content_path)))
    return digest.hexdigest()


def qwen3_parameter_count(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    required = {
        "model_type", "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim", "tie_word_embeddings", "attention_bias",
    }
    missing = sorted(required - config.keys())
    if missing or config.get("model_type") != "qwen3" or config.get("attention_bias") is not False:
        raise ValueError(f"unsupported Qwen3 config; missing={missing}")
    values = {key: config[key] for key in required - {"model_type", "tie_word_embeddings", "attention_bias"}}
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values.values()):
        raise ValueError("Qwen3 shape fields must be positive integers")
    h = config["hidden_size"]
    kv = config["num_key_value_heads"] * config["head_dim"]
    if config["num_attention_heads"] * config["head_dim"] != h:
        raise ValueError("Qwen3 attention heads do not span hidden_size")
    embedding = config["vocab_size"] * h
    attention = 2 * h * h + 2 * h * kv
    mlp = 3 * h * config["intermediate_size"]
    norms = 2 * h
    per_layer = attention + mlp + norms
    final_norm = h
    lm_head = 0 if config["tie_word_embeddings"] else config["vocab_size"] * h
    total = embedding + config["num_hidden_layers"] * per_layer + final_norm + lm_head
    return {
        "version": PARAMETER_COUNTER_VERSION,
        "components": {
            "embedding": embedding,
            "attention_per_layer": attention,
            "mlp_per_layer": mlp,
            "norms_per_layer": norms,
            "layers": config["num_hidden_layers"],
            "final_norm": final_norm,
            "lm_head": lm_head,
        },
        "total": total,
    }


def dataset_descriptor(name: str, path: Path) -> dict:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["extra_info"])
    uids = []
    for row in table.column("extra_info").to_pylist():
        uid = row.get("uid") if isinstance(row, dict) else None
        if not isinstance(uid, str) or not uid:
            raise ValueError(f"{name}: missing extra_info.uid")
        uids.append(uid)
    if len(uids) != len(set(uids)):
        raise ValueError(f"{name}: duplicate extra_info.uid")
    dataset_hash = file_sha256(path)
    mapping = {
        "schema_version": 1,
        "dataset_name": name,
        "dataset_sha256": dataset_hash,
        "strata": {"unstratified": sorted(uids)},
    }
    return {
        "name": name,
        "path": str(path),
        "sha256": dataset_hash,
        "row_count": len(uids),
        "uid_source": "parquet.extra_info.uid",
        "difficulty_resolution": "unavailable",
        "difficulty_mapping_sha256": hashlib.sha256(canonical_json(mapping)).hexdigest(),
        "difficulty_stratum_counts": {"unstratified": len(uids)},
    }


def model_source(role: str, path: Path) -> tuple[dict, dict]:
    count = qwen3_parameter_count(path / "config.json")
    return ({"role": role, "path": str(path), "artifact_sha256": artifact_sha256(path), "hash_algorithm": HASH_ALGORITHM}, count)


def build(args: argparse.Namespace) -> dict:
    validation_sources = [
        ("HumanEval+", args.humaneval_plus),
        ("MBPP+", args.mbpp_plus),
        ("LiveCodeBench", args.livecodebench),
    ]
    datasets = [
        dataset_descriptor(name, path) for name, path in validation_sources
    ]
    eligibility = _load_eligibility_compute()(
        repo_root=args.repo_root,
        tokenizer_path=args.tokenizer,
        datasets=validation_sources,
        max_prompt_length=args.max_prompt_length,
    )
    tokenizer = {
        "path": str(args.tokenizer),
        "config_sha256": file_sha256(args.tokenizer / "tokenizer_config.json"),
        "tokenizer_sha256": file_sha256(args.tokenizer / "tokenizer.json"),
    }
    specs = {
        "stage1": ("base_pretrained", [("rollout", args.stage1_model)]),
        "stage2": ("fixed_model2_joint_rollout", [("model1", args.stage2_model1), ("model2", args.stage2_model2)]),
        "stage3": ("stage2_model2_handoff", [("rollout", args.stage3_model)]),
    }
    result = {}
    for phase, (provenance, source_specs) in specs.items():
        sources, counts = zip(*(model_source(role, path) for role, path in source_specs), strict=True)
        totals = [item["total"] for item in counts]
        total = sum(totals)
        result[phase] = {
            "phase": phase,
            "parameter_counter_version": PARAMETER_COUNTER_VERSION,
            "rollout_model_parameter_counts": totals,
            "rollout_model_parameter_count_sum": total,
            "log2_rollout_model_parameter_count_sum": round(math.log2(total), 6),
            "model_provenance_class": provenance,
            "model_sources": list(sources),
            "datasets": datasets,
            "tokenizer": tokenizer,
            "validation_eligibility": eligibility,
            "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
            "parameter_count_evidence": list(counts),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--stage1-model", type=Path, required=True)
    parser.add_argument("--stage2-model1", type=Path, required=True)
    parser.add_argument("--stage2-model2", type=Path, required=True)
    parser.add_argument("--stage3-model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--humaneval-plus", type=Path, required=True)
    parser.add_argument("--mbpp-plus", type=Path, required=True)
    parser.add_argument("--livecodebench", type=Path, required=True)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
