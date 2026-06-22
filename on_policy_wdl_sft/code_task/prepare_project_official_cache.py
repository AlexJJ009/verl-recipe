#!/usr/bin/env python3
"""Materialize official code benchmark cache paths under the project data root."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


def file_info(path: Path) -> dict[str, Any]:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {"path": str(path), "size": path.stat().st_size, "sha256": h.hexdigest()}


def copy_if_needed(src: Path, dst: Path) -> dict[str, Any]:
    if not src.is_file():
        raise FileNotFoundError(f"source file not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or src.stat().st_size != dst.stat().st_size:
        shutil.copy2(src, dst)
    return {"source": str(src), **file_info(dst)}


def first_existing_source(candidates: list[Path], filename: str) -> Path:
    for root in candidates:
        path = root / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f"source file not found in candidate roots: {filename}: {[str(p) for p in candidates]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("/data-1/dataset/code/official_sources"))
    parser.add_argument("--project-cache-root", type=Path, default=Path("/data-1/.cache"))
    parser.add_argument("--hf-home", type=Path, default=Path("/data-1/.cache/huggingface"))
    parser.add_argument("--bcb-source-dir", type=Path, action="append", default=[])
    args = parser.parse_args()
    bcb_source_dirs = args.bcb_source_dir or [
        args.project_cache_root / "bigcodebench",
        Path("/root/.cache/bigcodebench"),
    ]

    bcb_root = args.project_root / "bigcodebench"
    os.environ.setdefault("PROJECT_CACHE_ROOT", str(args.project_cache_root))
    os.environ.setdefault("HF_HOME", str(args.hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(args.hf_home / "datasets"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(args.hf_home / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(args.hf_home))
    os.environ.setdefault("XDG_CACHE_HOME", str(args.project_cache_root))
    manifest: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "project_root": str(args.project_root),
        "project_cache_root": str(args.project_cache_root),
        "hf_home": str(args.hf_home),
        "hf_datasets_cache": str(args.hf_home / "datasets"),
        "huggingface_hub_cache": str(args.hf_home / "hub"),
        "xdg_cache_home": str(args.project_cache_root),
        "bcb_source_candidates": [str(p) for p in bcb_source_dirs],
        "runtime_policy": "Use project_root/HF_HOME/XDG_CACHE_HOME paths at runtime; /root/.cache may be used only as a one-time migration source.",
        "official_sources": {},
    }

    manifest["official_sources"]["bigcodebench_full_v0.1.4"] = copy_if_needed(
        first_existing_source(bcb_source_dirs, "BigCodeBench-v0.1.4.jsonl"),
        bcb_root / "BigCodeBench-v0.1.4.jsonl",
    )
    manifest["official_sources"]["bigcodebench_hard_v0.1.4"] = copy_if_needed(
        first_existing_source(bcb_source_dirs, "BigCodeBench-Hard-v0.1.4.jsonl"),
        bcb_root / "BigCodeBench-Hard-v0.1.4.jsonl",
    )

    lcb_arrow = (
        args.hf_home
        / "datasets/livecodebench___code_generation_lite/release_latest-version_tag=release_v1/0.0.0/"
        / "4c038560f391c4c05fdf7fd7ae61ae0e6dbd8672f8fe5b95597b78a8dc40a417/code_generation_lite-test.arrow"
    )
    if lcb_arrow.is_file():
        manifest["official_sources"]["livecodebench_release_v1_arrow"] = file_info(lcb_arrow)
    else:
        manifest["official_sources"]["livecodebench_release_v1_arrow"] = {
            "path": str(lcb_arrow),
            "missing": True,
        }

    manifest_path = args.project_root / "official_cache_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
