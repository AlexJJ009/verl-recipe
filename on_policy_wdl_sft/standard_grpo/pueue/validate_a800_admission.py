#!/usr/bin/env python3
"""Validate the minimal external handoff supplied by successor Batch GON-35."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

EXPECTED_IMAGE = "sha256:d380888dc8a10796c7f841e341bd775c2d6500ede539f4ea16bb7bf0de92665d"
EXPECTED_LAUNCHER_SHA256 = "58ad5632d4d8ad9a9568e0df81fe5fa000a526793e0865036efe047f9b977c55"
BASELINE = Path(__file__).resolve().parents[1] / "scheduler" / "job_130_baseline.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--root-candidate", required=True)
    parser.add_argument("--recipe-candidate", required=True)
    parser.add_argument("--runtime-env-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--print-runtime-env-sha256", action="store_true")
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    p0 = baseline["p0"]
    launcher = Path("/data_storage/yl_test/lgx/home/.local/bin/verl-dev-run")
    if not launcher.is_file() or digest(launcher) != EXPECTED_LAUNCHER_SHA256:
        fail("installed A800 launcher bytes differ from the admitted checksum")
    expected = {
        "schema_version": 1,
        "batch_id": "GON-35",
        "root_candidate_sha": args.root_candidate,
        "recipe_candidate_sha": args.recipe_candidate,
        "scheduler": "pueue",
        "group": "gpu8",
        "group_concurrency": 1,
        "host_launcher": "verl-dev-run --a800-dev-profile",
        "host_launcher_sha256": EXPECTED_LAUNCHER_SHA256,
        "image_digest": EXPECTED_IMAGE,
        "p0_config_match": True,
        "p1_review_complete": True,
        "model_sha256": p0["model"]["weights_sha256"],
        "data_sha256": p0["data"]["sha256"],
        "scorer_sha256": p0["scorer"]["sha256"],
        "full_gpu_submission_allowed": True,
        "runtime_env_sha256": digest(args.runtime_env_file),
        "output_root": str(args.output_root),
        "receipt_root": str(args.receipt_root),
    }
    mismatches = {
        key: {"expected": value, "actual": receipt.get(key)}
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if mismatches:
        fail(f"A800 admission mismatch: {json.dumps(mismatches, sort_keys=True)}")
    if receipt.get("ci_admission_mode") not in {"full_ci_pass", "base_relative_parity"}:
        fail("A800 admission has an unsupported ci_admission_mode")
    for key in (
        "p0_config_evidence_sha256",
        "p1_review_evidence_sha256",
        "ci_admission_evidence_sha256",
        "independent_review_evidence_sha256",
        "source_snapshot_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get(key, ""))):
            fail(f"A800 admission is missing a valid {key}")
    source_snapshot = args.receipt_root / "source-snapshot.json"
    if not source_snapshot.is_file() or digest(source_snapshot) != receipt["source_snapshot_sha256"]:
        fail("A800 source snapshot bytes differ from the admitted checksum")
    if args.print_runtime_env_sha256:
        print(expected["runtime_env_sha256"])
    else:
        print("ok: external GON-35 A800 admission receipt verified")


if __name__ == "__main__":
    main()
