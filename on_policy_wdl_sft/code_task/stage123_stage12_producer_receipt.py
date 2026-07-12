#!/usr/bin/env python3
"""Verify the checker-owned limited receipt for the exact Stage2 producer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
import stage123_preflight_receipt as preflight_tool


RUN_ID = "frac25-stage2"
RUN_PREFIX = "CODE-S2-QWEN3-1P7B-STAGE123-FRAC25_P40_S220_S340-BETA01-LAMBDA08-V1"
FINAL_STEP = 20
TRAIN_SHA256 = "160be1866e6c1dc439dcfbd594b54324f000f1f48db1f6a0fc88cf227c628dab"
OUTPUT_PATH = "/data-2/model_weights/code_task/qwen3_1p7b_stage123/frac25_p40_s220_s340/stage2_final_model2"
PROVENANCE_PATH = "/data-2/model_weights/code_task/qwen3_1p7b_stage123/frac25_p40_s220_s340/frac25-stage3.provenance.json"
RFC3339 = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()


def verify(args: argparse.Namespace, *, now: datetime | None = None) -> dict:
    receipt_bytes = args.receipt.read_bytes()
    receipt = json.loads(receipt_bytes)
    manifest = json.loads(args.normalized_manifest.read_text())
    preflight = json.loads(args.preflight_receipt.read_text())
    failures: list[str] = []
    if args.run_id != RUN_ID:
        failures.append("limited_receipt_scope_mismatch")
    try:
        preflight_result = preflight_tool.verify(argparse.Namespace(
            receipt=args.preflight_receipt, normalized_manifest=args.normalized_manifest,
            report=args.preflight_report, policy=args.preflight_policy, run_id=RUN_ID,
            calibration_phase=None, profile_hash=manifest.get("resource_profile", {}).get("sha256"),
            max_age_seconds=min(args.max_age_seconds, 86400),
        ))
    except (KeyError, TypeError, ValueError) as exc:
        failures.append(f"preflight verification failed: {exc}")
    else:
        if not preflight_result["ok"]:
            failures.append("preflight verification failed: " + "; ".join(preflight_result["failures"]))
    if preflight.get("authorized_calibration_phases") != ["stage1", "stage2"]:
        failures.append("preflight phase scope mismatch")
    expected_preflight_workloads = {
        phase: hashlib.sha256(json.dumps(manifest.get("calibration_workloads", {}).get(phase), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for phase in ("stage1", "stage2", "stage3")
    }
    if preflight.get("workload_descriptor_sha256") != expected_preflight_workloads:
        failures.append("preflight workload identity mismatch")
    if receipt_bytes != canonical(receipt):
        failures.append("receipt is not canonical JSON")
    expected_keys = {
        "schema_version", "receipt_type", "issued_at", "decision", "ttl_seconds",
        "queue_identity", "phase_scope", "authorized_run_ids", "authorized_final_steps",
        "producer", "selected_cohort_sha256_by_phase",
        "workload_descriptor_sha256_by_phase", "hashes", "failures",
    }
    if set(receipt) != expected_keys:
        failures.append("limited receipt schema mismatch")
    expected = {
        "receipt_type": "code_task_operational_calibration_stage12_producer",
        "decision": "stage12_calibrated",
        "phase_scope": ["stage1", "stage2"],
        "authorized_run_ids": [RUN_ID],
        "authorized_final_steps": {RUN_ID: FINAL_STEP},
        "producer": {
            "run_id": RUN_ID,
            "run_prefix": RUN_PREFIX,
            "final_step": FINAL_STEP,
            "train_file_sha256": TRAIN_SHA256,
            "expected_output_path": OUTPUT_PATH,
            "expected_provenance_path": PROVENANCE_PATH,
        },
        "queue_identity": args.queue_identity,
        "schema_version": 1,
        "ttl_seconds": 86400,
        "failures": [],
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            failures.append(f"{key} mismatch")
    contract = json.loads(args.prediction_contract.read_text())
    phase_docs = {item.get("phase"): item for item in contract.get("phases", [])}
    expected_cohorts = {
        phase: hashlib.sha256(canonical(phase_docs.get(phase, {}).get("eligible_run_ids", []))).hexdigest()
        for phase in ("stage1", "stage2")
    }
    if receipt.get("selected_cohort_sha256_by_phase") != expected_cohorts:
        failures.append("selected cohort hashes mismatch")
    workloads = manifest.get("calibration_workloads", {})
    expected_workloads = {
        phase: hashlib.sha256(canonical(workloads.get(phase))).hexdigest()
        for phase in ("stage1", "stage2")
    }
    if receipt.get("workload_descriptor_sha256_by_phase") != expected_workloads:
        failures.append("workload descriptor hashes mismatch")
    hashes = receipt.get("hashes", {})
    expected_hash_keys = {
        "report_sha256", "manifest_sha256", "rendered_manifest_sha256", "policy_sha256",
        "history_index_sha256", "prediction_contract_sha256", "preflight_receipt_sha256",
    }
    if not isinstance(hashes, dict) or set(hashes) != expected_hash_keys:
        failures.append("limited receipt hashes schema mismatch")
    expected_hashes = {
        "report_sha256": digest(args.report),
        "manifest_sha256": digest(args.manifest),
        "rendered_manifest_sha256": manifest.get("manifest_sha256"),
        "policy_sha256": digest(args.policy),
        "history_index_sha256": digest(args.history_index),
        "prediction_contract_sha256": digest(args.prediction_contract),
        "preflight_receipt_sha256": digest(args.preflight_receipt),
    }
    for key, value in expected_hashes.items():
        if hashes.get(key) != value:
            failures.append(f"{key} mismatch")
    issued = receipt.get("issued_at")
    if not isinstance(issued, str) or not RFC3339.match(issued):
        failures.append("issued_at must be UTC RFC3339 whole seconds with Z")
        age = 0.0
    else:
        age = ((now or datetime.now(timezone.utc)) - datetime.fromisoformat(issued.replace("Z", "+00:00"))).total_seconds()
        if age < -args.future_skew_seconds:
            failures.append("receipt issued_at exceeds future skew")
        if age > min(args.max_age_seconds, 86400):
            failures.append("receipt stale")
    return {"ok": not failures, "failures": failures, "age_seconds": round(age, 3)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--normalized-manifest", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--preflight-receipt", type=Path, required=True)
    p.add_argument("--preflight-report", type=Path, required=True)
    p.add_argument("--preflight-policy", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--history-index", type=Path, required=True)
    p.add_argument("--prediction-contract", type=Path, required=True)
    p.add_argument("--queue-identity", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--max-age-seconds", type=int, default=86400)
    p.add_argument("--future-skew-seconds", type=int, default=300)
    args = p.parse_args()
    try:
        result = verify(args)
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"limited producer receipt error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
