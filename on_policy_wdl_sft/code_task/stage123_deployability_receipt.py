#!/usr/bin/env python3
"""Verify Stage123 deployability receipts at formal admission time."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys


RFC3339_WHOLE_SECONDS_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def parse_issued_at(value: object) -> datetime:
    if not isinstance(value, str) or not RFC3339_WHOLE_SECONDS_Z.match(value):
        raise ValueError("issued_at must be UTC RFC3339 whole seconds with Z")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def receipt_value(receipt: dict, *names: str):
    for name in names:
        if name in receipt:
            return receipt[name]
    return None


def bound_value(receipt: dict, name: str):
    hashes = receipt.get("hashes")
    if isinstance(hashes, dict) and name in hashes:
        return hashes[name]
    return receipt_value(receipt, name)


def verify(args: argparse.Namespace, *, now: datetime | None = None) -> dict:
    receipt_bytes = args.receipt.read_bytes()
    receipt = json.loads(receipt_bytes)
    manifest = json.loads(args.normalized_manifest.read_text())
    failures: list[str] = []

    if receipt_bytes != canonical_json_bytes(receipt):
        failures.append("receipt is not canonical JSON")

    decision = receipt_value(receipt, "decision", "status")
    if receipt.get("receipt_type") == "code_task_operational_calibration_stage12_producer":
        failures.append("limited_receipt_scope_mismatch")
    if decision != "deployable":
        failures.append("receipt is not deployable")

    expected = {
        "preflight_receipt_sha256": digest(args.preflight_receipt),
        "queue_identity": args.queue_identity,
    }
    aliases = {
        "queue_identity": ("queue_identity", "formal_queue_identity"),
    }
    for key, expected_value in expected.items():
        actual = receipt_value(receipt, *aliases.get(key, (key,))) if key == "queue_identity" else bound_value(receipt, key)
        if actual != expected_value:
            failures.append(f"{key} mismatch")
    rendered_manifest = bound_value(receipt, "rendered_manifest_sha256") or receipt_value(receipt, "manifest_sha256")
    if rendered_manifest != manifest["manifest_sha256"]:
        failures.append("rendered_manifest_sha256 mismatch")
    profile = receipt.get("profile", {})
    observed_profile = profile.get("sha256") if isinstance(profile, dict) else None
    observed_profile = observed_profile or receipt_value(receipt, "profile_sha256")
    if observed_profile != args.profile_hash:
        failures.append("profile_sha256 mismatch")

    file_checks = {
        "report_sha256": args.report,
        "policy_sha256": args.policy,
        "history_index_sha256": args.history_index,
        "prediction_contract_sha256": args.prediction_contract,
    }
    alias_checks = {
        "report_sha256": ("report_sha256", "calibration_report_sha256"),
        "policy_sha256": ("policy_sha256", "calibration_policy_sha256"),
    }
    if args.semantic_contract is not None:
        file_checks["semantic_contract_sha256"] = args.semantic_contract
    for key, path in file_checks.items():
        actual = bound_value(receipt, key)
        if actual is None:
            actual = receipt_value(receipt, *alias_checks.get(key, (key,)))
        if actual != digest(path):
            failures.append(f"{key} mismatch")

    try:
        age = ((now or datetime.now(timezone.utc)) - parse_issued_at(receipt.get("issued_at"))).total_seconds()
    except ValueError as exc:
        failures.append(str(exc))
        age = 0.0
    else:
        if age < -args.future_skew_seconds:
            failures.append("receipt issued_at exceeds future skew")
        if age > args.max_age_seconds:
            failures.append("receipt stale")

    return {"ok": not failures, "failures": failures, "age_seconds": round(age, 3)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--normalized-manifest", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--history-index", type=Path, required=True)
    parser.add_argument("--prediction-contract", type=Path, required=True)
    parser.add_argument("--semantic-contract", type=Path)
    parser.add_argument("--queue-identity", required=True)
    parser.add_argument("--profile-hash", required=True)
    parser.add_argument("--max-age-seconds", type=int, default=86400)
    parser.add_argument("--future-skew-seconds", type=int, default=300)
    args = parser.parse_args()
    try:
        result = verify(args)
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"deployability receipt error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
