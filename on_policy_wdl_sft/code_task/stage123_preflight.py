#!/usr/bin/env python3
"""Machine-readable preflight gate for the Qwen3-1.7B Stage123 queue."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


REPO = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
PROFILE = SCRIPT_DIR / "qwen3_1p7b_stage123_resource_profile.sh"
MANIFEST = REPO / "recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml"
MANIFEST_TOOL = REPO / "scripts/experiment_manifest.py"
SCORER_PROBE = SCRIPT_DIR / "check_official_scorer_dependencies.py"
DATA_DIR = Path("/data-1/dataset/code/verl_rl")
MATCHED_CONTROL_RUN_ID = "frac25-stage1-control"
PRIMARY_RUN_IDS = (MATCHED_CONTROL_RUN_ID, "frac25-stage2", "frac25-stage3")


def command(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=False, env=env)


def add(checks: list[dict[str, object]], name: str, ok: bool, detail: object) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def scorer_dependency_check() -> tuple[bool, dict[str, object]]:
    scorer_pythonpath = os.pathsep.join((
        "/workspace/verl",
        "/data-1/code_eval_envs/official_site",
        "/data-1/code_eval_envs/LiveCodeBench",
    ))
    environment = {
        **os.environ,
        "PYTHONPATH": scorer_pythonpath,
        "LCB_INPUT_OUTPUT_INDEX": "/data-2/evaluator_assets/livecodebench_cache/index/release_v5_input_output.sqlite",
    }
    result = command(sys.executable, str(SCORER_PROBE), "--pythonpath", scorer_pythonpath, env=environment)
    payload_text = result.stdout if result.returncode == 0 else result.stderr
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        payload = {"ok": False, "failure_class": "dependency_failure", "error": payload_text.strip() or "scorer probe returned invalid JSON"}
    return result.returncode == 0 and payload.get("ok") is True, payload


def implementation_tree_sha256() -> tuple[str | None, str | None]:
    result = command(
        "python3",
        str(REPO / "scripts/implementation_tree_identity.py"),
        "--repo-root",
        str(REPO),
        "--boundary-manifest",
        str(REPO / "config/experiment_execution/stage123_implementation_boundary_v1.json"),
    )
    match = re.search(r'"implementation_tree_sha256"\s*:\s*"([0-9a-f]{64})"', result.stderr)
    return (match.group(1) if match else None, result.stderr.strip() or result.stdout.strip())


def load_host_facts(path: Path) -> tuple[dict[str, object], str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("artifact_type") != "stage123_host_facts":
        raise ValueError("unsupported host facts schema")
    return value, hashlib.sha256(path.read_bytes()).hexdigest()


def shard_manifest_path(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def load_shard_detail(path: Path) -> tuple[bool, dict[str, object]]:
    manifest_path = shard_manifest_path(path)
    ok = path.is_file() and manifest_path.is_file()
    detail: dict[str, object] = {"path": str(path), "manifest": str(manifest_path)}
    if not ok:
        detail["ok"] = False
        return False, detail
    detail["sha256"] = sha256(path)
    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sha = manifest_data.get("output_sha256")
        actual_sha = detail["sha256"]
        selected_rows = manifest_data.get("selected_row_count")
        sampler = manifest_data.get("sampler", {})
        expected_rows = sampler.get("length")
        detail.update(
            {
                "manifest_sha256": expected_sha,
                "selected_row_count": selected_rows,
                "sampler_offset": sampler.get("offset"),
                "sampler_length": expected_rows,
                "sampler_seed": sampler.get("seed"),
                "sampler_order": sampler.get("order"),
                "source_path": manifest_data.get("source_path"),
                "source_sha256": manifest_data.get("source_sha256"),
                "manifest_data": manifest_data,
            }
        )
        ok = expected_sha == actual_sha and selected_rows == expected_rows
    except (OSError, json.JSONDecodeError) as exc:
        detail["manifest_error"] = str(exc)
        ok = False
    detail["ok"] = ok
    return ok, detail


def run_ids_from_manifest(manifest: dict[str, object]) -> list[str]:
    runs = manifest.get("runs", [])
    if not isinstance(runs, list):
        return []
    return [str(item["id"]) for item in sorted(runs, key=lambda item: item["order"]) if isinstance(item, dict) and "id" in item]


def run_by_id(manifest: dict[str, object], run_id: str) -> dict[str, object]:
    for run in manifest.get("runs", []):
        if isinstance(run, dict) and run.get("id") == run_id:
            return run
    return {}


def matched_control_union_ok(details_by_run: dict[str, dict[str, object]]) -> tuple[bool, dict[str, object]]:
    required = ("frac25-stage2", "frac25-stage3", MATCHED_CONTROL_RUN_ID)
    if any(run_id not in details_by_run for run_id in required):
        return False, {"missing": [run_id for run_id in required if run_id not in details_by_run]}
    stage2 = details_by_run["frac25-stage2"].get("manifest_data", {})
    stage3 = details_by_run["frac25-stage3"].get("manifest_data", {})
    control = details_by_run[MATCHED_CONTROL_RUN_ID].get("manifest_data", {})
    stage2_sampler = stage2.get("sampler", {}) if isinstance(stage2, dict) else {}
    stage3_sampler = stage3.get("sampler", {}) if isinstance(stage3, dict) else {}
    control_sampler = control.get("sampler", {}) if isinstance(control, dict) else {}
    checks = {
        "same_source_path": control.get("source_path") == stage2.get("source_path") == stage3.get("source_path"),
        "same_source_sha256": control.get("source_sha256") == stage2.get("source_sha256") == stage3.get("source_sha256"),
        "same_sampler_seed": control_sampler.get("seed") == stage2_sampler.get("seed") == stage3_sampler.get("seed"),
        "same_sampler_order": control_sampler.get("order") == stage2_sampler.get("order") == stage3_sampler.get("order"),
        "control_starts_at_stage2": control_sampler.get("offset") == stage2_sampler.get("offset"),
        "stage3_follows_stage2": stage3_sampler.get("offset") == stage2_sampler.get("offset") + stage2_sampler.get("length"),
        "control_length_is_union": control_sampler.get("length") == stage2_sampler.get("length") + stage3_sampler.get("length"),
        "control_rows_match_length": control.get("selected_row_count") == control_sampler.get("length"),
    }
    return all(checks.values()), {
        "checks": checks,
        "stage2": {"offset": stage2_sampler.get("offset"), "length": stage2_sampler.get("length")},
        "stage3": {"offset": stage3_sampler.get("offset"), "length": stage3_sampler.get("length")},
        "control": {"offset": control_sampler.get("offset"), "length": control_sampler.get("length")},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-active", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--normalized-manifest")
    parser.add_argument("--calibration-result")
    parser.add_argument("--host-facts", type=Path, required=True)
    parser.add_argument("--implementation-tree-sha256")
    args = parser.parse_args()
    checks: list[dict[str, object]] = []
    started_at = now_utc()
    failures: list[dict[str, object]] = []
    manifest: dict[str, object] = {}
    host_facts: dict[str, object] = {}
    host_facts_sha256 = ""
    try:
        host_facts, host_facts_sha256 = load_host_facts(args.host_facts)
        add(checks, "host_facts", bool(host_facts.get("ok")), {"sha256": host_facts_sha256})
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        failures.append({"code": "host_facts_read", "message": str(exc), "context": {}})
        add(checks, "host_facts", False, {"error": str(exc)})

    try:
        calibration = json.loads(Path(args.calibration_result).read_text(encoding="utf-8")) if args.calibration_result else {}
    except (OSError, json.JSONDecodeError) as exc:
        calibration = {}
        failures.append({"code": "calibration_result_read", "message": str(exc), "context": {}})

    repo_host = os.environ.get("REPO_HOST", "")
    add(checks, "repo_topology", repo_host == "/data-1/code/verl" and REPO.exists(), {"repo_host": repo_host, "repo": str(REPO)})
    compat = Path("/data-1/verl07/verl")
    add(checks, "compat_symlink", compat.resolve() == Path("/data-1/code/verl"), str(compat.resolve()))
    add(checks, "checkpoint_mount", host_facts.get("mounts", {}).get("checkpoint_mount") == "/data-2/checkpoints", host_facts.get("mounts", {}))
    scorer_ok, scorer_detail = scorer_dependency_check()
    add(checks, "scorer_dependencies", scorer_ok, scorer_detail)

    profile = command(
        "bash",
        "-lc",
        f"source {PROFILE!s}; stage123_profile_snapshot",
    )
    profile_text = profile.stdout
    profile_hash = hashlib.sha256(profile_text.encode()).hexdigest()
    add(checks, "resource_profile_command", profile.returncode == 0, {"sha256": profile_hash})

    rendered_manifest = command("python3", str(MANIFEST_TOOL), "render", str(MANIFEST), "--format", "json")
    if args.normalized_manifest:
        normalized_path = Path(args.normalized_manifest)
        manifest_result = rendered_manifest
    else:
        temp = tempfile.NamedTemporaryFile(prefix="stage123-preflight-", suffix=".json", delete=False)
        temp.close()
        normalized_path = Path(temp.name)
        manifest_result = rendered_manifest
        if manifest_result.returncode == 0:
            normalized_path.write_text(manifest_result.stdout)
    try:
        manifest = json.loads(normalized_path.read_text()) if manifest_result.returncode == 0 else {}
        manifest_run_ids = run_ids_from_manifest(manifest)
        rendered = json.loads(rendered_manifest.stdout) if rendered_manifest.returncode == 0 else {}
        normalized_matches_repo = manifest == rendered
        phase_readiness = {
            phase: {
                "materialized": all(source.get("state") == "materialized" for source in manifest["calibration_workloads"][phase]["model_sources"]),
                "model_sources": manifest["calibration_workloads"][phase]["model_sources"],
            }
            for phase in ("stage1", "stage2", "stage3")
        }
        identity_ok = manifest_result.returncode == 0 and normalized_matches_repo and phase_readiness["stage1"]["materialized"] and phase_readiness["stage2"]["materialized"]
        run_identity_ok = manifest_run_ids == list(PRIMARY_RUN_IDS)
        add(checks, "run_identity", run_identity_ok, {"run_ids": manifest_run_ids, "expected": list(PRIMARY_RUN_IDS)})
        add(checks, "model_identity", identity_ok, {"manifest_sha256": manifest.get("manifest_sha256"), "normalized_matches_repo": normalized_matches_repo, "phases": phase_readiness})
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        add(checks, "model_identity", False, {"error": str(exc)})
    finally:
        if not args.normalized_manifest:
            normalized_path.unlink(missing_ok=True)

    shard_details: list[dict[str, object]] = []
    shard_details_by_run: dict[str, dict[str, object]] = {}
    shards_ok = True
    for run_id in PRIMARY_RUN_IDS:
        run = run_by_id(manifest, run_id)
        train_file = run.get("train_file")
        if not isinstance(train_file, str):
            ok, detail = False, {"run_id": run_id, "error": "missing train_file"}
        else:
            ok, detail = load_shard_detail(Path(train_file))
            detail["run_id"] = run_id
            detail["expected_sha256"] = run.get("train_file_sha256")
            ok = ok and detail.get("sha256") == run.get("train_file_sha256")
        detail["ok"] = ok
        shards_ok = shards_ok and ok
        shard_details_by_run[run_id] = detail
        shard_details.append(detail)
    add(checks, "dataset_shards", shards_ok, shard_details)
    union_ok, union_detail = matched_control_union_ok(shard_details_by_run)
    add(checks, "matched_control_union", union_ok, union_detail)
    control_run = run_by_id(manifest, MATCHED_CONTROL_RUN_ID)
    control_source = control_run.get("source", {}) if isinstance(control_run.get("source"), dict) else {}
    control_actor_path = Path(str(control_source.get("actor_path", "")))
    control_source_ok = (
        control_source.get("type") == "stage1_actor_checkpoint"
        and control_source.get("fraction") == "frac25"
        and control_source.get("beta") == 0.1
        and control_source.get("handoff_step") == 40
        and control_actor_path.is_dir()
    )
    add(checks, "matched_control_source", control_source_ok, {"source": control_source, "actor_path_exists": control_actor_path.is_dir()})

    gpu = command(
        "nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"
    )
    gpu_rows = [row.strip() for row in gpu.stdout.splitlines() if row.strip()]
    add(
        checks,
        "gpu_inventory",
        gpu.returncode == 0 and len(gpu_rows) == 8 and all(row.startswith("NVIDIA L40S") for row in gpu_rows),
        gpu_rows,
    )
    mem_kb = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal:")))
    add(checks, "host_ram", mem_kb >= 590_000_000, {"mem_total_kb": mem_kb})

    docker_image = host_facts.get("docker_image", {})
    add(checks, "docker_image", bool(docker_image.get("immutable_id")), docker_image)
    release_paths = (
        Path("/data-1/experiment_registry/experiment_registry.sqlite"),
        Path("/data-1/wandb_runs"),
        REPO / "scripts/training_result_release_gate.py",
        REPO / "scripts/code_task_training_release_hook.sh",
    )
    add(checks, "release_paths", all(path.exists() for path in release_paths), [str(path) for path in release_paths])

    active_names = host_facts.get("tmux", {}).get("stage123_conflicts", [])
    add(checks, "no_conflicting_run", args.allow_active or not active_names, active_names)

    tree_sha256, tree_detail = (
        (args.implementation_tree_sha256, "provided")
        if args.implementation_tree_sha256
        else implementation_tree_sha256()
    )
    add(checks, "implementation_tree", tree_sha256 is not None, {"sha256": tree_sha256, "detail": tree_detail})
    manifest_sha256 = manifest.get("manifest_sha256") if isinstance(manifest, dict) else None
    profile_sha256 = manifest.get("resource_profile", {}).get("sha256") if isinstance(manifest, dict) else None
    add(checks, "resource_profile_binding", profile_hash == profile_sha256, {"actual": profile_hash, "expected": profile_sha256})
    calibration_bindings_ok = (
        calibration.get("result_type") == "calibration_result"
        and calibration.get("decision") == "passed"
        and calibration.get("manifest_sha256") == manifest_sha256
        and calibration.get("resource_profile_sha256") == profile_sha256
        and calibration.get("implementation_tree_sha256") == tree_sha256
        and calibration.get("workload_identity", {}).get("run_ids") == run_ids_from_manifest(manifest)
    )
    add(checks, "calibration_binding", calibration_bindings_ok, {"manifest_sha256": manifest_sha256, "calibration_manifest_sha256": calibration.get("manifest_sha256")})
    report = {
        "schema_version": 1,
        "result_type": "preflight_result",
        "decision": "passed" if all(bool(item["ok"]) for item in checks) else "blocked",
        "gate": "qwen3_1p7b_stage123_preflight",
        "ok": all(bool(item["ok"]) for item in checks),
        "manifest_sha256": manifest_sha256,
        "resource_profile_sha256": profile_sha256,
        "implementation_tree_sha256": tree_sha256,
        "calibration_evidence_commit": calibration.get("evidence_commit"),
        "calibration_authorization_identity": calibration.get("authorization_identity"),
        "run_ids": run_ids_from_manifest(manifest),
        "host_facts_sha256": host_facts_sha256,
        "started_at": started_at,
        "completed_at": now_utc(),
        "docker_image_id": docker_image.get("immutable_id", ""),
        "checks": checks,
        "failures": failures,
    }
    rendered = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
