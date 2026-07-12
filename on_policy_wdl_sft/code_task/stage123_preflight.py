#!/usr/bin/env python3
"""Machine-readable preflight gate for the Qwen3-1.7B Stage123 queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


REPO = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
PROFILE = SCRIPT_DIR / "qwen3_1p7b_stage123_resource_profile.sh"
MANIFEST = REPO / "recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml"
MANIFEST_TOOL = REPO / "scripts/experiment_manifest.py"
DATA_DIR = Path("/data-1/dataset/code/verl_rl")
SHARDS = (
    "kodcode_stage2_after_s1_seed20260604_qwen3_1p7b_coldstart_frac25_beta01_p40_handoff_s2steps20.parquet",
    "kodcode_stage3_after_s2_seed20260604_qwen3_1p7b_coldstart_frac25_beta01_p40_s2steps20_s3steps40.parquet",
    "kodcode_stage2_after_s1_seed20260604_qwen3_1p7b_coldstart_frac50_beta01_p40_handoff_s2steps20.parquet",
    "kodcode_stage3_after_s2_seed20260604_qwen3_1p7b_coldstart_frac50_beta01_p40_s2steps20_s3steps40.parquet",
)


def command(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=False)


def add(checks: list[dict[str, object]], name: str, ok: bool, detail: object) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-active", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--normalized-manifest")
    args = parser.parse_args()
    checks: list[dict[str, object]] = []

    repo_real = REPO.resolve()
    compat = Path("/data-1/verl07/verl")
    add(checks, "repo_topology", repo_real == Path("/data-1/code/verl"), str(repo_real))
    add(checks, "compat_symlink", compat.resolve() == repo_real, str(compat.resolve()))
    add(
        checks,
        "checkpoint_mount",
        Path("/data-1/checkpoints").resolve() == Path("/data-2/checkpoints"),
        str(Path("/data-1/checkpoints").resolve()),
    )

    profile = command(
        "bash",
        "-lc",
        f"source {PROFILE!s}; stage123_profile_snapshot",
    )
    profile_text = profile.stdout
    profile_hash = hashlib.sha256(profile_text.encode()).hexdigest()
    add(checks, "resource_profile", profile.returncode == 0, {"sha256": profile_hash})

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
        add(checks, "model_identity", identity_ok, {"manifest_sha256": manifest.get("manifest_sha256"), "normalized_matches_repo": normalized_matches_repo, "phases": phase_readiness})
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        add(checks, "model_identity", False, {"error": str(exc)})
    finally:
        if not args.normalized_manifest:
            normalized_path.unlink(missing_ok=True)

    shard_details: list[dict[str, object]] = []
    shards_ok = True
    for name in SHARDS:
        shard = DATA_DIR / name
        manifest = shard.with_suffix(".manifest.json")
        ok = shard.is_file() and manifest.is_file()
        detail: dict[str, object] = {"path": str(shard), "manifest": str(manifest)}
        if ok:
            detail["sha256"] = sha256(shard)
            try:
                manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
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
                    }
                )
                ok = expected_sha == actual_sha and selected_rows == expected_rows
            except (OSError, json.JSONDecodeError) as exc:
                detail["manifest_error"] = str(exc)
                ok = False
        detail["ok"] = ok
        shards_ok = shards_ok and ok
        shard_details.append(detail)
    add(checks, "dataset_shards", shards_ok, shard_details)

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

    docker = command("docker", "image", "inspect", "verl-harness:latest", "--format", "{{.Id}}")
    add(checks, "docker_image", docker.returncode == 0, docker.stdout.strip() or docker.stderr.strip())
    release_paths = (
        Path("/data-1/experiment_registry/experiment_registry.sqlite"),
        Path("/data-1/wandb_runs"),
        REPO / "scripts/training_result_release_gate.py",
        REPO / "scripts/code_task_training_release_hook.sh",
    )
    add(checks, "release_paths", all(path.exists() for path in release_paths), [str(path) for path in release_paths])

    active = command("tmux", "list-sessions", "-F", "#{session_name}")
    active_names = [name for name in active.stdout.splitlines() if name.startswith("stage123_") or name == "code_task_qwen3_1p7b_stage123_queue"]
    add(checks, "no_conflicting_run", args.allow_active or not active_names, active_names)

    report = {
        "schema_version": 1,
        "gate": "qwen3_1p7b_stage123_preflight",
        "ok": all(bool(item["ok"]) for item in checks),
        "checks": checks,
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
