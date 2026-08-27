#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ADAPTER_DIR = Path(__file__).resolve().parent
SUBMIT = ADAPTER_DIR / "submit_math_stage1_grpo.sh"
WORKER = ADAPTER_DIR / "worker_math_stage1_grpo.sh"
VERIFY = ADAPTER_DIR / "verify_pueue_adapter.py"
VALIDATE = ADAPTER_DIR / "validate_a800_admission.py"
RECIPE_ROOT = ADAPTER_DIR.parents[2]
VERL_ROOT = RECIPE_ROOT.parent
SCHEDULER_DIR = RECIPE_ROOT / "on_policy_wdl_sft" / "standard_grpo" / "scheduler"


class PueueAdapterTests(unittest.TestCase):
    def run_submit(self, root: Path, **updates: str) -> subprocess.CompletedProcess[str]:
        env_file = root / "runtime.env"
        env_file.write_text("WANDB_MODE=offline\n", encoding="utf-8")
        env = {
            "PATH": os.environ["PATH"],
            "HOME": os.environ.get("HOME", "/tmp"),
            "PUEUE_GRPO_REPO_ROOT": str(VERL_ROOT),
            "PUEUE_GRPO_OUTPUT_ROOT": str(root / "output"),
            "PUEUE_GRPO_RECEIPT_ROOT": str(root / "receipts"),
            "PUEUE_GRPO_RUNTIME_ENV_FILE": str(env_file),
            "PUEUE_GRPO_DRY_RUN": "1",
        }
        env.update(updates)
        return subprocess.run(
            ["bash", str(SUBMIT)],
            cwd=RECIPE_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_dry_run_reaches_gpu8_worker_without_submission(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gon-41-") as directory:
            result = self.run_submit(Path(directory))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DRY_RUN pueue add --group gpu8", result.stdout)
        self.assertIn("--print-task-id", result.stdout)
        self.assertIn("worker_math_stage1_grpo.sh", result.stdout)
        self.assertIn("bash\\ -c", result.stdout)

    def test_missing_output_root_fails_before_queue(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gon-41-") as directory:
            result = self.run_submit(Path(directory), PUEUE_GRPO_OUTPUT_ROOT="")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("DRY_RUN pueue add", result.stdout)

    def test_repository_local_receipt_root_fails_before_queue(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gon-41-") as directory:
            result = self.run_submit(
                Path(directory),
                PUEUE_GRPO_RECEIPT_ROOT=str(RECIPE_ROOT / "mutable-receipts"),
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside both repositories", result.stderr)

    def test_different_recipe_checkout_fails_before_queue(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gon-41-") as directory:
            root = Path(directory)
            fake_checkout = root / "fake-verl"
            (fake_checkout / "verl").mkdir(parents=True)
            (fake_checkout / "recipe").mkdir()
            result = self.run_submit(root, PUEUE_GRPO_REPO_ROOT=str(fake_checkout))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("reviewed recipe checkout", result.stderr)
        self.assertNotIn("DRY_RUN pueue add", result.stdout)

    def test_worker_rejects_runtime_env_drift(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gon-41-worker-") as directory:
            root = Path(directory)
            env_file = root / "runtime.env"
            env_file.write_text("WANDB_MODE=offline\n", encoding="utf-8")
            candidate = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=RECIPE_ROOT,
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
            root_candidate = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=VERL_ROOT,
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
            result = subprocess.run(
                [
                    "bash",
                    str(WORKER),
                    str(VERL_ROOT),
                    str(root / "output"),
                    str(root / "receipts"),
                    str(env_file),
                    root_candidate,
                    candidate,
                    "0" * 64,
                ],
                env={"PATH": os.environ["PATH"], "PUEUE_TASK_ID": "42"},
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("runtime environment changed after admission", result.stderr)

    def test_worker_rejects_recipe_candidate_drift(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gon-41-worker-") as directory:
            root = Path(directory)
            env_file = root / "runtime.env"
            env_file.write_text("WANDB_MODE=offline\n", encoding="utf-8")
            result = subprocess.run(
                [
                    "bash",
                    str(WORKER),
                    str(VERL_ROOT),
                    str(root / "output"),
                    str(root / "receipts"),
                    str(env_file),
                    subprocess.run(
                        ["git", "rev-parse", "HEAD"], cwd=VERL_ROOT, text=True, capture_output=True, check=True
                    ).stdout.strip(),
                    "0" * 40,
                    "0" * 64,
                ],
                env={"PATH": os.environ["PATH"], "PUEUE_TASK_ID": "42"},
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("queued recipe checkout drifted", result.stderr)

    def test_worker_rejects_dirty_experiment_entry(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gon-41-dirty-") as directory:
            root = Path(directory)
            checkout = root / "checkout"
            subprocess.run(
                ["git", "clone", "--shared", "--quiet", str(RECIPE_ROOT), str(checkout / "recipe")],
                check=True,
            )
            (checkout / "verl").mkdir()
            (checkout / "verl" / "__init__.py").write_text("", encoding="utf-8")
            subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
            subprocess.run(["git", "-C", str(checkout), "config", "user.email", "test@example.com"], check=True)
            subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", str(checkout), "add", "verl", "recipe"], check=True)
            subprocess.run(["git", "-C", str(checkout), "commit", "--quiet", "-m", "test fixture"], check=True)
            root_candidate = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=checkout, text=True, capture_output=True, check=True
            ).stdout.strip()
            candidate = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=checkout / "recipe",
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
            entry = checkout / "recipe" / "on_policy_wdl_sft" / "standard_grpo" / "run_math_stage1_grpo.sh"
            entry.write_text(entry.read_text(encoding="utf-8") + "\n# dirty canary\n", encoding="utf-8")
            env_file = root / "runtime.env"
            env_file.write_text("WANDB_MODE=offline\n", encoding="utf-8")
            result = subprocess.run(
                [
                    "bash",
                    str(WORKER),
                    str(checkout),
                    str(root / "output"),
                    str(root / "receipts"),
                    str(env_file),
                    root_candidate,
                    candidate,
                    hashlib.sha256(env_file.read_bytes()).hexdigest(),
                ],
                env={"PATH": os.environ["PATH"], "PUEUE_TASK_ID": "42"},
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("tracked, staged or untracked changes", result.stderr)

    def test_boolean_only_admission_receipt_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gon-35-receipt-") as directory:
            root = Path(directory)
            env_file = root / "runtime.env"
            env_file.write_text("WANDB_MODE=offline\n", encoding="utf-8")
            output_root = root / "output"
            receipt_root = root / "receipts"
            candidate = "1" * 40
            receipt = {
                "schema_version": 1,
                "batch_id": "GON-35",
                "recipe_candidate_sha": candidate,
                "scheduler": "pueue",
                "group": "gpu8",
                "group_concurrency": 1,
                "host_launcher": "verl-dev-run --a800-dev-profile",
                "image_digest": "sha256:d380888dc8a10796c7f841e341bd775c2d6500ede539f4ea16bb7bf0de92665d",
                "p0_config_match": True,
                "p1_review_complete": True,
                "full_gpu_submission_allowed": True,
                "runtime_env_sha256": hashlib.sha256(env_file.read_bytes()).hexdigest(),
                "output_root": str(output_root),
                "receipt_root": str(receipt_root),
            }
            receipt_file = root / "admission.json"
            receipt_file.write_text(json.dumps(receipt), encoding="utf-8")
            result = subprocess.run(
                [
                    "python3",
                    str(VALIDATE),
                    "--receipt",
                    str(receipt_file),
                    "--root-candidate",
                    "2" * 40,
                    "--recipe-candidate",
                    candidate,
                    "--runtime-env-file",
                    str(env_file),
                    "--output-root",
                    str(output_root),
                    "--receipt-root",
                    str(receipt_root),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("model_sha256", result.stderr)

    def test_removed_candidate_guard_canary_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gon-41-static-") as directory:
            target = Path(directory) / "pueue"
            shutil.copytree(ADAPTER_DIR, target)
            worker = target / "worker_math_stage1_grpo.sh"
            content = worker.read_text(encoding="utf-8").replace(
                '[[ "$actual_recipe_candidate" == "$expected_recipe_candidate" ]]',
                "[[ 1 == 1 ]]",
            )
            worker.write_text(content, encoding="utf-8")
            result = subprocess.run(
                ["python3", str(VERIFY), "--adapter-dir", str(target)],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)

    def assert_static_mutation_fails(self, mutation: str) -> None:
        with tempfile.TemporaryDirectory(prefix="gon-41-static-") as directory:
            target = Path(directory) / "pueue"
            shutil.copytree(ADAPTER_DIR, target)
            worker = target / "worker_math_stage1_grpo.sh"
            worker.write_text(worker.read_text(encoding="utf-8") + mutation, encoding="utf-8")
            result = subprocess.run(
                ["python3", str(VERIFY), "--adapter-dir", str(target)],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)

    def test_scientific_override_canary_fails(self) -> None:
        self.assert_static_mutation_fails("\nLR=5e-7\n")

    def test_nested_launcher_canary_fails(self) -> None:
        self.assert_static_mutation_fails("\nsbatch forbidden.sh\n")

    def test_changed_entry_canary_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gon-41-static-") as directory:
            target = Path(directory) / "pueue"
            shutil.copytree(ADAPTER_DIR, target)
            worker = target / "worker_math_stage1_grpo.sh"
            content = worker.read_text(encoding="utf-8").replace(
                "run_math_stage1_grpo.sh", "run_math_cold_start_grpo.sh"
            )
            worker.write_text(content, encoding="utf-8")
            result = subprocess.run(
                ["python3", str(VERIFY), "--adapter-dir", str(target)],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)

    def test_unclassified_audit_difference_canary_fails(self) -> None:
        audit = json.loads((SCHEDULER_DIR / "scheduler_audit.json").read_text(encoding="utf-8"))
        audit["p1"][0]["decision"] = ""
        with tempfile.TemporaryDirectory(prefix="gon-41-audit-") as directory:
            path = Path(directory) / "audit.json"
            path.write_text(json.dumps(audit), encoding="utf-8")
            result = subprocess.run(
                [
                    "python3",
                    str(SCHEDULER_DIR / "verify_scheduler_audit.py"),
                    "--audit",
                    str(path),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
