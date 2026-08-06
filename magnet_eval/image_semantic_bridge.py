"""MAGNET bridge for the two-task FSBM semantic proxy certificate."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from magnet_eval.project import find_project_root


TASKS = {
    "gender": {
        "experiment": "ffhq_gen",
        "label_file": "gender.npy",
        "translation": "female->male",
    },
    "age": {
        "experiment": "ffhq_age",
        "label_file": "age.npy",
        "translation": "adult->child",
    },
}

EXPECTED_ASSET_SHA256 = {
    "evaluator": "bc2d910959852da2d18b3103be6fea00d3753462f7ead48f09dd9be85198591a",
    "latents": "46237baaccace670b953de74d62a689b9bdacad50207f97b166c4746f79fb684",
    "gender": {
        "labels": "cd1aabbe5b0cfaa917ec94c9b6a675376ceac69b3a430644b016f36da1421538",
        "checkpoint": "3a2278568b46b9b98dd4568bb599ff08455cc37d124045c8d12f7c8e8c0c9e1f",
        "hydra_config": "e68588c7b0afe6fd7cb98e46b416ed1e584fb8445113b21992b6670a07697444",
    },
    "age": {
        "labels": "0662e232785ca5674bd8c1cf9699bb570ec7b6265191ee62c2d05688994bfe25",
        "checkpoint": "18050bf31919fd81691271ab5708e9c86ae50e0cf5a6476448a7bfcfc6522551",
        "hydra_config": "4a9b1b5dc3d311081a20307ec1c074191d8f429e2c1818396e0b7053d0f1802e",
    },
}

EXPECTED_PRIOR_EXCLUSION_SHA256 = {
    "gender": [
        "0e3c1c0447c82997cd3cfba55d74997ae60a56fd0e3168eb730c3965abf572a1"
    ],
    "age": [
        "9979a7aa262e9ef235e170b0a883049e28c0ede950e61736e45c2fce7dd82c45"
    ],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_file(root: Path, value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _verify_sha256(path: Path, expected: str, label: str) -> str:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual} ({path})"
        )
    return actual


def combine_exclusion_manifests(
    manifests: Iterable[str | Path],
    destination: str | Path,
    *,
    train_size: int = 60_000,
) -> Dict[str, Any]:
    """Write a compact union of previously inspected validation sources.

    Generation manifests use ``source_global_index``. Full evaluator partition
    manifests use ``partition`` and ``global_latent_index``; for those files,
    only validation-side source rows are selected. Supporting both formats
    lets the card exclude classifier/calibration identities as well as the
    previously generated fit, certificate, and evaluation identities.
    """

    manifest_paths = [Path(path).expanduser().resolve() for path in manifests]
    if not manifest_paths:
        raise ValueError("at least one prior manifest is required for a fresh run")

    source_indices = set()
    inputs = []
    for path in manifest_paths:
        if not path.is_file():
            raise FileNotFoundError(f"prior generation manifest not found: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            if "source_global_index" in fields:
                kind = "generation_or_compact_exclusion"
            elif {"partition", "global_latent_index"} <= fields:
                kind = "model_partitions_validation_source"
            else:
                raise ValueError(
                    "prior manifest must contain source_global_index or "
                    f"partition/global_latent_index columns: {path}"
                )
            before = len(source_indices)
            rows = 0
            selected_rows = 0
            for row in reader:
                rows += 1
                if kind == "generation_or_compact_exclusion":
                    source_index = int(row["source_global_index"])
                else:
                    source_index = int(row["global_latent_index"])
                    if not row["partition"].endswith("_source"):
                        continue
                    if source_index < int(train_size):
                        continue
                selected_rows += 1
                source_indices.add(source_index)
        inputs.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "kind": kind,
                "rows": rows,
                "selected_validation_source_rows": selected_rows,
                "new_unique_source_identities": len(source_indices) - before,
            }
        )

    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source_global_index",))
        writer.writeheader()
        for source_index in sorted(source_indices):
            writer.writerow({"source_global_index": source_index})

    return {
        "path": str(output),
        "sha256": _sha256(output),
        "n_unique_source_identities": len(source_indices),
        "inputs": inputs,
    }


def run_task_evaluation(
    *,
    task: str,
    project_root: str | Path,
    checkpoint: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    prior_manifests: Sequence[str | Path],
    seed: int = 20260731,
    n_certificate: int = 512,
) -> Dict[str, Any]:
    """Run one fresh identity-disjoint semantic proxy certificate."""

    if task not in TASKS:
        raise ValueError(f"task must be one of {sorted(TASKS)}")
    root = find_project_root(project_root)
    spec = TASKS[task]
    checkpoint_path = _resolve_file(root, checkpoint, f"{task} checkpoint")
    hydra_config = _resolve_file(
        root,
        checkpoint_path.parent.parent / ".hydra" / "config.yaml",
        f"{task} checkpoint Hydra config",
    )

    data_path = Path(data_dir).expanduser()
    if not data_path.is_absolute():
        data_path = root / data_path
    data_path = data_path.resolve()
    latent_path = _resolve_file(root, data_path / "latents.npy", "FFHQ latent array")
    label_path = _resolve_file(
        root, data_path / str(spec["label_file"]), f"{task} labels"
    )
    evaluator_path = _resolve_file(
        root, root / "image_semantic_alarm.py", "semantic evaluator"
    )
    task_hashes = EXPECTED_ASSET_SHA256[task]
    _verify_sha256(
        evaluator_path, str(EXPECTED_ASSET_SHA256["evaluator"]), "evaluator"
    )
    _verify_sha256(
        latent_path, str(EXPECTED_ASSET_SHA256["latents"]), "FFHQ latents"
    )
    _verify_sha256(label_path, str(task_hashes["labels"]), f"{task} labels")
    _verify_sha256(
        checkpoint_path, str(task_hashes["checkpoint"]), f"{task} checkpoint"
    )
    _verify_sha256(
        hydra_config,
        str(task_hashes["hydra_config"]),
        f"{task} checkpoint Hydra config",
    )

    destination = Path(output_dir).expanduser()
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    resolved_manifests = [
        _resolve_file(root, value, f"{task} prior manifest")
        for value in prior_manifests
    ]
    prior_hashes = [_sha256(path) for path in resolved_manifests]
    if prior_hashes != EXPECTED_PRIOR_EXCLUSION_SHA256[task]:
        raise ValueError(
            f"{task} prior-exclusion SHA-256 mismatch: expected "
            f"{EXPECTED_PRIOR_EXCLUSION_SHA256[task]}, got {prior_hashes}"
        )
    exclusion = combine_exclusion_manifests(
        resolved_manifests, destination / "excluded_prior_source_identities.csv"
    )

    import sys

    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    import image_semantic_alarm

    argv = [
        "--experiment",
        str(spec["experiment"]),
        "--ckpt-file",
        str(checkpoint_path),
        "--data-dir",
        str(data_path),
        "--train-size",
        "60000",
        "--device",
        "cpu",
        "--output",
        str(destination),
        "--n-classifier-test",
        "64",
        "--n-monitor-calib",
        "64",
        "--n-output-test",
        "64",
        "--n-fit",
        "384",
        "--n-certificate",
        str(int(n_certificate)),
        "--n-eval",
        "192",
        "--exclude-source-indices-csv",
        str(exclusion["path"]),
        "--fit-runs",
        "3",
        "--evaluation-runs",
        "5",
        "--generation-nfe",
        "200",
        "--trajectory-steps",
        "6",
        "--stability-nfe",
        "100",
        "--delta0",
        "0.1",
        "--k-probes",
        "4",
        "--probe-aggregation",
        "max",
        "--old-alpha-total",
        "0.1",
        "--n-teachers",
        "5",
        "--teacher-consensus",
        "0.8",
        "--runtime-train-fraction",
        "0.5",
        "--attribute-regularization",
        "1.0",
        "--failure-model-c",
        "1.0",
        "--crossfit-folds",
        "5",
        "--tuning-risk-fraction",
        "0.5",
        "--target-risk",
        "0.10",
        "--minimum-coverage",
        "0.30",
        "--certificate-beta",
        "0.025",
        "--certify-gate",
        "semantic",
        "--confidence",
        "0.95",
        "--n-bootstrap",
        "2000",
        "--batch-size",
        "256",
        "--num-threads",
        "4",
        "--seed",
        str(int(seed)),
        "--save-generated-latents",
    ]
    summary = image_semantic_alarm.main(argv)
    summary["magnet_bridge"] = {
        "task": task,
        "arguments": argv,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
        },
        "checkpoint_hydra_config": {
            "path": str(hydra_config),
            "sha256": _sha256(hydra_config),
        },
        "frozen_assets": {
            "evaluator": {
                "path": str(evaluator_path),
                "sha256": _sha256(evaluator_path),
            },
            "latents": {"path": str(latent_path), "sha256": _sha256(latent_path)},
            "labels": {"path": str(label_path), "sha256": _sha256(label_path)},
        },
        "fresh_identity_exclusion": exclusion,
    }
    with (destination / "magnet_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def run_two_task_evaluation(
    *,
    project_root: str | Path,
    checkpoints: Mapping[str, str | Path],
    data_dir: str | Path,
    output_dir: str | Path,
    prior_manifests: Mapping[str, Sequence[str | Path]],
    seed: int = 20260731,
    n_certificate: int = 512,
) -> Dict[str, Any]:
    """Run the requested task certificates under one family error budget.

    The historical entry point name is retained for compatibility, but callers
    may provide any non-empty subset of ``TASKS``. This lets an AIQ card omit a
    task completely instead of executing it and merely hiding its result.
    """

    root = find_project_root(project_root)
    base = Path(output_dir).expanduser()
    if not base.is_absolute():
        base = root / base
    base = base.resolve()

    checkpoint_tasks = set(checkpoints)
    manifest_tasks = set(prior_manifests)
    unknown = sorted((checkpoint_tasks | manifest_tasks) - set(TASKS))
    if unknown:
        raise KeyError(f"unknown image semantic tasks: {unknown}")
    if checkpoint_tasks != manifest_tasks:
        raise KeyError(
            "checkpoint and prior-manifest task sets differ: "
            f"checkpoints={sorted(checkpoint_tasks)}, "
            f"prior_manifests={sorted(manifest_tasks)}"
        )
    selected_tasks = tuple(task for task in TASKS if task in checkpoint_tasks)
    if not selected_tasks:
        raise ValueError("at least one image semantic task is required")

    results = {}
    for task in selected_tasks:
        results[task] = run_task_evaluation(
            task=task,
            project_root=root,
            checkpoint=checkpoints[task],
            data_dir=data_dir,
            output_dir=base / task,
            prior_manifests=prior_manifests[task],
            seed=seed,
            n_certificate=n_certificate,
        )
    evaluation = {
        "schema_version": "fsbm-image-semantic-magnet-v1",
        "seed": int(seed),
        "n_certificate_per_task": int(n_certificate),
        "selected_tasks": list(selected_tasks),
        "tasks": results,
    }
    write_two_task_summary(evaluation, base / "magnet_summary.json")
    return evaluation


def _resolve_python_executable(value: str | Path) -> Path:
    """Resolve a Python executable supplied as either a path or command name."""

    text = str(value)
    discovered = shutil.which(text)
    candidate = Path(discovered if discovered is not None else text).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"scientific Python executable not found: {value}") from error
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise PermissionError(f"scientific Python is not executable: {resolved}")
    return resolved


def run_two_task_evaluation_isolated(
    *,
    scientific_python: str | Path,
    project_root: str | Path,
    checkpoints: Mapping[str, str | Path],
    data_dir: str | Path,
    output_dir: str | Path,
    prior_manifests: Mapping[str, Sequence[str | Path]],
    seed: int = 20260731,
    n_certificate: int = 512,
) -> Dict[str, Any]:
    """Run the scientific evaluation in a compatible Python interpreter.

    MAGNET 0.0.2 requires Python 3.11 or newer, while the frozen FSBM
    scientific environment uses Python 3.10. When the requested interpreter
    differs from the MAGNET interpreter, this function passes a JSON request
    to this module in a subprocess and reads back the complete evaluation.
    """

    root = find_project_root(project_root)
    python = _resolve_python_executable(scientific_python)
    current_python = Path(sys.executable).resolve(strict=True)
    arguments = {
        "project_root": str(root),
        "checkpoints": {task: str(path) for task, path in checkpoints.items()},
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "prior_manifests": {
            task: [str(path) for path in manifests]
            for task, manifests in prior_manifests.items()
        },
        "seed": int(seed),
        "n_certificate": int(n_certificate),
    }
    if python == current_python:
        previous_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            return run_two_task_evaluation(**arguments)
        finally:
            sys.dont_write_bytecode = previous_dont_write_bytecode

    base = Path(output_dir).expanduser()
    if not base.is_absolute():
        base = root / base
    base = base.resolve()
    base.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(root), existing_pythonpath) if item
    )
    with tempfile.TemporaryDirectory(
        prefix=".magnet-scientific-", dir=base
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        request_path = temporary / "request.json"
        response_path = temporary / "response.json"
        with request_path.open("w", encoding="utf-8") as handle:
            json.dump(arguments, handle, indent=2, sort_keys=True)
            handle.write("\n")
        subprocess.run(
            [
                str(python),
                "-m",
                "magnet_eval.image_semantic_bridge",
                "--request-json",
                str(request_path),
                "--response-json",
                str(response_path),
            ],
            cwd=root,
            env=environment,
            check=True,
        )
        if not response_path.is_file():
            raise RuntimeError(
                "scientific evaluation completed without writing its response: "
                f"{response_path}"
            )
        with response_path.open(encoding="utf-8") as handle:
            evaluation = json.load(handle)
    if not isinstance(evaluation, dict) or "tasks" not in evaluation:
        raise ValueError("scientific evaluation returned an invalid response")
    return evaluation


def summarize_two_task_evaluation(evaluation: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract certificate and descriptive metrics with unambiguous names."""

    task_summaries = {}
    total_family_beta = 0.0
    for task, report in evaluation["tasks"].items():
        certificate = report["certificate"]
        partitions = report["partitions"]
        ablations = {
            row["gate"]: row for row in report["final_evaluation"]["ablation"]
        }
        selected = report["final_evaluation"]["selected_gate_metrics"]
        bridge = report["magnet_bridge"]
        file_hashes = report["provenance"]["files"]
        source_git = report["provenance"]["git"]
        checkpoint_config = report["provenance"]["checkpoint_config"]
        total_family_beta += float(certificate["family_beta"])
        certificate_pass = bool(
            certificate["assumptions_satisfied"]
            and certificate["risk_criterion_pass"]
            and certificate["coverage_criterion_pass"]
        )
        no_gate = ablations["none"]
        old_monitor = ablations["old_three_signal"]
        separation_pass = bool(
            selected["risk_ci_high"] < no_gate["risk_ci_low"]
            and selected["risk_ci_high"] < old_monitor["risk_ci_low"]
            and selected["coverage_ci_low"] >= certificate["minimum_coverage"]
        )
        task_summaries[task] = {
            "experiment": report["experiment"],
            "translation": report["translation"],
            "source_git_commit": str(source_git["commit"]),
            "source_git_branch": str(source_git["branch"]),
            "source_git_dirty": bool(source_git["dirty"]),
            "keypoints_loaded": bool(checkpoint_config["keypoints_loaded"]),
            "proxy_certificate_pass": certificate_pass,
            "fresh_holdout_separation_pass": separation_pass,
            "assumptions_satisfied": bool(certificate["assumptions_satisfied"]),
            "certificate_data_used_for_threshold": bool(
                report["threshold_tuning"]["certificate_data_used"]
            ),
            "audit_outputs_excluded_from_features": bool(
                report["failure_predictor"]["audit_outputs_excluded_from_features"]
            ),
            "audit_independence_attested": bool(
                report["audit_source"]["independence_attested"]
            ),
            "threshold_frozen_before_certificate": bool(
                certificate["threshold_frozen_before_certificate"]
            ),
            "one_output_per_unique_identity": bool(
                certificate["one_output_per_unique_identity"]
            ),
            "certificate_identities": int(certificate["n_unique_source_identities"]),
            "certificate_admitted": int(certificate["n_admitted"]),
            "certificate_failures_among_admitted": int(
                certificate["n_failures_among_admitted"]
            ),
            "certificate_empirical_proxy_risk": float(
                certificate["empirical_selective_failure_risk"]
            ),
            "certificate_proxy_risk_ucb": float(
                certificate["risk_upper_confidence_bound"]
            ),
            "target_proxy_risk": float(certificate["target_risk"]),
            "certificate_empirical_admission": float(
                certificate["empirical_coverage"]
            ),
            "certificate_admission_lcb": float(
                certificate["coverage_lower_confidence_bound"]
            ),
            "minimum_admission": float(certificate["minimum_coverage"]),
            "task_family_beta": float(certificate["family_beta"]),
            "audit_label_source": certificate["audit_label_source"],
            "runtime_teacher_train_disjoint": bool(
                partitions["runtime_teacher_train_disjoint"]
            ),
            "fit_certificate_evaluation_disjoint": bool(
                partitions["generated_fit_certificate_evaluation_disjoint"]
            ),
            "prior_source_identities_excluded": int(
                partitions["prior_round_source_exclusion"]["n_excluded"]
            ),
            "prior_exclusion_sha256": [
                item["sha256"]
                for item in bridge["fresh_identity_exclusion"]["inputs"]
            ],
            "asset_sha256": {
                "evaluator": file_hashes["script"]["sha256"],
                "checkpoint": file_hashes["checkpoint"]["sha256"],
                "hydra_config": bridge["checkpoint_hydra_config"]["sha256"],
                "latents": file_hashes["latents"]["sha256"],
                "labels": file_hashes["labels"]["sha256"],
            },
            "fit_identities": int(partitions["fit_identities"]),
            "evaluation_identities": int(partitions["evaluation_identities"]),
            "evaluation_runs_per_identity": int(
                partitions["evaluation_runs_per_identity"]
            ),
            "descriptive_five_run_proxy_risk": float(
                selected["selective_failure_risk"]
            ),
            "descriptive_five_run_proxy_risk_ci_low": float(
                selected["risk_ci_low"]
            ),
            "descriptive_five_run_proxy_risk_ci_high": float(
                selected["risk_ci_high"]
            ),
            "descriptive_five_run_admission": float(selected["coverage"]),
            "descriptive_five_run_admission_ci_low": float(
                selected["coverage_ci_low"]
            ),
            "descriptive_five_run_admission_ci_high": float(
                selected["coverage_ci_high"]
            ),
            "descriptive_five_run_failure_capture": float(
                selected["failure_capture_rate"]
            ),
            "descriptive_no_gate_proxy_risk": float(
                no_gate["selective_failure_risk"]
            ),
            "descriptive_no_gate_proxy_risk_ci_low": float(
                no_gate["risk_ci_low"]
            ),
            "descriptive_no_gate_proxy_risk_ci_high": float(
                no_gate["risk_ci_high"]
            ),
            "descriptive_old_monitor_proxy_risk": float(
                old_monitor["selective_failure_risk"]
            ),
            "descriptive_old_monitor_proxy_risk_ci_low": float(
                old_monitor["risk_ci_low"]
            ),
            "descriptive_old_monitor_proxy_risk_ci_high": float(
                old_monitor["risk_ci_high"]
            ),
        }
    return {
        "tasks": task_summaries,
        "selected_tasks": list(task_summaries),
        "family_beta": total_family_beta,
        "simultaneous_confidence_lower_bound": 1.0 - total_family_beta,
        # Backward-compatible names used by archived two-task results.
        "two_task_family_beta": total_family_beta,
        "two_task_simultaneous_confidence_lower_bound": 1.0 - total_family_beta,
    }


def write_two_task_summary(
    evaluation: Mapping[str, Any], destination: str | Path
) -> Dict[str, Any]:
    """Persist the compact result used by the MAGNET claim."""

    summary = summarize_two_task_evaluation(evaluation)
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def run_evaluation(**kwargs: Any) -> Dict[str, Any]:
    """Task-subset alias for the historical two-task runner."""

    return run_two_task_evaluation(**kwargs)


def run_evaluation_isolated(**kwargs: Any) -> Dict[str, Any]:
    """Task-subset alias for the historical isolated runner."""

    return run_two_task_evaluation_isolated(**kwargs)


def summarize_evaluation(evaluation: Mapping[str, Any]) -> Dict[str, Any]:
    """Task-subset alias for the historical compact summarizer."""

    return summarize_two_task_evaluation(evaluation)


def _run_json_request(request_path: Path, response_path: Path) -> None:
    with request_path.open(encoding="utf-8") as handle:
        request = json.load(handle)
    if not isinstance(request, dict):
        raise TypeError("scientific evaluation request must be a JSON object")
    allowed = {
        "project_root",
        "checkpoints",
        "data_dir",
        "output_dir",
        "prior_manifests",
        "seed",
        "n_certificate",
    }
    unknown = sorted(set(request) - allowed)
    required = allowed - {"seed", "n_certificate"}
    missing = sorted(required - set(request))
    if unknown or missing:
        raise ValueError(
            f"invalid scientific evaluation request; missing={missing}, "
            f"unknown={unknown}"
        )
    evaluation = run_two_task_evaluation(**request)
    response_path.parent.mkdir(parents=True, exist_ok=True)
    with response_path.open("w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run an FSBM two-task scientific evaluation JSON request."
    )
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--response-json", type=Path, required=True)
    arguments = parser.parse_args()
    _run_json_request(arguments.request_json, arguments.response_json)


if __name__ == "__main__":
    _main()
