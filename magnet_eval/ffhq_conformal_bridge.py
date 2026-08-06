"""MAGNET bridge for FFHQ conformal transport-stability evaluation.

The bridge runs the same repeated split-conformal score used at runtime: the
maximum endpoint amplification over four sampled isotropic perturbations under
synchronous SDE noise.  It deliberately keeps this claim separate from the
semantic-proxy evaluation and presents it with the paired LPIPS comparison in
the FFHQ AIQ card.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping

from magnet_eval.project import find_project_root


TASKS = {
    "gender": {
        "experiment": "ffhq_gen",
        "label_file": "gender.npy",
        "translation": "female->male",
        "n_test": 1000,
    },
    "age": {
        "experiment": "ffhq_age",
        "label_file": "age.npy",
        "translation": "adult->child",
        "n_test": 600,
    },
}

EXPECTED_ASSET_SHA256 = {
    "evaluator": "48437c29bec9f4f02d7def91b58ddd5db639339819a9002b60c60fe9609bba2a",
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


def _resolve_python_executable(value: str | Path) -> Path:
    text = str(value)
    discovered = shutil.which(text)
    candidate = Path(discovered if discovered is not None else text).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"scientific Python executable not found: {value}"
        ) from error
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise PermissionError(f"scientific Python is not executable: {resolved}")
    return resolved


def run_task_evaluation(
    *,
    task: str,
    project_root: str | Path,
    checkpoint: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    seed: int = 20260716,
    n_splits: int = 5,
    n_calib: int = 1000,
    delta: float = 0.1,
    alpha: float = 0.05,
    pac_beta: float = 0.05,
    n_probes: int = 4,
    nfe: int = 50,
    batch_size: int = 128,
    torch_threads: int = 1,
) -> Dict[str, Any]:
    """Run one frozen FFHQ repeated split-conformal audit."""

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
    runtime_data_path = (root / "data").resolve()
    if data_path != runtime_data_path:
        raise ValueError(
            "the frozen FFHQ dataset loader resolves project_root/data; stage "
            f"the pinned arrays there instead of {data_path}"
        )
    latent_path = _resolve_file(root, data_path / "latents.npy", "FFHQ latents")
    label_path = _resolve_file(
        root, data_path / str(spec["label_file"]), f"{task} labels"
    )
    evaluator_path = _resolve_file(
        root, root / "darpa_conformal_eval.py", "conformal evaluator"
    )

    expected = EXPECTED_ASSET_SHA256[task]
    asset_sha256 = {
        "evaluator": _verify_sha256(
            evaluator_path,
            str(EXPECTED_ASSET_SHA256["evaluator"]),
            "conformal evaluator",
        ),
        "checkpoint": _verify_sha256(
            checkpoint_path, str(expected["checkpoint"]), f"{task} checkpoint"
        ),
        "hydra_config": _verify_sha256(
            hydra_config,
            str(expected["hydra_config"]),
            f"{task} checkpoint Hydra config",
        ),
        "latents": _verify_sha256(
            latent_path, str(EXPECTED_ASSET_SHA256["latents"]), "FFHQ latents"
        ),
        "labels": _verify_sha256(
            label_path, str(expected["labels"]), f"{task} labels"
        ),
    }

    destination = Path(output_dir).expanduser()
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()

    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    import torch
    import darpa_conformal_eval
    from fsbm.utils import restore_model

    if torch_threads < 1:
        raise ValueError("torch_threads must be >= 1")
    torch.set_num_threads(int(torch_threads))
    settings = darpa_conformal_eval.EvaluationSettings(
        directions=("fwd", "bwd"),
        deltas=(float(delta),),
        alphas=(float(alpha),),
        n_splits=int(n_splits),
        n_calib=int(n_calib),
        n_test=int(spec["n_test"]),
        n_probes=int(n_probes),
        aggregation="max",
        nfe=int(nfe),
        batch_size=int(batch_size),
        seed=int(seed),
        device="cpu",
        baseline_terminal_error=0.0,
        tolerance=None,
        pac_beta=float(pac_beta),
    )
    settings.validate()
    model, cfg = restore_model(
        checkpoint_path, device=settings.device, load_keypoints=False
    )
    split_rows, aggregate_rows, data_provenance = (
        darpa_conformal_eval.evaluate_model(model, settings, verbose=True)
    )
    provenance = darpa_conformal_eval.build_provenance(
        checkpoint_path, cfg, settings, data_provenance
    )
    paths = darpa_conformal_eval.write_outputs(
        destination,
        split_rows,
        aggregate_rows,
        provenance,
        figures=True,
    )

    overlaps = [row.get("identity_overlap_count") for row in split_rows]
    report = {
        "schema_version": "fsbm-ffhq-conformal-magnet-v1",
        "task": task,
        "experiment": str(spec["experiment"]),
        "translation": str(spec["translation"]),
        "settings": settings.to_dict(),
        "claims": {
            "ordinary_marginal": (
                "For a jointly exchangeable calibration/new K-probe maximum-"
                "amplification score, the ordinary conformal rank threshold "
                "has marginal coverage at least 1-alpha."
            ),
            "pac_fixed_deployment": (
                "With probability at least 1-pac_beta over iid continuous "
                "calibration scores, the PAC rank threshold has population "
                "coverage at least 1-alpha."
            ),
            "scope": (
                "Bounds cover exactly four sampled isotropic perturbations "
                "under synchronous SDE noise; they are neither all-direction "
                "Lipschitz bounds nor semantic-correctness guarantees."
            ),
        },
        "aggregate_results": aggregate_rows,
        "identity_audit": {
            "n_split_rows": len(split_rows),
            "all_calibration_test_splits_disjoint": bool(
                overlaps and all(value == 0 for value in overlaps)
            ),
            "maximum_identity_overlap_count": (
                max(int(value) for value in overlaps if value is not None)
                if any(value is not None for value in overlaps)
                else None
            ),
        },
        "provenance": provenance,
        "magnet_bridge": {
            "keypoints_loaded": False,
            "asset_sha256": asset_sha256,
            "artifacts": paths,
        },
    }
    report_path = destination / "magnet_conformal_summary.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def summarize_two_task_evaluation(
    evaluation: Mapping[str, Any], *, diagnostic_family_alpha: float = 0.05
) -> Dict[str, Any]:
    """Extract the conformal ranks, bounds, coverage, and exact diagnostics."""

    task_summaries: Dict[str, Any] = {}
    selected_tasks = list(evaluation.get("selected_tasks", evaluation["tasks"]))
    if set(selected_tasks) != set(evaluation["tasks"]):
        raise ValueError("selected_tasks must match the evaluated task keys")
    cells = sum(len(report["aggregate_results"]) for report in evaluation["tasks"].values())
    if cells < 1:
        raise ValueError("conformal evaluation has no primary cells")
    diagnostic_threshold = float(diagnostic_family_alpha) / cells
    all_consistent = True
    for task in selected_tasks:
        report = evaluation["tasks"][task]
        directions: Dict[str, Any] = {}
        for row in report["aggregate_results"]:
            direction = str(row["direction"])
            exact_p = float(
                row["ordinary_exact_repeated_split_beta_binomial_lower_tail_pvalue"]
            )
            consistent = exact_p >= diagnostic_threshold
            all_consistent = all_consistent and consistent
            directions[direction] = {
                "ordinary_order_statistic": int(row["ordinary_order_statistic"]),
                "ordinary_expected_rank_coverage": float(
                    row["ordinary_marginal_expected_rank_coverage"]
                ),
                "ordinary_pooled_empirical_coverage": float(
                    row["ordinary_pooled_empirical_coverage"]
                ),
                "ordinary_amplification_threshold_median": float(
                    row["ordinary_amplification_threshold_median"]
                ),
                "ordinary_endpoint_bound_median": float(
                    row["ordinary_endpoint_aggregate_bound_median"]
                ),
                "ordinary_exact_repeated_beta_binomial_pvalue": exact_p,
                "ordinary_marginal_diagnostic_consistent": consistent,
                "pac_order_statistic": int(row["pac_order_statistic"]),
                "pac_beta_cdf_at_target": float(row["pac_beta_cdf_at_target"]),
                "pac_expected_rank_coverage": float(
                    row["pac_marginal_expected_rank_coverage"]
                ),
                "pac_pooled_empirical_coverage": float(
                    row["pac_pooled_empirical_coverage"]
                ),
                "pac_amplification_threshold_median": float(
                    row["pac_amplification_threshold_median"]
                ),
                "pac_endpoint_bound_median": float(
                    row["pac_endpoint_aggregate_bound_median"]
                ),
                "pac_exact_repeated_beta_binomial_pvalue": float(
                    row["pac_exact_repeated_split_beta_binomial_lower_tail_pvalue"]
                ),
            }
        task_summaries[task] = {
            "experiment": report["experiment"],
            "translation": report["translation"],
            "settings": report["settings"],
            "directions": directions,
            "all_calibration_test_splits_disjoint": bool(
                report["identity_audit"][
                    "all_calibration_test_splits_disjoint"
                ]
            ),
            "maximum_identity_overlap_count": report["identity_audit"][
                "maximum_identity_overlap_count"
            ],
            "keypoints_loaded": bool(
                report["magnet_bridge"]["keypoints_loaded"]
            ),
            "asset_sha256": report["magnet_bridge"]["asset_sha256"],
        }
    return {
        "schema_version": "fsbm-ffhq-conformal-magnet-summary-v1",
        "claim": (
            "Finite-sample conformal upper bounds on sampled local FFHQ latent "
            "transport sensitivity, not semantic correctness."
        ),
        "n_primary_cells": cells,
        "diagnostic_family_alpha": float(diagnostic_family_alpha),
        "bonferroni_lower_tail_threshold": diagnostic_threshold,
        "all_ordinary_marginal_diagnostics_consistent": all_consistent,
        "selected_tasks": selected_tasks,
        "tasks": task_summaries,
    }


def write_two_task_summary(
    evaluation: Mapping[str, Any], destination: str | Path
) -> Dict[str, Any]:
    summary = summarize_two_task_evaluation(evaluation)
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def run_two_task_evaluation(
    *,
    project_root: str | Path,
    checkpoints: Mapping[str, str | Path],
    data_dir: str | Path,
    output_dir: str | Path,
    seed: int = 20260716,
) -> Dict[str, Any]:
    """Run conformal audits for a non-empty requested task subset.

    The historical entry point name remains API-compatible with the original
    two-task card, while a task-specific card can omit all other dependencies.
    """
    root = find_project_root(project_root)
    base = Path(output_dir).expanduser()
    if not base.is_absolute():
        base = root / base
    base = base.resolve()
    checkpoint_tasks = set(checkpoints)
    unknown = sorted(checkpoint_tasks - set(TASKS))
    if unknown:
        raise KeyError(f"unknown FFHQ conformal tasks: {unknown}")
    selected_tasks = tuple(task for task in TASKS if task in checkpoint_tasks)
    if not selected_tasks:
        raise ValueError("at least one FFHQ conformal task is required")

    results = {}
    for task in selected_tasks:
        results[task] = run_task_evaluation(
            task=task,
            project_root=root,
            checkpoint=checkpoints[task],
            data_dir=data_dir,
            output_dir=base / task,
            seed=seed,
        )
    evaluation = {
        "schema_version": "fsbm-ffhq-conformal-magnet-v1",
        "seed": int(seed),
        "selected_tasks": list(selected_tasks),
        "tasks": results,
    }
    write_two_task_summary(evaluation, base / "conformal_summary.json")
    return evaluation


def run_two_task_evaluation_isolated(
    *,
    scientific_python: str | Path,
    project_root: str | Path,
    checkpoints: Mapping[str, str | Path],
    data_dir: str | Path,
    output_dir: str | Path,
    seed: int = 20260716,
) -> Dict[str, Any]:
    """Run the conformal audit in the frozen Python 3.10 environment."""

    root = find_project_root(project_root)
    python = _resolve_python_executable(scientific_python)
    current_python = Path(sys.executable).resolve(strict=True)
    arguments = {
        "project_root": str(root),
        "checkpoints": {task: str(path) for task, path in checkpoints.items()},
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "seed": int(seed),
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
        prefix=".magnet-conformal-scientific-", dir=base
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
                "magnet_eval.ffhq_conformal_bridge",
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
                "scientific conformal evaluation completed without a response: "
                f"{response_path}"
            )
        with response_path.open(encoding="utf-8") as handle:
            evaluation = json.load(handle)
    if not isinstance(evaluation, dict) or "tasks" not in evaluation:
        raise ValueError("scientific conformal evaluation returned an invalid response")
    return evaluation


def run_evaluation_isolated(**kwargs: Any) -> Dict[str, Any]:
    """Task-subset alias for the historical isolated conformal runner."""

    return run_two_task_evaluation_isolated(**kwargs)


def summarize_evaluation(
    evaluation: Mapping[str, Any], *, diagnostic_family_alpha: float = 0.05
) -> Dict[str, Any]:
    """Task-subset alias for the historical conformal summarizer."""

    return summarize_two_task_evaluation(
        evaluation, diagnostic_family_alpha=diagnostic_family_alpha
    )


def _run_json_request(request_path: Path, response_path: Path) -> None:
    with request_path.open(encoding="utf-8") as handle:
        request = json.load(handle)
    if not isinstance(request, dict):
        raise TypeError("scientific conformal request must be a JSON object")
    allowed = {"project_root", "checkpoints", "data_dir", "output_dir", "seed"}
    unknown = sorted(set(request) - allowed)
    required = allowed - {"seed"}
    missing = sorted(required - set(request))
    if unknown or missing:
        raise ValueError(
            f"invalid conformal request; missing={missing}, unknown={unknown}"
        )
    evaluation = run_two_task_evaluation(**request)
    response_path.parent.mkdir(parents=True, exist_ok=True)
    with response_path.open("w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run an FSBM FFHQ conformal evaluation JSON request."
    )
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--response-json", type=Path, required=True)
    arguments = parser.parse_args()
    _run_json_request(arguments.request_json, arguments.response_json)


if __name__ == "__main__":
    _main()
