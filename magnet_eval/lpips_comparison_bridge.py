"""Pinned paired-LPIPS comparison for the combined FFHQ MAGNET card.

This bridge deliberately recomputes all reported ranks from the per-identity
score CSVs.  It does not rerun the LPIPS network: the image pairs and the
scientific environment used for that computation are archived separately by
``lpips_conformal_compare.py``.  The SHA-256 checks make that distinction
explicit and prevent a card run from silently consuming different scores.
"""

from __future__ import annotations

import csv
import hashlib
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from magnet_eval.project import find_project_root


TASKS = ("gender", "age")

DEFAULT_SCORE_FILES = {
    "gender": (
        "reports/artifacts/lpips_conformal_fsbm_lsbm/"
        "gender_forward/per_image_lpips.csv"
    ),
    "age": (
        "reports/artifacts/lpips_conformal_fsbm_lsbm/"
        "age_forward/per_image_lpips.csv"
    ),
}

EXPECTED_SCORE_SHA256 = {
    "gender": "f4b5b707f1cb981a3470d82c3891a4ee6dd74c69626b64ac49cc5adb5cd19439",
    "age": "70bf9f64a2688cc1b90b936c2a3e47677e0e559a6b328ae2cf628c2b7f2f1c2a",
}

REQUIRED_COLUMNS = {
    "sample_id",
    "lpips_fsbm",
    "lpips_baseline",
    "delta_fsbm_minus_baseline",
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


def conformal_upper_threshold(
    scores: Sequence[float], alpha: float
) -> Dict[str, Any]:
    """Finite-sample upper conformal rank with the augmented infinity value."""
    values = [float(value) for value in scores]
    if not values:
        raise ValueError("scores must be non-empty")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("scores must all be finite")
    if not 0.0 <= float(alpha) < 1.0:
        raise ValueError("alpha must satisfy 0 <= alpha < 1")
    order = int(math.ceil((len(values) + 1) * (1.0 - float(alpha))))
    threshold = math.inf if order > len(values) else sorted(values)[order - 1]
    return {
        "alpha": float(alpha),
        "coverage": float(1.0 - alpha),
        "n_calibration": len(values),
        "order_statistic": order,
        "threshold": float(threshold),
        "finite_threshold": bool(math.isfinite(threshold)),
    }


def _read_scores(path: Path) -> Dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"LPIPS score file is missing columns {missing}: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"LPIPS score file is empty: {path}")

    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"LPIPS score identities are not unique: {path}")
    fsbm = [float(row["lpips_fsbm"]) for row in rows]
    baseline = [float(row["lpips_baseline"]) for row in rows]
    delta = [float(row["delta_fsbm_minus_baseline"]) for row in rows]
    if not all(math.isfinite(value) for value in fsbm + baseline + delta):
        raise ValueError(f"LPIPS score file contains a non-finite value: {path}")
    for fsbm_value, baseline_value, delta_value in zip(fsbm, baseline, delta):
        if not math.isclose(
            fsbm_value - baseline_value,
            delta_value,
            rel_tol=0.0,
            abs_tol=1e-7,
        ):
            raise ValueError(f"paired LPIPS delta is internally inconsistent: {path}")
    return {
        "sample_ids": sample_ids,
        "fsbm": fsbm,
        "baseline": baseline,
        "delta": delta,
    }


def _metric_summary(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def load_paired_lpips_comparison(
    *,
    project_root: str | Path,
    score_files: Mapping[str, str | Path] | None = None,
    expected_sha256: Mapping[str, str] | None = None,
    tasks: Sequence[str] | None = None,
    nominal_alpha: float = 0.05,
    family_alpha: float = 0.05,
) -> Dict[str, Any]:
    """Verify score artifacts and recompute task-wise comparative bounds."""
    root = find_project_root(project_root)
    selected_tasks = tuple(TASKS if tasks is None else tasks)
    if not selected_tasks:
        raise ValueError("at least one LPIPS task is required")
    if len(selected_tasks) != len(set(selected_tasks)):
        raise ValueError("LPIPS tasks must be unique")
    unknown = sorted(set(selected_tasks) - set(TASKS))
    if unknown:
        raise ValueError(f"unknown LPIPS tasks: {unknown}")
    source_files = DEFAULT_SCORE_FILES if score_files is None else score_files
    source_hashes = EXPECTED_SCORE_SHA256 if expected_sha256 is None else expected_sha256
    files = {task: source_files[task] for task in selected_tasks if task in source_files}
    hashes = {task: source_hashes[task] for task in selected_tasks if task in source_hashes}
    if set(files) != set(selected_tasks) or set(hashes) != set(selected_tasks):
        raise ValueError(
            "score files and hashes must cover exactly the selected tasks "
            f"{list(selected_tasks)}"
        )
    if not 0.0 < float(family_alpha) < 1.0:
        raise ValueError("family_alpha must satisfy 0 < family_alpha < 1")
    per_task_family_alpha = float(family_alpha) / len(selected_tasks)

    task_summaries: Dict[str, Any] = {}
    for task in selected_tasks:
        path = _resolve_file(root, files[task], f"{task} LPIPS score artifact")
        actual_hash = _sha256(path)
        if actual_hash != hashes[task]:
            raise ValueError(
                f"{task} LPIPS score SHA-256 mismatch: expected "
                f"{hashes[task]}, got {actual_hash} ({path})"
            )
        scores = _read_scores(path)
        fsbm_bound = conformal_upper_threshold(scores["fsbm"], nominal_alpha)
        baseline_bound = conformal_upper_threshold(
            scores["baseline"], nominal_alpha
        )
        paired_bound = conformal_upper_threshold(scores["delta"], nominal_alpha)
        family_paired_bound = conformal_upper_threshold(
            scores["delta"], per_task_family_alpha
        )
        fsbm_threshold = float(fsbm_bound["threshold"])
        baseline_threshold = float(baseline_bound["threshold"])
        task_summaries[task] = {
            "n_matched_identities": len(scores["sample_ids"]),
            "fsbm": {
                **_metric_summary(scores["fsbm"]),
                "nominal_conformal_upper_bound": fsbm_bound,
            },
            "lsbm": {
                **_metric_summary(scores["baseline"]),
                "nominal_conformal_upper_bound": baseline_bound,
            },
            "paired_difference_fsbm_minus_lsbm": {
                **_metric_summary(scores["delta"]),
                "nominal_conformal_upper_bound": paired_bound,
                "family_adjusted_conformal_upper_bound": family_paired_bound,
                "fsbm_lower_count": sum(value < 0.0 for value in scores["delta"]),
                "fsbm_lower_fraction": statistics.fmean(
                    value < 0.0 for value in scores["delta"]
                ),
            },
            "fsbm_nominal_bound_lower_than_lsbm": bool(
                fsbm_threshold < baseline_threshold
            ),
            "fsbm_nominal_bound_relative_reduction": float(
                1.0 - fsbm_threshold / baseline_threshold
            ),
            "nominal_paired_bound_strictly_negative": bool(
                float(paired_bound["threshold"]) < 0.0
            ),
            "family_adjusted_paired_bound_strictly_negative": bool(
                float(family_paired_bound["threshold"]) < 0.0
            ),
            "provenance": {
                "score_file": str(path),
                "score_file_sha256": actual_hash,
            },
        }

    all_nominal = all(
        result["nominal_paired_bound_strictly_negative"]
        for result in task_summaries.values()
    )
    all_family_adjusted = all(
        result["family_adjusted_paired_bound_strictly_negative"]
        for result in task_summaries.values()
    )
    return {
        "schema_version": "fsbm-lsbm-paired-lpips-magnet-summary-v1",
        "metric": {
            "name": "LPIPS",
            "network": "AlexNet",
            "version": "0.1",
            "evaluation_resolution": [256, 256],
            "reference": "same source image for each matched method output",
            "direction": "lower is less perceptual deviation from source",
        },
        "primary_method": "FSBM",
        "baseline_method": "LSBM",
        "nominal_alpha_per_task": float(nominal_alpha),
        "selected_tasks": list(selected_tasks),
        "family_alpha": float(family_alpha),
        "n_family_claims": len(selected_tasks),
        # Backward-compatible name used by archived two-task summaries.
        "two_task_family_alpha": float(family_alpha),
        "family_adjusted_alpha_per_task": per_task_family_alpha,
        "all_nominal_paired_bounds_strictly_negative": all_nominal,
        "all_family_adjusted_paired_bounds_strictly_negative": all_family_adjusted,
        "supports_better_in_distribution_lpips_generalization_for_fsbm": bool(
            all_family_adjusted
        ),
        "claim": (
            "On the declared source-referenced LPIPS endpoint, FSBM has a "
            "tighter conformal upper bound and a strictly negative paired "
            "FSBM-minus-LSBM bound on every selected in-distribution FFHQ task."
        ),
        "scope": (
            "Finite-sample marginal prediction for a new exchangeable identity "
            "under the archived generation procedure; not conditional, shifted-"
            "distribution, semantic-correctness, identity, or perceptual-quality "
            "generalization. All matched identities calibrate the bounds, so no "
            "separate held-out LPIPS coverage audit is claimed."
        ),
        "tasks": task_summaries,
    }
