import csv
from pathlib import Path

import pytest

from magnet_eval.lpips_comparison_bridge import (
    EXPECTED_SCORE_SHA256,
    conformal_upper_threshold,
    load_paired_lpips_comparison,
)


def test_augmented_conformal_rank():
    result = conformal_upper_threshold(list(range(100)), 0.05)
    assert result["order_statistic"] == 96
    assert result["threshold"] == 95.0


def test_pinned_reference_comparison_supports_scoped_generalization_claim():
    root = Path(__file__).resolve().parents[1]
    summary = load_paired_lpips_comparison(project_root=root)

    assert summary["baseline_method"] == "LSBM"
    assert summary["nominal_alpha_per_task"] == 0.05
    assert summary["family_adjusted_alpha_per_task"] == 0.025
    assert summary["all_nominal_paired_bounds_strictly_negative"] is True
    assert summary["all_family_adjusted_paired_bounds_strictly_negative"] is True
    assert (
        summary["supports_better_in_distribution_lpips_generalization_for_fsbm"]
        is True
    )

    gender = summary["tasks"]["gender"]
    age = summary["tasks"]["age"]
    assert gender["n_matched_identities"] == 100
    assert age["n_matched_identities"] == 100
    assert gender["fsbm"]["nominal_conformal_upper_bound"]["threshold"] == pytest.approx(
        0.43132367730140686
    )
    assert gender["lsbm"]["nominal_conformal_upper_bound"]["threshold"] == pytest.approx(
        0.6351377367973328
    )
    assert gender["paired_difference_fsbm_minus_lsbm"][
        "family_adjusted_conformal_upper_bound"
    ]["threshold"] == pytest.approx(-0.04947167634963989)
    assert age["paired_difference_fsbm_minus_lsbm"][
        "nominal_conformal_upper_bound"
    ]["threshold"] == pytest.approx(-0.08455556631088257)


def test_pinned_age_only_comparison_has_one_claim_family():
    root = Path(__file__).resolve().parents[1]
    summary = load_paired_lpips_comparison(
        project_root=root,
        tasks=("age",),
        expected_sha256={"age": EXPECTED_SCORE_SHA256["age"]},
    )
    assert summary["selected_tasks"] == ["age"]
    assert summary["n_family_claims"] == 1
    assert summary["family_adjusted_alpha_per_task"] == 0.05
    assert set(summary["tasks"]) == {"age"}
    assert summary["tasks"]["age"]["paired_difference_fsbm_minus_lsbm"][
        "family_adjusted_conformal_upper_bound"
    ]["order_statistic"] == 96


def test_score_hash_mismatch_fails_closed(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    files = {}
    for task in ("gender", "age"):
        path = tmp_path / f"{task}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "sample_id",
                    "lpips_fsbm",
                    "lpips_baseline",
                    "delta_fsbm_minus_baseline",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "sample_id": "0",
                    "lpips_fsbm": 0.1,
                    "lpips_baseline": 0.2,
                    "delta_fsbm_minus_baseline": -0.1,
                }
            )
        files[task] = path

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_paired_lpips_comparison(
            project_root=root,
            score_files=files,
            expected_sha256=EXPECTED_SCORE_SHA256,
        )
