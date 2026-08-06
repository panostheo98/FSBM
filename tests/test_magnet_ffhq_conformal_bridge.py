import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from magnet_eval.ffhq_conformal_bridge import (
    run_two_task_evaluation,
    run_two_task_evaluation_isolated,
    summarize_two_task_evaluation,
    write_two_task_summary,
)


def _aggregate(direction, *, coverage, bound, exact_p):
    threshold = bound / 0.1
    return {
        "direction": direction,
        "ordinary_order_statistic": 951,
        "ordinary_marginal_expected_rank_coverage": 951 / 1001,
        "ordinary_pooled_empirical_coverage": coverage,
        "ordinary_amplification_threshold_median": threshold,
        "ordinary_endpoint_aggregate_bound_median": bound,
        "ordinary_exact_repeated_split_beta_binomial_lower_tail_pvalue": exact_p,
        "pac_order_statistic": 962,
        "pac_beta_cdf_at_target": 0.043348190606897485,
        "pac_marginal_expected_rank_coverage": 962 / 1001,
        "pac_pooled_empirical_coverage": min(1.0, coverage + 0.012),
        "pac_amplification_threshold_median": threshold + 0.03,
        "pac_endpoint_aggregate_bound_median": bound + 0.003,
        "pac_exact_repeated_split_beta_binomial_lower_tail_pvalue": 0.25,
    }


def _task_report(task):
    gender = task == "gender"
    return {
        "experiment": "ffhq_gen" if gender else "ffhq_age",
        "translation": "female->male" if gender else "adult->child",
        "settings": {
            "directions": ["fwd", "bwd"],
            "deltas": [0.1],
            "alphas": [0.05],
            "n_splits": 5,
            "n_calib": 1000,
            "n_test": 1000 if gender else 600,
            "n_probes": 4,
            "aggregation": "max",
            "nfe": 50,
            "pac_beta": 0.05,
        },
        "aggregate_results": [
            _aggregate("fwd", coverage=0.9568, bound=0.1181, exact_p=0.94),
            _aggregate("bwd", coverage=0.9450, bound=0.1437, exact_p=0.13),
        ],
        "identity_audit": {
            "all_calibration_test_splits_disjoint": True,
            "maximum_identity_overlap_count": 0,
        },
        "magnet_bridge": {
            "keypoints_loaded": False,
            "asset_sha256": {
                "evaluator": "evaluator-hash",
                "checkpoint": f"{task}-checkpoint-hash",
                "hydra_config": f"{task}-hydra-hash",
                "latents": "latent-hash",
                "labels": f"{task}-label-hash",
            },
        },
    }


def _evaluation():
    return {
        "tasks": {
            "gender": _task_report("gender"),
            "age": _task_report("age"),
        }
    }


def test_summary_preserves_declared_order_after_sorted_json_round_trip():
    evaluation = _evaluation()
    evaluation["selected_tasks"] = ["gender", "age"]
    sorted_round_trip = json.loads(json.dumps(evaluation, sort_keys=True))
    summary = summarize_two_task_evaluation(sorted_round_trip)
    assert summary["selected_tasks"] == ["gender", "age"]
    assert list(summary["tasks"]) == ["gender", "age"]


def test_summary_exposes_conformal_rank_bound_coverage_and_pac_fields():
    summary = summarize_two_task_evaluation(_evaluation())
    assert summary["n_primary_cells"] == 4
    assert summary["bonferroni_lower_tail_threshold"] == 0.0125
    assert summary["all_ordinary_marginal_diagnostics_consistent"] is True
    gender = summary["tasks"]["gender"]
    assert gender["all_calibration_test_splits_disjoint"] is True
    assert gender["directions"]["fwd"]["ordinary_order_statistic"] == 951
    assert gender["directions"]["fwd"]["ordinary_endpoint_bound_median"] == 0.1181
    assert gender["directions"]["bwd"]["ordinary_pooled_empirical_coverage"] == 0.945
    assert gender["directions"]["fwd"]["pac_order_statistic"] == 962


def test_summary_flags_only_family_adjusted_exact_marginal_warning():
    evaluation = _evaluation()
    evaluation["tasks"]["gender"]["aggregate_results"][0][
        "ordinary_exact_repeated_split_beta_binomial_lower_tail_pvalue"
    ] = 0.01
    summary = summarize_two_task_evaluation(evaluation)
    assert summary["all_ordinary_marginal_diagnostics_consistent"] is False
    assert (
        summary["tasks"]["gender"]["directions"]["fwd"]
        ["ordinary_marginal_diagnostic_consistent"]
        is False
    )


def test_conformal_summary_can_be_persisted():
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / "summary.json"
        summary = write_two_task_summary(_evaluation(), destination)
        assert destination.is_file()
        assert '"ordinary_order_statistic": 951' in destination.read_text()
        assert summary["tasks"]["age"]["settings"]["n_test"] == 600


def test_age_only_summary_uses_two_direction_family_adjustment():
    summary = summarize_two_task_evaluation(
        {"tasks": {"age": _task_report("age")}}
    )
    assert summary["selected_tasks"] == ["age"]
    assert summary["n_primary_cells"] == 2
    assert summary["bonferroni_lower_tail_threshold"] == 0.025
    assert set(summary["tasks"]) == {"age"}


def test_runner_executes_only_the_requested_age_task():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as directory, patch(
        "magnet_eval.ffhq_conformal_bridge.run_task_evaluation",
        return_value={"task": "age"},
    ) as run_task, patch(
        "magnet_eval.ffhq_conformal_bridge.write_two_task_summary"
    ):
        result = run_two_task_evaluation(
            project_root=root,
            checkpoints={"age": "age.ckpt"},
            data_dir="data",
            output_dir=directory,
        )
    assert result["selected_tasks"] == ["age"]
    assert set(result["tasks"]) == {"age"}
    run_task.assert_called_once()
    assert run_task.call_args.kwargs["task"] == "age"


def _isolated_arguments(root, output):
    return {
        "scientific_python": sys.executable,
        "project_root": root,
        "checkpoints": {"gender": "gender.ckpt", "age": "age.ckpt"},
        "data_dir": "data",
        "output_dir": output,
        "seed": 23,
    }


def test_isolated_conformal_runner_calls_directly_for_current_python():
    root = Path(__file__).resolve().parents[1]
    expected = {"tasks": {"gender": {}, "age": {}}}
    with tempfile.TemporaryDirectory() as directory:
        arguments = _isolated_arguments(root, directory)
        previous_dont_write_bytecode = sys.dont_write_bytecode
        with patch(
            "magnet_eval.ffhq_conformal_bridge.run_two_task_evaluation",
            return_value=expected,
        ) as direct:
            assert run_two_task_evaluation_isolated(**arguments) == expected
        assert sys.dont_write_bytecode is previous_dont_write_bytecode
        direct.assert_called_once()
        assert direct.call_args.kwargs["seed"] == 23


def test_isolated_conformal_runner_transfers_json_to_external_python():
    root = Path(__file__).resolve().parents[1]
    expected = {"tasks": {"gender": {"ok": True}, "age": {"ok": True}}}
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        fake_python = temporary / "scientific-python"
        fake_python.write_text("#!/bin/sh\n", encoding="utf-8")
        fake_python.chmod(0o755)
        arguments = _isolated_arguments(root, temporary / "output")
        arguments["scientific_python"] = fake_python

        def complete(command, **kwargs):
            request_path = Path(command[command.index("--request-json") + 1])
            response_path = Path(command[command.index("--response-json") + 1])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            assert request["seed"] == 23
            assert kwargs["cwd"] == root
            assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
            response_path.write_text(json.dumps(expected), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0)

        with patch(
            "magnet_eval.ffhq_conformal_bridge.subprocess.run",
            side_effect=complete,
        ) as external:
            assert run_two_task_evaluation_isolated(**arguments) == expected
        external.assert_called_once()
        assert external.call_args.args[0][0] == str(fake_python.resolve())


def test_clean_image_card_contains_only_conformal_and_lpips_claims():
    root = Path(__file__).resolve().parents[1]
    card = root / "magnet_eval/cards/fsbm_ffhq_conformal_stability_lpips.yaml"
    text = card.read_text(encoding="utf-8")
    assert "PRIMARY CONFORMAL CLAIM" in text
    assert "FINITE-SAMPLE CONFORMAL TRANSPORT-STABILITY BOUNDS" in text
    assert "PAIRED IN-DISTRIBUTION LPIPS GENERALIZATION COMPARISON" in text
    assert "B=delta*S_hat" in text
    assert "conformal_expected_rank" in text
    assert "conformal_expected_pac_rank" in text
    assert "ordinary_exact_repeated_beta_binomial_pvalue" in text
    assert "Semantic accuracy" in text
    assert "conformal_summary['n_primary_cells'] == 4" in text
    assert "FSBM_FFHQ_GENDER_CHECKPOINT" in text
    assert "FSBM_FFHQ_AGE_CHECKPOINT" in text
    assert "image_semantic_bridge" not in text
    assert "semantic_result" not in text
    assert "Clopper" not in text
    assert "admission" not in text.lower()
    assert "selective" not in text.lower()
    assert "PLAIN-LANGUAGE TASK INTUITION" in text
    assert "Strong local stability can coexist" in text
    assert "limited change in the requested target attribute" in text
    assert "an edit that is too subtle" in text
    assert "Assess requested-edit success with a dedicated semantic evaluation" in text
