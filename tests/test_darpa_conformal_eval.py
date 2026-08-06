import csv
import itertools
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import darpa_conformal_eval as evaluation


class _Dataset:
    def __init__(self, size=64, dim=2):
        self.x_data = torch.arange(size * dim, dtype=torch.float32).reshape(size, dim)


class _DatasetSampler:
    def __init__(self, size=64, dim=2):
        self.dataset = _Dataset(size=size, dim=dim)

    def __call__(self, n):
        raise AssertionError("dataset mode must index x_data, not sample with replacement")


class _GaussianSampler:
    def __init__(self, dim=2, generator=None):
        self.mean = torch.zeros(dim)
        self.std = torch.ones(dim)
        self.generator = generator

    def set_generator(self, generator):
        self.generator = generator

    def __call__(self, n):
        return torch.randn(n, self.mean.numel(), generator=self.generator)


class _Model:
    sigma = 0.5

    def __init__(self, sampler):
        self.sampler = sampler

    def eval(self):
        return self

    def get_contraction_reference(self, direction):
        return self.sampler

    def build_drift(self, direction):
        return lambda x, t: torch.zeros_like(x)


class DistributionTests(unittest.TestCase):
    def test_pac_rank_matches_requested_reference_case(self):
        result = evaluation.pac_order_statistic(1000, alpha=0.05, beta=0.05)
        self.assertEqual(result["order_statistic"], 962)
        self.assertLessEqual(result["beta_cdf_at_target"], 0.05)
        previous = evaluation.beta_cdf_integer_shape(0.95, 961, 40)
        self.assertGreater(previous, 0.05)

    def test_beta_binomial_matches_exchangeable_rank_enumeration(self):
        # Every choice of n calibration ranks among n+m pooled continuous scores
        # is equally likely.  Count held-out ranks below the k-th calibration rank.
        n_calib, n_test, rank = 4, 3, 3
        counts = []
        for calibration_ranks in itertools.combinations(
            range(1, n_calib + n_test + 1), n_calib
        ):
            threshold_rank = calibration_ranks[rank - 1]
            counts.append(threshold_rank - rank)
        for observed in range(n_test + 1):
            brute = sum(value <= observed for value in counts) / len(counts)
            exact = evaluation.beta_binomial_lower_tail_cdf(
                observed,
                n_test,
                rank,
                n_calib + 1 - rank,
            )
            self.assertAlmostEqual(exact, brute, places=12)

    def test_repeated_beta_binomial_tail_is_exact_convolution(self):
        trials, splits, a, b = 3, 2, 2, 3
        single = [
            evaluation.beta_binomial_pmf(k, trials, a, b)
            for k in range(trials + 1)
        ]
        brute = np.convolve(single, single)
        for observed in range(trials * splits + 1):
            exact = evaluation.repeated_beta_binomial_lower_tail_cdf(
                observed, splits, trials, a, b
            )
            self.assertAlmostEqual(exact, float(brute[: observed + 1].sum()), places=12)

    def test_fixed_threshold_binomial_diagnostic_matches_direct_sum(self):
        observed, trials, probability = 3, 7, 0.8
        direct = sum(
            math.comb(trials, k)
            * probability**k
            * (1 - probability) ** (trials - k)
            for k in range(observed + 1)
        )
        self.assertAlmostEqual(
            evaluation.binomial_lower_tail_cdf(observed, trials, probability),
            direct,
            places=13,
        )


class SamplingTests(unittest.TestCase):
    def test_dataset_split_is_disjoint_without_replacement_and_reproducible(self):
        sampler = _DatasetSampler(size=40)
        first = evaluation.draw_query_split(
            sampler, 12, 15, seed=17, direction="fwd", split=2
        )
        second = evaluation.draw_query_split(
            sampler, 12, 15, seed=17, direction="fwd", split=2
        )
        self.assertTrue(torch.equal(first.calibration, second.calibration))
        self.assertTrue(torch.equal(first.test, second.test))
        self.assertEqual(
            first.metadata["sampling_mode"],
            "dataset_disjoint_without_replacement",
        )
        self.assertEqual(first.metadata["identity_overlap_count"], 0)
        calibration_ids = set(first.calibration[:, 0].tolist())
        test_ids = set(first.test[:, 0].tolist())
        self.assertFalse(calibration_ids.intersection(test_ids))
        with self.assertRaisesRegex(ValueError, "exceeds validation dataset size"):
            evaluation.draw_query_split(
                sampler, 30, 20, seed=0, direction="fwd", split=0
            )

    def test_gaussian_draws_use_distinct_reproducible_seeds(self):
        sampler = _GaussianSampler()
        first = evaluation.draw_query_split(
            sampler, 20, 20, seed=9, direction="bwd", split=1
        )
        second = evaluation.draw_query_split(
            sampler, 20, 20, seed=9, direction="bwd", split=1
        )
        self.assertTrue(torch.equal(first.calibration, second.calibration))
        self.assertTrue(torch.equal(first.test, second.test))
        self.assertNotEqual(
            first.metadata["calibration_draw_seed"],
            first.metadata["test_draw_seed"],
        )
        self.assertFalse(torch.equal(first.calibration, first.test))
        self.assertEqual(
            first.metadata["sampling_mode"], "independent_seeded_sampler_draws"
        )


class EvaluationTests(unittest.TestCase):
    @staticmethod
    def _fake_local_amplification(drift, queries, **kwargs):
        # Distinct deterministic scores.  The offsets make it possible to verify
        # that the requested runtime aggregation, rather than a fallback score,
        # was calibrated.
        base = queries[:, 0].float() / 100.0
        n_probes = kwargs["n_probes"]
        probe_scores = base[:, None] + torch.arange(n_probes).float()[None, :]
        return {
            "mean": probe_scores.mean(dim=1),
            "max": probe_scores.max(dim=1).values,
            "median": probe_scores.median(dim=1).values,
            "std": probe_scores.std(dim=1),
            "probe_scores": probe_scores,
        }

    def test_runtime_score_configuration_and_both_claims_are_reported(self):
        settings = evaluation.EvaluationSettings(
            directions=("fwd",),
            deltas=(0.1, 0.2),
            alphas=(0.2,),
            n_splits=2,
            n_calib=20,
            n_test=10,
            n_probes=3,
            aggregation="max",
            nfe=7,
            batch_size=11,
            seed=123,
            device="cpu",
            pac_beta=0.2,
        )
        calls = []

        def recorder(*args, **kwargs):
            calls.append(kwargs)
            return self._fake_local_amplification(*args, **kwargs)

        with mock.patch.object(evaluation, "local_amplification", side_effect=recorder):
            rows, aggregates, provenance = evaluation.evaluate_model(
                _Model(_DatasetSampler(size=80)), settings, verbose=False
            )

        # One score computation per direction/split/delta; alphas reuse it.
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(call["n_probes"] == 3 for call in calls))
        self.assertTrue(all(call["nfe"] == 7 for call in calls))
        self.assertTrue(all(call["batch_size"] == 11 for call in calls))
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(aggregates), 2)
        self.assertTrue(all(row["aggregation"] == "max" for row in rows))
        self.assertTrue(all(row["ordinary_order_statistic"] == 17 for row in rows))
        self.assertTrue(all(row["pac_order_statistic"] == 19 for row in rows))
        self.assertTrue(
            all(
                row["pac_amplification_threshold"]
                >= row["ordinary_amplification_threshold"]
                for row in rows
            )
        )
        self.assertIn("exact_beta_binomial", "_".join(rows[0].keys()))
        self.assertEqual(
            provenance["fwd"]["sampling_mode"],
            "dataset_disjoint_without_replacement",
        )

    def test_outputs_include_csv_json_claim_language_and_hashes(self):
        settings = evaluation.EvaluationSettings(
            directions=("fwd",),
            deltas=(0.1,),
            alphas=(0.2,),
            n_splits=1,
            n_calib=5,
            n_test=4,
            n_probes=2,
            aggregation="mean",
            nfe=3,
            batch_size=16,
            seed=4,
            device="cpu",
        )
        with mock.patch.object(
            evaluation,
            "local_amplification",
            side_effect=self._fake_local_amplification,
        ):
            rows, aggregates, data = evaluation.evaluate_model(
                _Model(_GaussianSampler()), settings, verbose=False
            )
        manifest = {
            "checkpoint": {"sha256": "a" * 64},
            "hydra_config": {"resolved_sha256": "b" * 64},
            "data": data,
            "run_config_sha256": rows[0]["run_config_sha256"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            paths = evaluation.write_outputs(
                Path(tmp), rows, aggregates, manifest, figures=False
            )
            for path in paths.values():
                self.assertTrue(Path(path).is_file())
            with Path(paths["per_split_csv"]).open(newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 1)
            self.assertIn(
                "ordinary_exact_beta_binomial_marginal_lower_tail_pvalue",
                csv_rows[0],
            )
            with Path(paths["aggregate_json"]).open() as handle:
                payload = json.load(handle)
            ordinary_claim = payload["claims"]["ordinary_marginal_split_conformal"]
            self.assertIn("not that every realized split", ordinary_claim)
            diagnostic = payload["statistical_diagnostics"][
                "fixed_threshold_binomial_diagnostic"
            ]
            self.assertIn("not the conformal validity test", diagnostic)
            self.assertEqual(
                payload["provenance"]["checkpoint"]["sha256"], "a" * 64
            )


if __name__ == "__main__":
    unittest.main()
