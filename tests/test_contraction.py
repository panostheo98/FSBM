import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import torch
from omegaconf import OmegaConf

import contraction_eval
from fsbm.contraction import (
    DEFAULT_CONTRACTION_CFG,
    build_terminal_certificate,
    compute_timewise_w2_bound,
    compute_local_jacobian_diagnostic,
    conformal_quantile,
    evaluate_directional_contraction,
    pac_tolerance_quantile,
)
from fsbm.sde import build_g_guidance_potential, paired_sdeint


class GaussianSampler:
    def __init__(self, dim):
        self.dim = dim
        self.generator = None

    def set_generator(self, generator):
        self.generator = generator

    def __call__(self, n):
        return torch.randn(n, self.dim, generator=self.generator)


def linear_drift(scale):
    def drift(x, t):
        return scale * x

    return drift


class DummyModel:
    def __init__(self, results):
        self.results = results
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def evaluate_contraction_suite(self, contraction_cfg):
        return self.results


class ContractionTests(unittest.TestCase):
    def test_paired_sdeint_zero_drift_preserves_distance(self):
        x0 = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
        y0 = x0 + torch.tensor([[0.1, -0.2], [0.3, 0.4]])
        output = paired_sdeint(
            x0,
            y0,
            drift=lambda x, t: torch.zeros_like(x),
            diffusion=lambda x, t: 0.75,
            direction="fwd",
            nfe=32,
            log_steps=8,
            generator=torch.Generator().manual_seed(0),
        )

        initial = output["initial_distance"]
        terminal = output["terminal_distance"]
        self.assertTrue(torch.allclose(initial, terminal, atol=1e-5, rtol=1e-5))
        logged = output["distances"]
        self.assertTrue(torch.allclose(logged, initial[:, None], atol=1e-5, rtol=1e-5))

    def test_conformal_quantile_matches_order_statistic(self):
        scores = torch.tensor([0.4, 0.1, 0.9, 0.2, 0.5])
        result = conformal_quantile(scores, alpha=0.2)
        self.assertEqual(result["order_statistic"], 5)
        self.assertAlmostEqual(result["quantile"], 0.9)
        self.assertTrue(result["finite_threshold"])

    def test_conformal_quantile_uses_infinity_when_rank_is_unattainable(self):
        scores = torch.tensor([0.1, 0.2, 0.3, 0.4])
        result = conformal_quantile(scores, alpha=0.0)
        self.assertEqual(result["order_statistic"], 5)
        self.assertEqual(result["quantile"], float("inf"))
        self.assertFalse(result["finite_threshold"])

    def test_pac_tolerance_quantile_uses_more_conservative_rank(self):
        scores = torch.linspace(0.0, 1.0, 1000)
        marginal = conformal_quantile(scores, alpha=0.05)
        tolerance = pac_tolerance_quantile(scores, alpha=0.05, beta=0.05)
        self.assertEqual(marginal["order_statistic"], 951)
        self.assertEqual(tolerance["order_statistic"], 962)
        self.assertGreater(tolerance["quantile"], marginal["quantile"])
        self.assertAlmostEqual(tolerance["calibration_confidence"], 0.95)

    def test_pac_tolerance_quantile_returns_infinity_when_sample_too_small(self):
        scores = torch.arange(10, dtype=torch.float32)
        tolerance = pac_tolerance_quantile(scores, alpha=0.05, beta=0.05)
        self.assertEqual(tolerance["order_statistic"], 11)
        self.assertEqual(tolerance["quantile"], float("inf"))

    def test_terminal_certificate_builds_expected_bound(self):
        certificate = build_terminal_certificate(
            delta0=0.2,
            alpha=0.05,
            n_calib=128,
            conformal={"quantile": 1.5, "order_statistic": 123},
            baseline_terminal_error=0.01,
            tau=0.4,
        )
        self.assertAlmostEqual(certificate["terminal_distance_bound"], 0.3)
        self.assertAlmostEqual(certificate["predicted_total_error_bound"], 0.31)
        self.assertAlmostEqual(certificate["max_certified_delta0_for_tau"], 0.26)
        self.assertTrue(certificate["within_tau"])

    def test_paired_sdeint_backward_integration_fields_are_consistent(self):
        x0 = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
        y0 = x0 + torch.tensor([[0.1, -0.2], [0.3, 0.4]])
        output = paired_sdeint(
            x0,
            y0,
            drift=lambda x, t: torch.zeros_like(x),
            diffusion=lambda x, t: 0.75,
            direction="bwd",
            nfe=16,
            log_steps=5,
            generator=torch.Generator().manual_seed(0),
        )

        self.assertTrue(
            torch.allclose(
                output["distances_integration"][:, 0],
                output["initial_distance"],
                atol=1e-5,
                rtol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                output["distances_integration"][:, -1],
                output["terminal_distance"],
                atol=1e-5,
                rtol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                output["distances_integration"],
                output["distances"].flip(1),
                atol=1e-5,
                rtol=1e-5,
            )
        )
        self.assertTrue((output["t_integration"][1:] < output["t_integration"][:-1]).all())

    def test_g_guidance_potential_is_zero_at_anchor_time(self):
        xt_ks = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 1.0]],
                [[2.0, 2.0], [3.0, 3.0]],
            ]
        )
        anchors = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
        potential = build_g_guidance_potential(
            xt_ks, anchors, tau=1.0, direction="fwd"
        )
        t = torch.zeros(anchors.shape[0])
        out = potential(anchors, t)
        self.assertTrue(torch.allclose(out, torch.zeros_like(out), atol=1e-6, rtol=1e-6))

    def test_linear_contractive_empirical_amplification(self):
        nfe = 50
        rate = 0.5
        expected = (1.0 - rate / nfe) ** nfe
        sampler = GaussianSampler(dim=2)
        result = evaluate_directional_contraction(
            drift=linear_drift(-rate),
            reference_sampler=sampler,
            direction="fwd",
            sigma=1.0,
            n_calib=16,
            alpha=0.1,
            delta0s=[0.2],
            nfe=nfe,
            log_steps=4,
            seed=7,
            device="cpu",
        )

        s_hat_alpha = result["per_delta"][0]["conformal"]["quantile"]
        self.assertAlmostEqual(s_hat_alpha, expected, places=4)

    def test_timewise_w2_bound_holds_for_linear_contractive_drift(self):
        rate = 0.4
        sampler = GaussianSampler(dim=2)
        result = evaluate_directional_contraction(
            drift=linear_drift(-rate),
            reference_sampler=sampler,
            direction="fwd",
            sigma=1.0,
            n_calib=8,
            alpha=0.1,
            delta0s=[0.1],
            nfe=24,
            log_steps=6,
            seed=5,
            device="cpu",
            w2_bound_cfg={"enabled": True, "num_states": 4, "compute_ot": True},
        )

        bound = result["per_delta"][0]["w2_bound"]
        self.assertTrue(bound["all_coupled_w2_sq_upper_holds"])
        self.assertTrue(bound["all_empirical_w2_sq_holds"])
        self.assertEqual(len(bound["physical_times"]), 6)
        self.assertAlmostEqual(bound["local_L_sup"][0], -2.0 * rate, places=5)
        self.assertAlmostEqual(
            bound["lambda_max_sym_quantile_level"],
            0.9,
            places=7,
        )
        quant_env = bound["envelopes"]["sampled_quantile"]
        self.assertTrue(quant_env["all_coupled_holds"])
        self.assertTrue(quant_env["all_empirical_holds"])
        self.assertAlmostEqual(quant_env["local_L"][0], -2.0 * rate, places=5)
        certificate = result["per_delta"][0]["terminal_certificate"]
        self.assertAlmostEqual(
            certificate["terminal_distance_bound"],
            certificate["s_hat_alpha"] * 0.1,
            places=6,
        )

    def test_timewise_w2_bound_uses_custom_local_matrix(self):
        xs = torch.tensor(
            [
                [[1.0, 0.0], [0.8, 0.0], [0.6, 0.0]],
                [[0.0, 1.0], [0.0, 0.8], [0.0, 0.6]],
            ]
        )
        ys = xs + 0.05
        ts = torch.tensor([0.0, 0.5, 1.0])
        custom_matrix = torch.tensor([[-0.4, 0.0], [0.0, -0.8]])

        def matrix_fn(state_idx, x, t_scalar):
            return custom_matrix.to(x)

        result = compute_timewise_w2_bound(
            drift_x=linear_drift(0.0),
            drift_y=linear_drift(0.0),
            xs=xs,
            ys=ys,
            ts=ts,
            num_states=2,
            compute_ot=False,
            local_matrix_x=matrix_fn,
            local_matrix_y=matrix_fn,
            local_operator_description="custom operator",
        )

        self.assertAlmostEqual(result["local_L_sup"][0], -0.8, places=5)
        self.assertEqual(result["local_operator_description"], "custom operator")

    def test_directional_smoke_for_forward_and_backward(self):
        sampler = GaussianSampler(dim=3)
        for direction in ("fwd", "bwd"):
            result = evaluate_directional_contraction(
                drift=linear_drift(-0.2),
                reference_sampler=sampler,
                direction=direction,
                sigma=0.5,
                n_calib=8,
                alpha=0.2,
                delta0s=[0.05, 0.1],
                nfe=20,
                log_steps=5,
                seed=3,
                device="cpu",
            )
            self.assertEqual(result["direction"], direction)
            self.assertEqual(len(result["per_delta"]), 2)

    def test_local_jacobian_diagnostic(self):
        matrix = torch.tensor([[-1.0, 0.0], [0.0, -2.0]])

        def drift(x, t):
            return x @ matrix.t()

        xs = torch.randn(4, 3, 2)
        ts = torch.tensor([0.0, 0.5, 1.0])
        result = compute_local_jacobian_diagnostic(
            drift=drift,
            xs=xs,
            ts=ts,
            num_states=2,
            num_times=2,
        )
        self.assertAlmostEqual(result["mean_lambda_max_sym"], -1.0, places=5)
        self.assertAlmostEqual(result["mean_local_L"], -2.0, places=5)
        self.assertEqual(len(result["times"]), 2)

    def test_end_to_end_script_smoke(self):
        results = {
            "fwd": {
                "direction": "fwd",
                "per_delta": [
                    {
                        "delta0": 0.1,
                        "conformal": {"quantile": 0.8, "order_statistic": 3},
                    }
                ],
            }
        }
        cfg = OmegaConf.create({"name": "toy", "contraction": DEFAULT_CONTRACTION_CFG})
        model = DummyModel(results)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            args = Namespace(
                ckpt_file=Path(tmpdir) / "model.ckpt",
                device="cpu",
                output=str(output_path),
                tau=None,
                directions=None,
                n_calib=None,
                alpha=None,
                delta0s=None,
                nfe=None,
                log_steps=None,
                seed=None,
                save_scores=False,
                save_terminal_distances=False,
                save_histograms=False,
                hist_bins=None,
                verbose=False,
                jacobian=False,
                jacobian_num_states=None,
                jacobian_num_times=None,
                w2_bound=False,
                w2_bound_num_states=None,
                w2_bound_quantile_level=None,
                no_w2_bound_ot=False,
                baseline_terminal_error=None,
                tolerance_tau=None,
            )

            with mock.patch.object(contraction_eval, "restore_model", return_value=(model, cfg)):
                contraction_eval.main(args)

            self.assertTrue(model.eval_called)
            self.assertTrue(output_path.exists())
            with open(output_path) as fin:
                payload = json.load(fin)
            self.assertEqual(payload["experiment"], "toy")
            self.assertEqual(payload["results"]["fwd"]["per_delta"][0]["conformal"]["quantile"], 0.8)


if __name__ == "__main__":
    unittest.main()
