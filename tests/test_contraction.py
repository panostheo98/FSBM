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
    compute_local_jacobian_diagnostic,
    conformal_quantile,
    evaluate_directional_contraction,
)
from fsbm.sde import paired_sdeint


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
