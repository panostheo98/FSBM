import unittest

import torch

from fsbm.trust import (
    TrustCertificate,
    calibrate_threshold,
    calibrate_typicality,
    composite_alarm,
    empirical_coverage,
    instance_alarm,
    local_amplification,
)


class GaussianSampler:
    def __init__(self, dim, mu=0.0):
        self.dim = dim
        self.mu = mu

    def __call__(self, n):
        return torch.randn(n, self.dim) + self.mu


def linear_drift(scale):
    def drift(x, t):
        return scale * x

    return drift


class TrustTests(unittest.TestCase):
    def test_calibrated_threshold_matches_linear_amplification(self):
        nfe, rate = 50, 0.5
        expected = (1.0 - rate / nfe) ** nfe
        cert = calibrate_threshold(
            linear_drift(-rate),
            GaussianSampler(2),
            direction="fwd",
            sigma=1.0,
            delta0=0.2,
            alpha=0.1,
            n_calib=64,
            nfe=nfe,
            seed=7,
            device="cpu",
        )
        self.assertAlmostEqual(cert.s_hat_alpha, expected, places=4)
        self.assertAlmostEqual(cert.coverage, 0.9, places=7)

    def test_calibration_matches_multi_probe_runtime_score(self):
        nfe, rate = 40, 0.35
        expected = (1.0 - rate / nfe) ** nfe
        kwargs = dict(
            direction="fwd", sigma=1.0, delta0=0.2, alpha=0.1,
            n_calib=64, nfe=nfe, seed=11, device="cpu",
            n_probes=5, score_aggregation="max",
        )
        cert = calibrate_threshold(
            linear_drift(-rate), GaussianSampler(3), **kwargs
        )
        cert_again = calibrate_threshold(
            linear_drift(-rate), GaussianSampler(3), **kwargs
        )
        self.assertAlmostEqual(cert.s_hat_alpha, expected, places=4)
        self.assertAlmostEqual(cert.s_hat_alpha, cert_again.s_hat_alpha, places=7)
        self.assertEqual(cert.n_probes, 5)
        self.assertEqual(cert.score_aggregation, "max")

    def test_local_amplification_is_constant_for_linear_drift(self):
        nfe, rate = 40, 0.4
        expected = (1.0 - rate / nfe) ** nfe
        queries = torch.randn(128, 2) * 3.0  # spread across the plane
        out = local_amplification(
            linear_drift(-rate),
            queries,
            direction="fwd",
            sigma=1.0,
            delta_probe=0.2,
            n_probes=8,
            nfe=nfe,
            seed=3,
            device="cpu",
        )
        self.assertEqual(out["mean"].shape, (128,))
        self.assertEqual(out["probe_scores"].shape, (128, 8))
        # Noise cancels under synchronous coupling => amplification is a constant.
        self.assertLess((out["mean"] - expected).abs().max().item(), 1e-3)

    def test_local_amplification_flags_expansive_region(self):
        # Expansive drift (positive rate) must score above a contractive one.
        queries = torch.randn(64, 2)
        expansive = local_amplification(
            linear_drift(+0.5), queries, direction="fwd", sigma=1.0,
            delta_probe=0.1, n_probes=4, nfe=40, seed=1, device="cpu",
        )["mean"]
        contractive = local_amplification(
            linear_drift(-0.5), queries, direction="fwd", sigma=1.0,
            delta_probe=0.1, n_probes=4, nfe=40, seed=1, device="cpu",
        )["mean"]
        self.assertGreater(expansive.mean().item(), 1.0)
        self.assertLess(contractive.mean().item(), 1.0)
        self.assertGreater(expansive.mean().item(), contractive.mean().item())

    def test_instance_alarm_rules(self):
        cert = TrustCertificate(
            direction="fwd", delta0=0.2, alpha=0.1, n_calib=100,
            s_hat_alpha=1.0, tau=0.25, baseline_terminal_error=0.0,
        )
        scores = torch.tensor([0.5, 0.9, 1.1, 2.0])
        alarm = instance_alarm(scores, cert)
        # vs quantile: score > 1.0 -> last two.
        self.assertTrue(torch.equal(
            alarm["alarm_vs_quantile"], torch.tensor([False, False, True, True])
        ))
        # vs tau: score*0.2 > 0.25 -> score > 1.25 -> only the last.
        self.assertTrue(torch.equal(
            alarm["alarm_vs_tau"], torch.tensor([False, False, False, True])
        ))
        # overall alarm is the union; trust is its complement.
        self.assertTrue(torch.equal(
            alarm["alarm"], torch.tensor([False, False, True, True])
        ))
        self.assertTrue(torch.equal(alarm["trust"], ~alarm["alarm"]))
        self.assertAlmostEqual(alarm["alarm_rate"], 0.5, places=6)

    def test_typicality_gate_flags_off_manifold_inputs(self):
        ref = GaussianSampler(8)
        gate = calibrate_typicality(
            ref, alpha=0.1, n_fit=384, n_calib=512, seed=0, device="cpu"
        )
        gate_again = calibrate_typicality(
            ref, alpha=0.1, n_fit=384, n_calib=512, seed=0, device="cpu"
        )
        self.assertEqual(gate.n_fit, 384)
        self.assertAlmostEqual(gate.threshold, gate_again.threshold, places=7)
        self.assertTrue(torch.equal(gate.mu, gate_again.mu))
        # In-distribution inputs mostly pass; far-shifted inputs mostly alarm.
        id_alarm = gate.alarm(ref(256)).float().mean().item()
        ood_alarm = gate.alarm(ref(256) + 6.0).float().mean().item()
        self.assertLess(id_alarm, 0.2)   # ~alpha, up to finite-sample noise
        self.assertGreater(ood_alarm, 0.9)
        self.assertGreater(gate.score(ref(64) + 6.0).mean().item(),
                           gate.score(ref(64)).mean().item())

    def test_composite_alarm_unions_both_signals(self):
        ref = GaussianSampler(4)
        gate = calibrate_typicality(ref, alpha=0.1, n_calib=256, seed=0, device="cpu")
        cert = TrustCertificate(direction="fwd", delta0=0.1, alpha=0.1,
                                n_calib=100, s_hat_alpha=1.0)
        queries = torch.cat([ref(8), ref(8) + 6.0], dim=0)  # 8 ID + 8 OOD
        # Amplification all below threshold -> transport says trust everything;
        # the typicality gate must still alarm the OOD half.
        amp_scores = torch.full((16,), 0.5)
        out = composite_alarm(queries, amp_scores, cert, gate)
        self.assertEqual(int(out["transport_alarm"].sum()), 0)
        self.assertGreaterEqual(int(out["typicality_alarm"][8:].sum()), 6)
        self.assertTrue(torch.equal(out["alarm"], out["typicality_alarm"]))
        self.assertAlmostEqual(
            out["nominal_id_false_alarm_upper_bound"], 0.2, places=7
        )

    def test_empirical_coverage_reports_gap(self):
        cert = TrustCertificate(
            direction="fwd", delta0=0.1, alpha=0.1, n_calib=100, s_hat_alpha=1.0,
        )
        scores = torch.tensor([0.2, 0.5, 0.9, 1.5])  # 3/4 covered
        cov = empirical_coverage(scores, cert)
        self.assertAlmostEqual(cov["empirical_coverage"], 0.75, places=6)
        self.assertAlmostEqual(cov["target_coverage"], 0.9, places=6)
        self.assertAlmostEqual(cov["coverage_gap"], -0.15, places=6)
        self.assertEqual(cov["num_covered"], 3)


if __name__ == "__main__":
    unittest.main()
