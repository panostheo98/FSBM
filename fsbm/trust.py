"""Per-instance runtime trust scoring for the FSBM generative SDE.

The contraction certificate in :mod:`fsbm.contraction` produces a *global*
conformal amplification quantile ``S_hat_alpha``: with probability ``1 - alpha``
a perturbation of size ``delta0`` at the boundary moves the generated endpoint by
at most ``S_hat_alpha * delta0``.  That is a population-level guarantee.

For a runtime "should I trust this output" alarm we need a *per-instance* score:
given the specific input the user is about to generate from, how sensitively does
the model behave *there*?  This module estimates the local amplification of the
learned flow at each query point by launching a small cloud of probe
perturbations and measuring how far their coupled trajectories separate.  The
per-instance score is then compared against the conformally calibrated threshold
``S_hat_alpha`` (and against a task tolerance ``tau``) to raise the alarm.

Both objects share the same nonconformity score (``amplification`` from
``paired_sdeint``), so the per-instance alarm is calibrated in the same units as
the population certificate.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch

from .contraction import (
    build_generator,
    conformal_quantile,
    sample_reference,
    sample_unit_sphere,
)
from .sde import paired_sdeint


Drift = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class TrustCertificate:
    """Population-level conformal amplification threshold used for the alarm.

    ``s_hat_alpha`` is the split-conformal ``(1 - alpha)`` quantile of the
    per-draw amplification on in-distribution calibration data.  ``tau`` and
    ``baseline_terminal_error`` are the optional task tolerance and any
    irreducible sampling error that a per-instance bound is added on top of.
    """

    direction: str
    delta0: float
    alpha: float
    n_calib: int
    s_hat_alpha: float
    n_probes: int = 1
    score_aggregation: str = "mean"
    order_statistic: Optional[int] = None
    tau: Optional[float] = None
    baseline_terminal_error: float = 0.0

    @property
    def coverage(self) -> float:
        return 1.0 - self.alpha

    def as_dict(self) -> Dict[str, object]:
        terminal_bound = (
            self.baseline_terminal_error + self.s_hat_alpha * self.delta0
        )
        payload = {
            "direction": self.direction,
            "delta0": self.delta0,
            "alpha": self.alpha,
            "coverage": self.coverage,
            "n_calib": self.n_calib,
            "s_hat_alpha": self.s_hat_alpha,
            "n_probes": self.n_probes,
            "score_aggregation": self.score_aggregation,
            "baseline_terminal_error": self.baseline_terminal_error,
            "terminal_deviation_bound": terminal_bound,
        }
        if self.order_statistic is not None:
            payload["order_statistic"] = self.order_statistic
        if self.tau is not None:
            payload["tau"] = self.tau
            payload["tolerance_margin"] = float(self.tau - terminal_bound)
            payload["within_tau"] = bool(terminal_bound <= self.tau)
        return payload


def calibrate_threshold(
    drift: Drift,
    reference_sampler,
    *,
    direction: str,
    sigma: float,
    delta0: float,
    alpha: float,
    n_calib: int,
    nfe: int,
    seed: int,
    device,
    tau: Optional[float] = None,
    baseline_terminal_error: float = 0.0,
    log_steps: int = 2,
    n_probes: int = 1,
    score_aggregation: str = "mean",
    batch_size: int = 4096,
) -> TrustCertificate:
    """Calibrate the conformal amplification threshold on in-distribution data.

    The exact runtime score must be calibrated: for every input, draw
    ``n_probes`` isotropic perturbations, aggregate their amplification using
    ``score_aggregation`` (``mean``, ``max``, or ``median``), and take the
    split-conformal ``(1 - alpha)`` quantile across calibration inputs.

    With ``n_probes=1`` this reduces to the single random-probe certificate used
    by :func:`fsbm.contraction.evaluate_directional_contraction`.  With
    ``score_aggregation="max"``, the guarantee covers the maximum over the K
    *sampled* probes for a new exchangeable input; it is not a worst-case
    Lipschitz certificate over every possible perturbation direction.
    """
    device = torch.device(device)
    if score_aggregation not in {"mean", "max", "median"}:
        raise ValueError("score_aggregation must be one of: mean, max, median")
    x0 = sample_reference(reference_sampler, int(n_calib), int(seed)).to(device)
    score_payload = local_amplification(
        drift,
        x0,
        direction=direction,
        sigma=sigma,
        delta_probe=float(delta0),
        n_probes=int(n_probes),
        nfe=int(nfe),
        seed=int(seed) + 100_003,
        device=device,
        batch_size=int(batch_size),
    )
    scores = score_payload[score_aggregation]
    conformal = conformal_quantile(scores, alpha)
    return TrustCertificate(
        direction=direction,
        delta0=float(delta0),
        alpha=float(alpha),
        n_calib=int(n_calib),
        s_hat_alpha=float(conformal["quantile"]),
        n_probes=int(n_probes),
        score_aggregation=score_aggregation,
        order_statistic=int(conformal["order_statistic"]),
        tau=None if tau is None else float(tau),
        baseline_terminal_error=float(baseline_terminal_error),
    )


@torch.no_grad()
def local_amplification(
    drift: Drift,
    queries: torch.Tensor,
    *,
    direction: str,
    sigma: float,
    delta_probe: float,
    n_probes: int,
    nfe: int,
    seed: int,
    device,
    batch_size: int = 4096,
) -> Dict[str, torch.Tensor]:
    """Estimate the local amplification of the flow at each query point.

    For every query ``x`` we draw ``n_probes`` isotropic perturbations of radius
    ``delta_probe``, roll ``x`` and each perturbed point through the generative
    SDE under shared noise, and record the endpoint separation ratio
    ``||x_T - y_T|| / delta_probe``.  Aggregating over probes gives a per-instance
    sensitivity estimate.

    Returns a dict with, per query (shape ``(B,)`` unless noted):

    - ``mean``   : direction-averaged sampled amplification;
    - ``max``    : worst sampled-probe amplification (a local sensitivity proxy);
    - ``median`` : robust central amplification;
    - ``std``    : spread across probe directions;
    - ``probe_scores`` : the full ``(B, n_probes)`` matrix.
    """
    device = torch.device(device)
    queries = queries.float().to(device)
    if queries.ndim != 2:
        raise ValueError("queries must have shape (B, D)")
    B, D = queries.shape
    n_probes = int(n_probes)
    if n_probes < 1:
        raise ValueError("n_probes must be >= 1")

    generator = build_generator(seed, device)
    all_scores = []
    for start in range(0, B, batch_size):
        chunk = queries[start : start + batch_size]
        cb = chunk.shape[0]
        # (cb * n_probes, D) replicated queries and their probe partners.
        x0 = chunk.repeat_interleave(n_probes, dim=0)
        dirs = sample_unit_sphere(cb * n_probes, D, device, generator)
        y0 = x0 + float(delta_probe) * dirs
        rollout = paired_sdeint(
            x0,
            y0,
            drift=drift,
            diffusion=lambda x, t: sigma,
            direction=direction,
            nfe=nfe,
            log_steps=2,
            generator=generator,
        )
        all_scores.append(rollout["amplification"].detach().cpu().reshape(cb, n_probes))

    probe_scores = torch.cat(all_scores, dim=0)
    return {
        "mean": probe_scores.mean(dim=1),
        "max": probe_scores.max(dim=1).values,
        "median": probe_scores.median(dim=1).values,
        "std": probe_scores.std(dim=1) if n_probes > 1 else torch.zeros(B),
        "probe_scores": probe_scores,
    }


def instance_alarm(
    scores: torch.Tensor,
    certificate: TrustCertificate,
    *,
    delta0: Optional[float] = None,
) -> Dict[str, object]:
    """Turn per-instance amplification scores into trust / alarm decisions.

    Two complementary rules, both in amplification units:

    - ``alarm_vs_quantile``: the instance is more sensitive than the population
      ``(1 - alpha)`` conformal quantile, ``score > s_hat_alpha``.  It sits in a
      more-expansive region than all but an ``alpha`` fraction of calibration
      draws.
    - ``alarm_vs_tau`` (only if ``tau`` is set): the observed sampled-probe output
      deviation ``baseline + score * delta0`` exceeds the task tolerance ``tau``.

    ``trust`` is the logical complement (no rule fired).
    """
    if scores.ndim != 1:
        scores = scores.reshape(-1)
    delta0 = certificate.delta0 if delta0 is None else float(delta0)
    s_hat = certificate.s_hat_alpha

    alarm_vs_quantile = scores > s_hat
    predicted_deviation = certificate.baseline_terminal_error + scores * delta0

    result = {
        "s_hat_alpha": s_hat,
        "delta0": delta0,
        "alarm_vs_quantile": alarm_vs_quantile,
        "predicted_output_deviation": predicted_deviation,
        "alarm_rate_vs_quantile": float(alarm_vs_quantile.float().mean().item()),
    }

    if certificate.tau is not None:
        alarm_vs_tau = predicted_deviation > float(certificate.tau)
        result["tau"] = float(certificate.tau)
        result["alarm_vs_tau"] = alarm_vs_tau
        result["alarm_rate_vs_tau"] = float(alarm_vs_tau.float().mean().item())
        result["alarm"] = alarm_vs_quantile | alarm_vs_tau
    else:
        result["alarm"] = alarm_vs_quantile

    result["trust"] = ~result["alarm"]
    result["alarm_rate"] = float(result["alarm"].float().mean().item())
    return result


@dataclass
class TypicalityGate:
    """Conformal input-typicality gate — a complement to the transport certificate.

    The amplification certificate certifies that the *transport* is stable at an
    input; it says nothing about whether the input itself is one the model should
    ever have been asked to generate from.  A wildly out-of-distribution input can
    be transported perfectly stably (near-isometrically) yet decode to garbage.

    This gate scores how atypical an input is under the calibration distribution
    (diagonal-Mahalanobis / RMS z-score) and thresholds it at the conformal
    ``(1 - alpha)`` quantile, so it flags inputs the transport certificate is blind
    to.  Together they form a two-signal trust monitor.
    """

    mu: torch.Tensor
    std: torch.Tensor
    threshold: float
    alpha: float
    n_fit: int
    n_calib: int

    @property
    def coverage(self) -> float:
        return 1.0 - self.alpha

    def score(self, queries: torch.Tensor) -> torch.Tensor:
        """RMS z-score per input: ~1 for typical inputs, grows as they get atypical."""
        q = queries.float().to(self.mu.device)
        z = (q - self.mu) / self.std
        return torch.sqrt(torch.mean(z * z, dim=-1)).detach().cpu()

    def alarm(self, queries: torch.Tensor) -> torch.Tensor:
        return self.score(queries) > self.threshold

    def as_dict(self) -> Dict[str, object]:
        return {"threshold": self.threshold, "alpha": self.alpha,
                "coverage": self.coverage, "n_fit": self.n_fit,
                "n_calib": self.n_calib}


def calibrate_typicality(
    reference_sampler,
    *,
    alpha: float,
    n_calib: int,
    seed: int,
    device,
    n_fit: Optional[int] = None,
    eps: float = 1e-6,
) -> TypicalityGate:
    """Fit and calibrate an input-typicality gate on disjoint draws.

    The location/scale transform is estimated on ``n_fit`` samples.  Its
    conformal threshold is then computed on a separate ``n_calib`` sample so
    that calibration and future exchangeable scores use the same fixed scoring
    function.  This is a marginal in-distribution false-alarm guarantee, not a
    proof that an arbitrary accepted input is in-distribution.
    """
    device = torch.device(device)
    n_fit = int(n_calib if n_fit is None else n_fit)
    fit = sample_reference(reference_sampler, n_fit, int(seed)).to(device)
    calib = sample_reference(reference_sampler, int(n_calib), int(seed) + 1).to(device)
    mu = fit.mean(dim=0)
    std = fit.std(dim=0).clamp_min(eps)
    z = torch.sqrt(torch.mean(((calib - mu) / std) ** 2, dim=-1))
    conformal = conformal_quantile(z.detach().cpu(), alpha)
    return TypicalityGate(
        mu=mu, std=std, threshold=float(conformal["quantile"]),
        alpha=float(alpha), n_fit=n_fit, n_calib=int(n_calib),
    )


def composite_alarm(
    queries: torch.Tensor,
    amplification_scores: torch.Tensor,
    certificate: TrustCertificate,
    gate: TypicalityGate,
    *,
    delta0: Optional[float] = None,
) -> Dict[str, object]:
    """Two-signal trust monitor: transport-stability OR input-atypicality.

    Raises the alarm if either the amplification exceeds the certified quantile
    (transport unstable) or the input is more atypical than the calibrated gate
    (off-manifold).  Returns the per-signal masks plus the combined decision.
    """
    amp = instance_alarm(amplification_scores, certificate, delta0=delta0)
    atypical = gate.alarm(queries)
    combined = amp["alarm"] | atypical
    result = {
        "transport_alarm": amp["alarm"],
        "typicality_score": gate.score(queries),
        "typicality_alarm": atypical,
        "typicality_threshold": gate.threshold,
        "alarm": combined,
        "trust": ~combined,
        "alarm_rate": float(combined.float().mean().item()),
        "transport_alarm_rate": amp["alarm_rate"],
        "typicality_alarm_rate": float(atypical.float().mean().item()),
    }
    # No independence is needed for this union-bound statement.  It applies
    # under the exchangeability assumptions of both calibrations, and only when
    # the transport alarm has no extra, uncalibrated task-tolerance rule.
    if certificate.tau is None:
        result["nominal_id_false_alarm_upper_bound"] = min(
            1.0, float(certificate.alpha + gate.alpha)
        )
    return result


def empirical_coverage(
    test_scores: torch.Tensor,
    certificate: TrustCertificate,
) -> Dict[str, float]:
    """Fraction of held-out amplifications that fall under ``s_hat_alpha``.

    This is a diagnostic for one fixed calibration split.  Split conformal
    guarantees marginal coverage over the joint calibration/test draw; it does
    not require every finite held-out batch (or every fixed calibration split)
    to achieve a fraction at least ``1 - alpha``.  ``coverage_gap`` is
    ``empirical - target``.
    """
    if test_scores.ndim != 1:
        test_scores = test_scores.reshape(-1)
    covered = (test_scores <= certificate.s_hat_alpha).float()
    empirical = float(covered.mean().item())
    return {
        "empirical_coverage": empirical,
        "target_coverage": certificate.coverage,
        "coverage_gap": empirical - certificate.coverage,
        "num_covered": int(covered.sum().item()),
        "n_test": int(test_scores.numel()),
    }
