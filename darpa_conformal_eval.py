"""Repeated split-conformal endpoint-amplification evaluation for FSBM.

This script evaluates the *runtime score that will actually be deployed*: for
each validation input it launches ``n_probes`` synchronous-noise perturbation
rollouts with :func:`fsbm.trust.local_amplification`, aggregates them by
``mean``, ``max``, or ``median``, and calibrates that same aggregate.

Two different guarantees are reported and deliberately kept separate:

1. Ordinary split conformal (always reported).  For
   ``k = ceil((n_calib + 1) * (1 - alpha))``, the k-th calibration score gives
   marginal coverage for a new exchangeable score.  Coverage is marginal over
   the joint calibration/test draw; an individual realized split need not have
   empirical coverage above ``1 - alpha``.
2. Fixed-deployment PAC tolerance (optional ``--pac-beta``).  The calibration
   rank ``r`` is the smallest rank satisfying
   ``BetaCDF(1-alpha; r, n_calib+1-r) <= beta``.  With probability at least
   ``1-beta`` over calibration, the resulting fixed threshold has population
   coverage at least ``1-alpha`` (under the usual iid/continuous-score
   assumptions).  This is a stronger claim and normally yields a larger bound.

For the ordinary threshold, and also for the optional PAC rank, held-out count
lower-tail probabilities use the exact beta-binomial rank distribution.  A
familiar fixed-threshold binomial calculation is also emitted, but every such
column is explicitly named ``fixed_threshold_binomial_diagnostic`` because it
conditions away calibration-threshold randomness and is not the conformal
validity calculation.

Validation identities are sampled disjointly without replacement whenever the
reference sampler exposes ``dataset.x_data`` (the FFHQ latent tasks).  Otherwise
calibration and test queries are independent seeded sampler draws (the stunnel
Gaussian task).  All query, score, checkpoint, data, configuration, and run
settings are hashed for auditability.

Example (CPU-friendly starting point)::

    python darpa_conformal_eval.py \
      --checkpoint outputs/runs/stunnel/.../checkpoints/last.ckpt \
      --directions fwd,bwd --deltas 0.05,0.1,0.2 --alphas 0.05,0.1 \
      --n-splits 10 --n-calib 128 --n-test 256 \
      --n-probes 4 --aggregation max --nfe 50 --device cpu
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from fsbm.contraction import conformal_quantile, sample_reference
from fsbm.trust import local_amplification


SCHEMA_VERSION = "darpa-conformal-eval-v1"
AGGREGATIONS = {"mean", "max", "median"}
DIRECTIONS = {"fwd", "bwd"}


@dataclass(frozen=True)
class EvaluationSettings:
    """Complete, hashable settings for one repeated evaluation."""

    directions: Tuple[str, ...] = ("fwd", "bwd")
    deltas: Tuple[float, ...] = (0.05, 0.1, 0.2)
    alphas: Tuple[float, ...] = (0.05, 0.1)
    n_splits: int = 10
    n_calib: int = 128
    n_test: int = 256
    n_probes: int = 4
    aggregation: str = "max"
    nfe: int = 50
    batch_size: int = 512
    seed: int = 20260716
    device: str = "cpu"
    baseline_terminal_error: float = 0.0
    tolerance: Optional[float] = None
    pac_beta: Optional[float] = None

    def validate(self) -> None:
        if not self.directions or any(d not in DIRECTIONS for d in self.directions):
            raise ValueError("directions must be a non-empty subset of: fwd, bwd")
        if len(set(self.directions)) != len(self.directions):
            raise ValueError("directions must not contain duplicates")
        if not self.deltas or any((not math.isfinite(d) or d <= 0) for d in self.deltas):
            raise ValueError("deltas must be finite and strictly positive")
        if len(set(self.deltas)) != len(self.deltas):
            raise ValueError("deltas must not contain duplicates")
        if not self.alphas or any((not math.isfinite(a) or not 0 < a < 1) for a in self.alphas):
            raise ValueError("alphas must lie strictly between 0 and 1")
        if len(set(self.alphas)) != len(self.alphas):
            raise ValueError("alphas must not contain duplicates")
        for name in ("n_splits", "n_calib", "n_test", "n_probes", "nfe", "batch_size"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {sorted(AGGREGATIONS)}")
        minimum_finite_alpha = 1.0 / (self.n_calib + 1)
        invalid = [a for a in self.alphas if a < minimum_finite_alpha]
        if invalid:
            raise ValueError(
                "ordinary conformal thresholds would be +infinity for alpha "
                f"{invalid}; use alpha >= {minimum_finite_alpha:.8g} or increase "
                "n_calib"
            )
        if not math.isfinite(self.baseline_terminal_error):
            raise ValueError("baseline_terminal_error must be finite")
        if self.tolerance is not None and not math.isfinite(self.tolerance):
            raise ValueError("tolerance must be finite when supplied")
        if self.pac_beta is not None:
            if not 0 < self.pac_beta < 1:
                raise ValueError("pac_beta must lie strictly between 0 and 1")
            for alpha in self.alphas:
                pac_order_statistic(self.n_calib, alpha, self.pac_beta)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["directions"] = list(self.directions)
        out["deltas"] = list(self.deltas)
        out["alphas"] = list(self.alphas)
        return out


@dataclass
class QuerySplit:
    calibration: torch.Tensor
    test: torch.Tensor
    metadata: Dict[str, Any]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(value: torch.Tensor, chunk_size: int = 1024 * 1024) -> str:
    """Hash tensor shape, dtype, and canonical contiguous CPU bytes."""
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    array = tensor.numpy()
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes({"shape": list(array.shape), "dtype": str(array.dtype)}))
    raw = memoryview(array).cast("B")
    for start in range(0, len(raw), chunk_size):
        digest.update(raw[start : start + chunk_size])
    return digest.hexdigest()


def derive_seed(master_seed: int, *labels: Any) -> int:
    """Stable seed derivation independent of Python's randomized ``hash``."""
    payload = _canonical_json_bytes([int(master_seed), *labels])
    # torch.Generator accepts signed 64-bit seeds; stay in its positive range.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if torch.is_tensor(value):
        converted = value.detach().cpu().tolist()
        if isinstance(converted, list):
            return [_json_safe(v) for v in converted]
        return _json_safe(converted)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _logsumexp(log_values: Sequence[float]) -> float:
    if not log_values:
        return -math.inf
    maximum = max(log_values)
    if maximum == -math.inf:
        return -math.inf
    return maximum + math.log(math.fsum(math.exp(v - maximum) for v in log_values))


def _log_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def binomial_lower_tail_cdf(successes: int, trials: int, probability: float) -> float:
    """Exact-distribution (non-asymptotic) binomial lower-tail probability."""
    successes, trials, probability = int(successes), int(trials), float(probability)
    if trials < 0 or successes < 0:
        return 0.0
    if successes >= trials:
        return 1.0
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    if probability == 0:
        return 1.0
    if probability == 1:
        return 0.0
    logs = [
        _log_choose(trials, k)
        + k * math.log(probability)
        + (trials - k) * math.log1p(-probability)
        for k in range(successes + 1)
    ]
    return min(1.0, max(0.0, math.exp(_logsumexp(logs))))


def _binomial_upper_tail_cdf(minimum_successes: int, trials: int, probability: float) -> float:
    """P[Binomial(trials, probability) >= minimum_successes]."""
    minimum_successes, trials = int(minimum_successes), int(trials)
    if minimum_successes <= 0:
        return 1.0
    if minimum_successes > trials:
        return 0.0
    p = float(probability)
    if p == 0:
        return 0.0
    if p == 1:
        return 1.0
    logs = [
        _log_choose(trials, k)
        + k * math.log(p)
        + (trials - k) * math.log1p(-p)
        for k in range(minimum_successes, trials + 1)
    ]
    return min(1.0, max(0.0, math.exp(_logsumexp(logs))))


def beta_cdf_integer_shape(x: float, a: int, b: int) -> float:
    """Regularized Beta CDF for positive integer shapes, without SciPy.

    Uses ``I_x(a,b) = P[Binomial(a+b-1, x) >= a]``.
    """
    if not 0 <= float(x) <= 1:
        raise ValueError("x must be in [0, 1]")
    if int(a) < 1 or int(b) < 1:
        raise ValueError("a and b must be positive integers")
    return _binomial_upper_tail_cdf(int(a), int(a) + int(b) - 1, float(x))


def pac_order_statistic(n_calib: int, alpha: float, beta: float) -> Dict[str, Any]:
    """Smallest finite order rank giving the fixed-deployment PAC claim."""
    n, alpha, beta = int(n_calib), float(alpha), float(beta)
    if n < 1:
        raise ValueError("n_calib must be >= 1")
    if not 0 < alpha < 1 or not 0 < beta < 1:
        raise ValueError("alpha and beta must lie strictly between 0 and 1")
    target = 1.0 - alpha
    # The CDF decreases monotonically with rank.  Even the sample maximum may
    # be insufficient for a requested (alpha, beta) at a small n.
    at_maximum = beta_cdf_integer_shape(target, n, 1)
    if at_maximum > beta:
        raise ValueError(
            "no finite PAC order statistic exists for "
            f"n_calib={n}, alpha={alpha:g}, beta={beta:g}; even rank n has "
            f"BetaCDF={at_maximum:.6g} > beta. Increase n_calib."
        )
    low, high = 1, n
    while low < high:
        middle = (low + high) // 2
        cdf = beta_cdf_integer_shape(target, middle, n + 1 - middle)
        if cdf <= beta:
            high = middle
        else:
            low = middle + 1
    rank = low
    cdf = beta_cdf_integer_shape(target, rank, n + 1 - rank)
    return {
        "order_statistic": rank,
        "beta_cdf_at_target": cdf,
        "alpha": alpha,
        "beta": beta,
        "target_coverage": target,
    }


def beta_binomial_pmf(
    successes: int, trials: int, a: int, b: int
) -> float:
    """Beta-binomial probability in log space."""
    k, m, a, b = int(successes), int(trials), int(a), int(b)
    if m < 0 or a < 1 or b < 1:
        raise ValueError("trials >= 0 and a,b >= 1 are required")
    if k < 0 or k > m:
        return 0.0
    log_p = (
        _log_choose(m, k)
        + math.lgamma(k + a)
        + math.lgamma(m - k + b)
        - math.lgamma(m + a + b)
        + math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
    )
    return math.exp(log_p)


def beta_binomial_lower_tail_cdf(
    successes: int, trials: int, a: int, b: int
) -> float:
    """Exact marginal rank-model lower tail for held-out covered counts."""
    k, m = int(successes), int(trials)
    if k < 0:
        return 0.0
    if k >= m:
        return 1.0
    logs = []
    for value in range(k + 1):
        probability = beta_binomial_pmf(value, m, a, b)
        logs.append(-math.inf if probability == 0 else math.log(probability))
    return min(1.0, max(0.0, math.exp(_logsumexp(logs))))


def repeated_beta_binomial_lower_tail_cdf(
    total_successes: int, n_splits: int, trials_per_split: int, a: int, b: int
) -> float:
    """Exact repeated-split null tail via beta-binomial PMF convolution.

    The discrete null is exact; FFT is used only to evaluate its polynomial
    convolution efficiently.  Independence of the repeated random splits is
    part of this aggregate calculation.
    """
    s, m = int(n_splits), int(trials_per_split)
    if s < 1 or m < 0:
        raise ValueError("n_splits >= 1 and trials_per_split >= 0 are required")
    maximum = s * m
    if total_successes < 0:
        return 0.0
    if total_successes >= maximum:
        return 1.0
    pmf = np.array([beta_binomial_pmf(k, m, a, b) for k in range(m + 1)])
    pmf /= pmf.sum()
    result_length = maximum + 1
    fft_length = 1 << (result_length - 1).bit_length()
    spectrum = np.fft.rfft(pmf, fft_length)
    total_pmf = np.fft.irfft(spectrum**s, fft_length)[:result_length]
    total_pmf = np.maximum(total_pmf, 0.0)
    total_pmf /= total_pmf.sum()
    return float(np.clip(total_pmf[: int(total_successes) + 1].sum(), 0.0, 1.0))


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if trials < 1:
        return float("nan"), float("nan")
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials**2)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _dataset_x_data(reference_sampler: Any) -> Optional[torch.Tensor]:
    dataset = getattr(reference_sampler, "dataset", None)
    data = getattr(dataset, "x_data", None)
    if data is None:
        return None
    return torch.as_tensor(data)


def sampler_provenance(reference_sampler: Any) -> Dict[str, Any]:
    descriptor: Dict[str, Any] = {
        "sampler_class": (
            f"{reference_sampler.__class__.__module__}."
            f"{reference_sampler.__class__.__qualname__}"
        )
    }
    x_data = _dataset_x_data(reference_sampler)
    if x_data is not None:
        descriptor.update(
            {
                "sampling_mode": "dataset_disjoint_without_replacement",
                "dataset_class": (
                    f"{reference_sampler.dataset.__class__.__module__}."
                    f"{reference_sampler.dataset.__class__.__qualname__}"
                ),
                "x_data_shape": list(x_data.shape),
                "x_data_dtype": str(x_data.dtype),
                "x_data_sha256": sha256_tensor(x_data),
            }
        )
    else:
        descriptor["sampling_mode"] = "independent_seeded_sampler_draws"
        for name in ("mean", "std"):
            value = getattr(reference_sampler, name, None)
            if value is not None:
                tensor = torch.as_tensor(value)
                descriptor[f"{name}_shape"] = list(tensor.shape)
                descriptor[f"{name}_sha256"] = sha256_tensor(tensor)
                if tensor.numel() <= 32:
                    descriptor[name] = _json_safe(tensor)
    descriptor["data_sha256"] = sha256_json(descriptor)
    return descriptor


def draw_query_split(
    reference_sampler: Any,
    n_calib: int,
    n_test: int,
    *,
    seed: int,
    direction: str,
    split: int,
) -> QuerySplit:
    """Draw one auditable calibration/test split."""
    split_seed = derive_seed(seed, "query_split", direction, int(split))
    x_data = _dataset_x_data(reference_sampler)
    if x_data is not None:
        total = int(n_calib) + int(n_test)
        if total > int(x_data.shape[0]):
            raise ValueError(
                f"n_calib+n_test={total} exceeds validation dataset size "
                f"{x_data.shape[0]} for {direction}; disjoint sampling requested"
            )
        generator = torch.Generator().manual_seed(split_seed)
        indices = torch.randperm(int(x_data.shape[0]), generator=generator)[:total]
        calibration_indices = indices[:n_calib]
        test_indices = indices[n_calib:]
        flat = x_data.detach().cpu().float().reshape(x_data.shape[0], -1)
        calibration = flat.index_select(0, calibration_indices)
        test = flat.index_select(0, test_indices)
        metadata = {
            "sampling_mode": "dataset_disjoint_without_replacement",
            "query_split_seed": split_seed,
            "calibration_draw_seed": None,
            "test_draw_seed": None,
            "calibration_identity_sha256": sha256_tensor(calibration_indices),
            "test_identity_sha256": sha256_tensor(test_indices),
            "identity_overlap_count": 0,
        }
    else:
        calibration_seed = derive_seed(seed, "calibration_draw", direction, int(split))
        test_seed = derive_seed(seed, "test_draw", direction, int(split))
        calibration = sample_reference(reference_sampler, int(n_calib), calibration_seed)
        test = sample_reference(reference_sampler, int(n_test), test_seed)
        calibration = calibration.detach().cpu().float().reshape(n_calib, -1)
        test = test.detach().cpu().float().reshape(n_test, -1)
        metadata = {
            "sampling_mode": "independent_seeded_sampler_draws",
            "query_split_seed": split_seed,
            "calibration_draw_seed": calibration_seed,
            "test_draw_seed": test_seed,
            "calibration_identity_sha256": None,
            "test_identity_sha256": None,
            "identity_overlap_count": None,
        }
    metadata.update(
        {
            "calibration_query_sha256": sha256_tensor(calibration),
            "test_query_sha256": sha256_tensor(test),
        }
    )
    return QuerySplit(calibration=calibration, test=test, metadata=metadata)


def _tensor_summary(values: torch.Tensor, prefix: str) -> Dict[str, float]:
    values = values.detach().cpu().float().reshape(-1)
    quantiles = torch.quantile(values, torch.tensor([0.05, 0.5, 0.95]))
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_q05": float(quantiles[0]),
        f"{prefix}_median": float(quantiles[1]),
        f"{prefix}_q95": float(quantiles[2]),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
    }


def _threshold_metrics(
    calibration_scores: torch.Tensor,
    test_scores: torch.Tensor,
    *,
    rank: int,
    alpha: float,
    delta: float,
    baseline_terminal_error: float,
    tolerance: Optional[float],
    prefix: str,
) -> Dict[str, Any]:
    n_calib = int(calibration_scores.numel())
    n_test = int(test_scores.numel())
    sorted_calibration = torch.sort(calibration_scores).values
    threshold = float(sorted_calibration[int(rank) - 1])
    covered = test_scores <= threshold
    successes = int(covered.sum())
    coverage = successes / n_test
    expected_rank_coverage = rank / (n_calib + 1)
    a, b = int(rank), n_calib + 1 - int(rank)
    beta_p = beta_binomial_lower_tail_cdf(successes, n_test, a, b)
    fixed_p = binomial_lower_tail_cdf(successes, n_test, 1.0 - alpha)
    wilson_low, wilson_high = wilson_interval(successes, n_test)
    endpoint_bound = float(delta) * threshold
    total_bound = float(baseline_terminal_error) + endpoint_bound
    tie_count = int((calibration_scores == threshold).sum())
    return {
        f"{prefix}_order_statistic": int(rank),
        f"{prefix}_amplification_threshold": threshold,
        f"{prefix}_endpoint_aggregate_bound": endpoint_bound,
        f"{prefix}_predicted_total_error_bound": total_bound,
        f"{prefix}_num_covered": successes,
        f"{prefix}_empirical_coverage": coverage,
        f"{prefix}_coverage_gap_vs_nominal": coverage - (1.0 - alpha),
        f"{prefix}_marginal_expected_rank_coverage": expected_rank_coverage,
        f"{prefix}_marginal_beta_binomial_a": a,
        f"{prefix}_marginal_beta_binomial_b": b,
        f"{prefix}_exact_beta_binomial_marginal_lower_tail_pvalue": beta_p,
        f"{prefix}_fixed_threshold_binomial_diagnostic_null_coverage": 1.0 - alpha,
        f"{prefix}_fixed_threshold_binomial_diagnostic_lower_tail_pvalue": fixed_p,
        f"{prefix}_fixed_threshold_binomial_diagnostic_wilson95_low": wilson_low,
        f"{prefix}_fixed_threshold_binomial_diagnostic_wilson95_high": wilson_high,
        f"{prefix}_calibration_threshold_tie_count": tie_count,
        f"{prefix}_within_tolerance": (
            None if tolerance is None else bool(total_bound <= float(tolerance))
        ),
    }


def evaluate_model(
    model: Any, settings: EvaluationSettings, *, verbose: bool = True
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Run repeated experiments on an already restored FSBM-like model."""
    settings.validate()
    if hasattr(model, "eval"):
        model.eval()
    device = torch.device(settings.device)
    rows: List[Dict[str, Any]] = []
    data_provenance: Dict[str, Any] = {}
    run_hash = sha256_json({"schema": SCHEMA_VERSION, **settings.to_dict()})

    for direction in settings.directions:
        reference = model.get_contraction_reference(direction)
        direction_data = sampler_provenance(reference)
        data_provenance[direction] = direction_data
        drift = model.build_drift(direction)

        for split in range(settings.n_splits):
            query_split = draw_query_split(
                reference,
                settings.n_calib,
                settings.n_test,
                seed=settings.seed,
                direction=direction,
                split=split,
            )
            queries = torch.cat(
                [query_split.calibration, query_split.test], dim=0
            )

            for delta in settings.deltas:
                score_seed = derive_seed(
                    settings.seed,
                    "runtime_score",
                    direction,
                    split,
                    format(delta, ".17g"),
                    settings.n_probes,
                    settings.aggregation,
                )
                payload = local_amplification(
                    drift,
                    queries,
                    direction=direction,
                    sigma=float(model.sigma),
                    delta_probe=float(delta),
                    n_probes=int(settings.n_probes),
                    nfe=int(settings.nfe),
                    seed=score_seed,
                    device=device,
                    batch_size=int(settings.batch_size),
                )
                if settings.aggregation not in payload:
                    raise KeyError(
                        f"local_amplification did not return {settings.aggregation!r}"
                    )
                scores = payload[settings.aggregation].detach().cpu().float().reshape(-1)
                expected = settings.n_calib + settings.n_test
                if scores.numel() != expected:
                    raise RuntimeError(
                        f"local_amplification returned {scores.numel()} scores; "
                        f"expected {expected}"
                    )
                calibration_scores = scores[: settings.n_calib]
                test_scores = scores[settings.n_calib :]

                for alpha in settings.alphas:
                    conformal = conformal_quantile(calibration_scores, float(alpha))
                    if not conformal["finite_threshold"]:
                        raise RuntimeError(
                            "settings validation permitted a non-finite threshold"
                        )
                    row: Dict[str, Any] = {
                        "schema_version": SCHEMA_VERSION,
                        "run_config_sha256": run_hash,
                        "data_sha256": direction_data["data_sha256"],
                        "direction": direction,
                        "split": split,
                        "delta": float(delta),
                        "alpha": float(alpha),
                        "nominal_coverage": 1.0 - float(alpha),
                        "n_calib": settings.n_calib,
                        "n_test": settings.n_test,
                        "n_probes": settings.n_probes,
                        "aggregation": settings.aggregation,
                        "nfe": settings.nfe,
                        "batch_size": settings.batch_size,
                        "baseline_terminal_error": settings.baseline_terminal_error,
                        "tolerance": settings.tolerance,
                        "pac_beta": settings.pac_beta,
                        "runtime_score_seed": score_seed,
                        "calibration_score_sha256": sha256_tensor(calibration_scores),
                        "test_score_sha256": sha256_tensor(test_scores),
                        **query_split.metadata,
                        **_tensor_summary(calibration_scores, "calibration_score"),
                        **_tensor_summary(test_scores, "test_score"),
                    }
                    ordinary = _threshold_metrics(
                        calibration_scores,
                        test_scores,
                        rank=int(conformal["order_statistic"]),
                        alpha=float(alpha),
                        delta=float(delta),
                        baseline_terminal_error=settings.baseline_terminal_error,
                        tolerance=settings.tolerance,
                        prefix="ordinary",
                    )
                    row.update(ordinary)

                    # Fixed-deployment PAC fields are always present so the CSV
                    # schema is stable; they remain blank when PAC is disabled.
                    pac_fields = {
                        "pac_beta_cdf_at_target": None,
                        "pac_order_statistic": None,
                        "pac_amplification_threshold": None,
                        "pac_endpoint_aggregate_bound": None,
                        "pac_predicted_total_error_bound": None,
                        "pac_num_covered": None,
                        "pac_empirical_coverage": None,
                        "pac_coverage_gap_vs_nominal": None,
                        "pac_marginal_expected_rank_coverage": None,
                        "pac_marginal_beta_binomial_a": None,
                        "pac_marginal_beta_binomial_b": None,
                        "pac_exact_beta_binomial_marginal_lower_tail_pvalue": None,
                        "pac_fixed_threshold_binomial_diagnostic_null_coverage": None,
                        "pac_fixed_threshold_binomial_diagnostic_lower_tail_pvalue": None,
                        "pac_fixed_threshold_binomial_diagnostic_wilson95_low": None,
                        "pac_fixed_threshold_binomial_diagnostic_wilson95_high": None,
                        "pac_calibration_threshold_tie_count": None,
                        "pac_within_tolerance": None,
                    }
                    if settings.pac_beta is not None:
                        pac_spec = pac_order_statistic(
                            settings.n_calib, float(alpha), settings.pac_beta
                        )
                        pac_fields.update(
                            _threshold_metrics(
                                calibration_scores,
                                test_scores,
                                rank=int(pac_spec["order_statistic"]),
                                alpha=float(alpha),
                                delta=float(delta),
                                baseline_terminal_error=settings.baseline_terminal_error,
                                tolerance=settings.tolerance,
                                prefix="pac",
                            )
                        )
                        pac_fields["pac_beta_cdf_at_target"] = pac_spec[
                            "beta_cdf_at_target"
                        ]
                    row.update(pac_fields)
                    rows.append(row)

                if verbose:
                    latest = rows[-1]
                    message = (
                        f"[{direction}] split={split + 1}/{settings.n_splits} "
                        f"delta={delta:g} ordinary coverage="
                        f"{latest['ordinary_empirical_coverage']:.3f}, "
                        f"bound={latest['ordinary_endpoint_aggregate_bound']:.4g}"
                    )
                    if settings.pac_beta is not None:
                        message += (
                            f"; PAC coverage={latest['pac_empirical_coverage']:.3f}, "
                            f"bound={latest['pac_endpoint_aggregate_bound']:.4g}"
                        )
                    print(message, flush=True)

    aggregates = aggregate_results(rows)
    return rows, aggregates, data_provenance


def _summary(values: Sequence[float], prefix: str) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_std_across_splits": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        f"{prefix}_q05": float(np.quantile(array, 0.05)),
        f"{prefix}_median": float(np.quantile(array, 0.5)),
        f"{prefix}_q95": float(np.quantile(array, 0.95)),
        f"{prefix}_min": float(array.min()),
        f"{prefix}_max": float(array.max()),
    }


def aggregate_results(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate repeated splits and evaluate their exact rank-model null."""
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (row["direction"], float(row["delta"]), float(row["alpha"]))
        groups.setdefault(key, []).append(row)

    output: List[Dict[str, Any]] = []
    for (direction, delta, alpha), group in sorted(groups.items()):
        first = group[0]
        n_splits = len(group)
        n_test = int(first["n_test"])
        total_trials = n_splits * n_test
        ordinary_total = sum(int(r["ordinary_num_covered"]) for r in group)
        ordinary_rank = int(first["ordinary_order_statistic"])
        ordinary_a = ordinary_rank
        ordinary_b = int(first["n_calib"]) + 1 - ordinary_rank
        ordinary_wilson = wilson_interval(ordinary_total, total_trials)
        aggregate: Dict[str, Any] = {
            "schema_version": first["schema_version"],
            "run_config_sha256": first["run_config_sha256"],
            "data_sha256": first["data_sha256"],
            "direction": direction,
            "delta": delta,
            "alpha": alpha,
            "nominal_coverage": 1.0 - alpha,
            "n_splits": n_splits,
            "n_calib_per_split": int(first["n_calib"]),
            "n_test_per_split": n_test,
            "total_test_evaluations": total_trials,
            "n_probes": int(first["n_probes"]),
            "aggregation": first["aggregation"],
            "nfe": int(first["nfe"]),
            "tolerance": first["tolerance"],
            "pac_beta": first["pac_beta"],
            "ordinary_order_statistic": ordinary_rank,
            "ordinary_marginal_expected_rank_coverage": first[
                "ordinary_marginal_expected_rank_coverage"
            ],
            "ordinary_total_covered": ordinary_total,
            "ordinary_pooled_empirical_coverage": ordinary_total / total_trials,
            "ordinary_num_splits_below_nominal_descriptive": sum(
                float(r["ordinary_empirical_coverage"]) < 1.0 - alpha for r in group
            ),
            "ordinary_exact_repeated_split_beta_binomial_lower_tail_pvalue": (
                repeated_beta_binomial_lower_tail_cdf(
                    ordinary_total,
                    n_splits,
                    n_test,
                    ordinary_a,
                    ordinary_b,
                )
            ),
            # Explicitly diagnostic: this treats all realized thresholds as if
            # their success probability were the fixed nominal target.
            "ordinary_pooled_fixed_threshold_binomial_diagnostic_lower_tail_pvalue": (
                binomial_lower_tail_cdf(ordinary_total, total_trials, 1.0 - alpha)
            ),
            "ordinary_pooled_fixed_threshold_binomial_diagnostic_wilson95_low": ordinary_wilson[0],
            "ordinary_pooled_fixed_threshold_binomial_diagnostic_wilson95_high": ordinary_wilson[1],
            **_summary(
                [float(r["ordinary_empirical_coverage"]) for r in group],
                "ordinary_split_coverage",
            ),
            **_summary(
                [float(r["ordinary_amplification_threshold"]) for r in group],
                "ordinary_amplification_threshold",
            ),
            **_summary(
                [float(r["ordinary_endpoint_aggregate_bound"]) for r in group],
                "ordinary_endpoint_aggregate_bound",
            ),
            **_summary(
                [
                    float(r["ordinary_exact_beta_binomial_marginal_lower_tail_pvalue"])
                    for r in group
                ],
                "ordinary_split_exact_beta_binomial_pvalue",
            ),
        }

        if first["pac_beta"] is not None:
            pac_total = sum(int(r["pac_num_covered"]) for r in group)
            pac_rank = int(first["pac_order_statistic"])
            pac_a = pac_rank
            pac_b = int(first["n_calib"]) + 1 - pac_rank
            pac_wilson = wilson_interval(pac_total, total_trials)
            aggregate.update(
                {
                    "pac_order_statistic": pac_rank,
                    "pac_beta_cdf_at_target": first["pac_beta_cdf_at_target"],
                    "pac_marginal_expected_rank_coverage": first[
                        "pac_marginal_expected_rank_coverage"
                    ],
                    "pac_total_covered": pac_total,
                    "pac_pooled_empirical_coverage": pac_total / total_trials,
                    "pac_num_splits_below_nominal_descriptive": sum(
                        float(r["pac_empirical_coverage"]) < 1.0 - alpha for r in group
                    ),
                    "pac_exact_repeated_split_beta_binomial_lower_tail_pvalue": (
                        repeated_beta_binomial_lower_tail_cdf(
                            pac_total, n_splits, n_test, pac_a, pac_b
                        )
                    ),
                    "pac_pooled_fixed_threshold_binomial_diagnostic_lower_tail_pvalue": (
                        binomial_lower_tail_cdf(pac_total, total_trials, 1.0 - alpha)
                    ),
                    "pac_pooled_fixed_threshold_binomial_diagnostic_wilson95_low": pac_wilson[0],
                    "pac_pooled_fixed_threshold_binomial_diagnostic_wilson95_high": pac_wilson[1],
                    **_summary(
                        [float(r["pac_empirical_coverage"]) for r in group],
                        "pac_split_coverage",
                    ),
                    **_summary(
                        [float(r["pac_amplification_threshold"]) for r in group],
                        "pac_amplification_threshold",
                    ),
                    **_summary(
                        [float(r["pac_endpoint_aggregate_bound"]) for r in group],
                        "pac_endpoint_aggregate_bound",
                    ),
                    **_summary(
                        [
                            float(r["pac_exact_beta_binomial_marginal_lower_tail_pvalue"])
                            for r in group
                        ],
                        "pac_split_exact_beta_binomial_pvalue",
                    ),
                }
            )
        output.append(aggregate)
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key)) for key in columns})


def _group_aggregates(
    aggregates: Sequence[Mapping[str, Any]]
) -> Iterable[Tuple[str, float, List[Mapping[str, Any]]]]:
    keys = sorted({(str(r["direction"]), float(r["alpha"])) for r in aggregates})
    for direction, alpha in keys:
        group = [
            r
            for r in aggregates
            if r["direction"] == direction and float(r["alpha"]) == alpha
        ]
        yield direction, alpha, sorted(group, key=lambda r: float(r["delta"]))


def make_figures(
    aggregates: Sequence[Mapping[str, Any]], figure_dir: Path
) -> List[str]:
    """Write repeated-split coverage and endpoint-bound figures."""
    # Lazy import keeps the statistical/sampling API usable in minimal CPU
    # environments and makes ``--no-figures`` genuinely independent of plotting.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    for direction, alpha, group in _group_aggregates(aggregates):
        x = np.array([float(r["delta"]) for r in group])
        y = np.array([float(r["ordinary_split_coverage_mean"]) for r in group])
        low = np.array([float(r["ordinary_split_coverage_q05"]) for r in group])
        high = np.array([float(r["ordinary_split_coverage_q95"]) for r in group])
        line = ax.plot(x, y, "-o", label=f"ordinary {direction}, alpha={alpha:g}")[0]
        ax.fill_between(x, low, high, color=line.get_color(), alpha=0.12)
        ax.axhline(1.0 - alpha, color=line.get_color(), ls=":", lw=0.9)
        if group[0].get("pac_beta") is not None:
            pac_y = np.array([float(r["pac_split_coverage_mean"]) for r in group])
            ax.plot(
                x,
                pac_y,
                "--s",
                color=line.get_color(),
                label=f"PAC {direction}, alpha={alpha:g}",
            )
    if all(float(r["delta"]) > 0 for r in aggregates) and len(
        {float(r["delta"]) for r in aggregates}
    ) > 1:
        ax.set_xscale("log")
    ax.set_ylim(0.0, 1.01)
    ax.set_xlabel("input perturbation radius delta")
    ax.set_ylabel("held-out empirical coverage")
    ax.set_title(
        "Repeated-split endpoint-amplification coverage\n"
        "bands: 5th–95th percentile across splits; dotted: nominal target"
    )
    ax.text(
        0.01,
        0.01,
        "Split conformal is marginal: individual split coverage may be below target.",
        transform=ax.transAxes,
        fontsize=8,
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    coverage_path = figure_dir / "coverage_repeated_splits.png"
    fig.savefig(coverage_path, dpi=180)
    plt.close(fig)
    written.append(str(coverage_path))

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    for direction, alpha, group in _group_aggregates(aggregates):
        x = np.array([float(r["delta"]) for r in group])
        y = np.array([float(r["ordinary_endpoint_aggregate_bound_median"]) for r in group])
        low = np.array([float(r["ordinary_endpoint_aggregate_bound_q05"]) for r in group])
        high = np.array([float(r["ordinary_endpoint_aggregate_bound_q95"]) for r in group])
        line = ax.plot(x, y, "-o", label=f"ordinary {direction}, alpha={alpha:g}")[0]
        ax.fill_between(x, low, high, color=line.get_color(), alpha=0.12)
        if group[0].get("pac_beta") is not None:
            pac_y = np.array([float(r["pac_endpoint_aggregate_bound_median"]) for r in group])
            ax.plot(
                x,
                pac_y,
                "--s",
                color=line.get_color(),
                label=(
                    f"PAC {direction}, alpha={alpha:g}, "
                    f"beta={float(group[0]['pac_beta']):g}"
                ),
            )
    tolerance_values = {
        float(r["tolerance"]) for r in aggregates if r.get("tolerance") is not None
    }
    if len(tolerance_values) == 1:
        tolerance = next(iter(tolerance_values))
        ax.axhline(tolerance, color="black", ls=":", label=f"task tolerance={tolerance:g}")
    if all(float(r["delta"]) > 0 for r in aggregates) and len(
        {float(r["delta"]) for r in aggregates}
    ) > 1:
        ax.set_xscale("log")
    ax.set_xlabel("input perturbation radius delta")
    ax.set_ylabel("endpoint aggregate bound = delta × calibrated amplification")
    ax.set_title(
        "Calibrated endpoint-amplification bounds\n"
        "lines: split median; bands: 5th–95th percentile"
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    bound_path = figure_dir / "endpoint_bound_repeated_splits.png"
    fig.savefig(bound_path, dpi=180)
    plt.close(fig)
    written.append(str(bound_path))
    return written


def build_provenance(
    checkpoint: Path,
    cfg: Any,
    settings: EvaluationSettings,
    data_provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    checkpoint = Path(checkpoint).expanduser().resolve()
    try:
        from omegaconf import OmegaConf

        resolved_config = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        resolved_config = _json_safe(cfg)

    config_path: Optional[Path] = None
    try:
        from fsbm.utils import get_job_directory

        config_path = Path(get_job_directory(str(checkpoint))) / ".hydra" / "config.yaml"
    except Exception:
        config_path = None

    source_path = Path(__file__).resolve()
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        "hydra_config": {
            "path": str(config_path) if config_path is not None else None,
            "file_sha256": (
                sha256_file(config_path)
                if config_path is not None and config_path.is_file()
                else None
            ),
            "resolved_sha256": sha256_json(resolved_config),
            "resolved": resolved_config,
        },
        "data": data_provenance,
        "data_combined_sha256": sha256_json(data_provenance),
        "evaluation_settings": settings.to_dict(),
        "run_config_sha256": sha256_json(
            {"schema": SCHEMA_VERSION, **settings.to_dict()}
        ),
        "evaluator": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": settings.device,
        },
    }
    return provenance


def write_outputs(
    output_dir: Path,
    split_rows: Sequence[Mapping[str, Any]],
    aggregate_rows: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    *,
    figures: bool = True,
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_split_path = output_dir / "per_split_results.csv"
    aggregate_csv_path = output_dir / "aggregate_results.csv"
    aggregate_json_path = output_dir / "aggregate_results.json"
    _write_csv(per_split_path, split_rows)
    _write_csv(aggregate_csv_path, aggregate_rows)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "claims": {
            "ordinary_marginal_split_conformal": (
                "For a jointly exchangeable calibration/new runtime score, the "
                "ordinary rank threshold gives marginal coverage. The claim is "
                "not that every realized split's empirical coverage exceeds the "
                "nominal target."
            ),
            "fixed_deployment_pac_tolerance": (
                "When pac_beta is set, with probability at least 1-beta over the "
                "calibration sample, the PAC rank threshold has population "
                "coverage at least 1-alpha, under iid continuous scores."
            ),
            "endpoint_bound": (
                "The bound applies to the configured aggregation over exactly "
                "n_probes sampled isotropic perturbations under synchronous SDE "
                "noise; it is not a worst-case Lipschitz bound over all directions."
            ),
        },
        "statistical_diagnostics": {
            "exact_beta_binomial": (
                "Held-out covered-count tails integrate over the random order-"
                "statistic threshold: K ~ BetaBinomial(n_test, rank, "
                "n_calib+1-rank). Repeated-split tails convolve this exact "
                "discrete law. Exactness assumes exchangeability and distinct "
                "scores; threshold ties are reported and generally make coverage "
                "conservative."
            ),
            "fixed_threshold_binomial_diagnostic": (
                "Columns containing this phrase condition away threshold "
                "randomness and compare with a Binomial(n_test, 1-alpha) model. "
                "They are descriptive diagnostics, not the conformal validity test."
            ),
        },
        "provenance": provenance,
        "aggregate_results": [_json_safe(row) for row in aggregate_rows],
    }
    with aggregate_json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)

    paths = {
        "per_split_csv": str(per_split_path),
        "aggregate_csv": str(aggregate_csv_path),
        "aggregate_json": str(aggregate_json_path),
    }
    if figures:
        figure_paths = make_figures(aggregate_rows, output_dir / "figures")
        paths["coverage_figure"] = figure_paths[0]
        paths["bound_figure"] = figure_paths[1]
    return paths


def _parse_csv(value: str, cast: Any) -> Tuple[Any, ...]:
    items = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated non-empty list")
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible repeated split-conformal FSBM endpoint-amplification "
            "evaluation with ordinary marginal and optional PAC guarantees."
        )
    )
    parser.add_argument(
        "--checkpoint",
        "--ckpt_file",
        dest="checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--directions", default="fwd,bwd")
    parser.add_argument("--deltas", "--delta0s", dest="deltas", default="0.05,0.1,0.2")
    parser.add_argument("--alphas", default="0.05,0.1")
    parser.add_argument("--n-splits", "--n_splits", dest="n_splits", type=int, default=10)
    parser.add_argument("--n-calib", "--n_calib", dest="n_calib", type=int, default=128)
    parser.add_argument("--n-test", "--n_test", dest="n_test", type=int, default=256)
    parser.add_argument("--n-probes", "--n_probes", dest="n_probes", type=int, default=4)
    parser.add_argument("--aggregation", choices=sorted(AGGREGATIONS), default="max")
    parser.add_argument("--nfe", type=int, default=50)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--baseline-terminal-error", type=float, default=0.0)
    parser.add_argument("--tolerance", type=float, default=None)
    parser.add_argument(
        "--pac-beta",
        type=float,
        default=None,
        help=(
            "Enable the separate fixed-deployment PAC tolerance guarantee with "
            "calibration failure probability beta."
        ),
    )
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def settings_from_args(args: argparse.Namespace) -> EvaluationSettings:
    return EvaluationSettings(
        directions=_parse_csv(args.directions, str),
        deltas=_parse_csv(args.deltas, float),
        alphas=_parse_csv(args.alphas, float),
        n_splits=args.n_splits,
        n_calib=args.n_calib,
        n_test=args.n_test,
        n_probes=args.n_probes,
        aggregation=args.aggregation,
        nfe=args.nfe,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        baseline_terminal_error=args.baseline_terminal_error,
        tolerance=args.tolerance,
        pac_beta=args.pac_beta,
    )


def _default_output(experiment: str) -> Path:
    stamp = datetime.now().strftime("%Y.%m.%d/%H%M%S")
    return Path("outputs") / "darpa_conformal" / str(experiment) / stamp


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be >= 1")
    torch.set_num_threads(args.torch_threads)
    settings = settings_from_args(args)
    settings.validate()

    # Heavy Lightning/model imports remain CLI-only so the statistical and
    # sampling helpers can be unit-tested quickly.
    from fsbm.utils import restore_model

    model, cfg = restore_model(args.checkpoint, device=settings.device)
    rows, aggregates, data_provenance = evaluate_model(
        model, settings, verbose=not args.quiet
    )
    experiment = getattr(cfg, "name", "fsbm")
    output = args.output if args.output is not None else _default_output(experiment)
    provenance = build_provenance(
        args.checkpoint, cfg, settings, data_provenance
    )
    paths = write_outputs(
        output,
        rows,
        aggregates,
        provenance,
        figures=not args.no_figures,
    )
    print(json.dumps(paths, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
