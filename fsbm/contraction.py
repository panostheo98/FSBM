import copy
import math
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from .sde import paired_sdeint


DEFAULT_CONTRACTION_CFG = {
    "enabled": False,
    "log_on_validation": False,
    "directions": ["fwd", "bwd"],
    "n_calib": 128,
    "alpha": 0.05,
    "delta0s": [0.1],
    "nfe": 200,
    "log_steps": 8,
    "seed": 0,
    "save_scores": False,
    "save_terminal_distances": False,
    "save_histograms": False,
    "hist_bins": 20,
    "verbose": False,
    "jacobian": {
        "enabled": False,
        "num_states": 2,
        "num_times": 2,
    },
    "w2_bound": {
        "enabled": False,
        "num_states": 8,
        "compute_ot": True,
        "quantile_level": 0.9,
    },
    "terminal_certificate": {
        "baseline_terminal_error": 0.0,
        "tau": None,
    },
}


def ensure_contraction_cfg(cfg):
    default_cfg = OmegaConf.create({"contraction": DEFAULT_CONTRACTION_CFG})
    if cfg is None:
        return default_cfg
    return OmegaConf.merge(default_cfg, cfg)


def build_generator(seed: int, device: Optional[torch.device] = None) -> torch.Generator:
    if device is not None and torch.device(device).type == "cuda":
        generator = torch.Generator(device=torch.device(device))
    else:
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def sample_unit_sphere(batch: int, dim: int, device, generator: torch.Generator):
    vec = torch.randn(batch, dim, device=device, generator=generator)
    return vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def conformal_quantile(scores: torch.Tensor, alpha: float) -> Dict[str, object]:
    if scores.ndim != 1:
        raise ValueError("scores must be a 1-D tensor")
    n = scores.numel()
    if n == 0:
        raise ValueError("scores must be non-empty")
    if not 0.0 <= float(alpha) < 1.0:
        raise ValueError("alpha must satisfy 0 <= alpha < 1")

    sorted_scores = torch.sort(scores).values
    order = int(math.ceil((1 - alpha) * (n + 1)))
    # The usual split-conformal construction augments the n finite scores with
    # +infinity.  Clamping an unattainable (n + 1)-st order statistic to the
    # observed maximum silently loses the requested coverage guarantee (notably
    # for alpha=0, or alpha < 1/(n+1)).
    if order > n:
        quantile = math.inf
    else:
        order = max(order, 1)
        quantile = float(sorted_scores[order - 1].item())
    return {
        "order_statistic": order,
        "quantile": quantile,
        "minimum_finite_alpha": float(1.0 / (n + 1)),
        "finite_threshold": bool(math.isfinite(quantile)),
    }


def _binomial_upper_tail(n: int, p: float, first_success: int) -> float:
    """P[Binomial(n, p) >= first_success], evaluated in log space."""
    if first_success <= 0:
        return 1.0
    if first_success > n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    logs = [
        (
            math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
            + k * math.log(p)
            + (n - k) * math.log1p(-p)
        )
        for k in range(first_success, n + 1)
    ]
    max_log = max(logs)
    return float(math.exp(max_log) * sum(math.exp(value - max_log) for value in logs))


def pac_tolerance_quantile(
    scores: torch.Tensor,
    alpha: float,
    beta: float,
) -> Dict[str, object]:
    """Distribution-free one-sided tolerance threshold.

    Unlike ordinary split conformal's *marginal* guarantee, this chooses a more
    conservative order statistic such that, with probability at least
    ``1 - beta`` over the calibration sample, the fixed deployed threshold
    covers at least ``1 - alpha`` of the score population.

    If the calibration sample is too small to make that statement with a finite
    observed order statistic, the valid threshold is ``+infinity``.
    """
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a non-empty 1-D tensor")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must satisfy 0 < alpha < 1")
    if not 0.0 < float(beta) < 1.0:
        raise ValueError("beta must satisfy 0 < beta < 1")

    n = int(scores.numel())
    population_coverage = 1.0 - float(alpha)
    lo, hi = 1, n
    if _binomial_upper_tail(n, population_coverage, n) > float(beta):
        order = n + 1
        quantile = math.inf
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if _binomial_upper_tail(n, population_coverage, mid) <= float(beta):
                hi = mid
            else:
                lo = mid + 1
        order = lo
        quantile = float(torch.sort(scores).values[order - 1].item())

    return {
        "order_statistic": int(order),
        "quantile": quantile,
        "population_miscoverage": float(alpha),
        "calibration_failure_probability": float(beta),
        "population_coverage": population_coverage,
        "calibration_confidence": 1.0 - float(beta),
        "finite_threshold": bool(math.isfinite(quantile)),
    }


def build_terminal_certificate(
    *,
    delta0: float,
    alpha: float,
    n_calib: int,
    conformal: Dict[str, float],
    baseline_terminal_error: float = 0.0,
    tau: Optional[float] = None,
) -> Dict[str, object]:
    s_hat_alpha = float(conformal["quantile"])
    terminal_distance_bound = float(s_hat_alpha * float(delta0))
    baseline_terminal_error = float(baseline_terminal_error)
    predicted_total_error_bound = baseline_terminal_error + terminal_distance_bound

    certificate = {
        "delta0": float(delta0),
        "n_calib": int(n_calib),
        "miscoverage": float(alpha),
        "coverage": float(1.0 - alpha),
        "order_statistic": int(conformal["order_statistic"]),
        "s_hat_alpha": s_hat_alpha,
        "terminal_distance_bound": terminal_distance_bound,
        "baseline_terminal_error": baseline_terminal_error,
        "predicted_total_error_bound": float(predicted_total_error_bound),
    }

    if tau is not None:
        tau = float(tau)
        if s_hat_alpha <= 0:
            max_delta0 = math.inf if tau >= baseline_terminal_error else 0.0
        else:
            max_delta0 = max((tau - baseline_terminal_error) / s_hat_alpha, 0.0)
        certificate.update(
            {
                "tau": tau,
                "max_certified_delta0_for_tau": float(max_delta0),
                "within_tau": bool(predicted_total_error_bound <= tau + 1e-12),
            }
        )

    return certificate


def summarize_tensor(values: torch.Tensor) -> Dict[str, float]:
    if values.ndim != 1:
        values = values.reshape(-1)
    return {
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def _histogram(values: torch.Tensor, bins: int) -> Dict[str, List[float]]:
    counts, edges = np.histogram(values.detach().cpu().numpy(), bins=bins)
    return {
        "counts": counts.astype(int).tolist(),
        "bin_edges": edges.astype(float).tolist(),
    }


def _select_indices(size: int, count: int) -> List[int]:
    if count <= 0 or size <= 0:
        return []
    if count >= size:
        return list(range(size))
    return np.linspace(0, size - 1, count).round().astype(int).tolist()


def sample_reference(reference_sampler, n: int, seed: int) -> torch.Tensor:
    """Draw a reproducible sample without mutating the caller's sampler."""
    sampler = copy.deepcopy(reference_sampler)
    if hasattr(sampler, "set_generator"):
        sampler.set_generator(build_generator(seed))
        samples = sampler(n)
    else:
        # Generic callables may use torch's global RNG and expose no generator.
        # fork_rng restores the caller's RNG state after this deterministic draw.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            samples = sampler(n)
    if not torch.is_tensor(samples):
        samples = torch.as_tensor(samples)
    return samples.float()


# Backward-compatible private name used by older call sites.
_sample_reference = sample_reference


def _integration_order_rollout(rollout: Dict[str, torch.Tensor], direction: str) -> Dict[str, torch.Tensor]:
    if direction == "bwd":
        return {
            "t": rollout["t"].flip(0),
            "xs": rollout["xs"].flip(1),
            "ys": rollout["ys"].flip(1),
            "distances": rollout["distances"].flip(1),
        }
    return {
        "t": rollout["t"],
        "xs": rollout["xs"],
        "ys": rollout["ys"],
        "distances": rollout["distances"],
    }


def _elapsed_time(ts: torch.Tensor) -> torch.Tensor:
    elapsed = torch.zeros_like(ts)
    if ts.numel() > 1:
        elapsed[1:] = torch.cumsum((ts[1:] - ts[:-1]).abs(), dim=0)
    return elapsed


def _cumulative_trapezoid(values: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1 or ts.ndim != 1 or values.shape != ts.shape:
        raise ValueError("values and ts must be 1-D tensors with identical shape")
    integral = torch.zeros_like(values)
    if values.numel() > 1:
        dt = (ts[1:] - ts[:-1]).abs()
        increments = 0.5 * (values[1:] + values[:-1]) * dt
        integral[1:] = torch.cumsum(increments, dim=0)
    return integral


def _build_w2_envelope(
    local_L: torch.Tensor,
    ts: torch.Tensor,
    coupled_sq: torch.Tensor,
    empirical_w2_sq: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    integral_L = _cumulative_trapezoid(local_L, ts)
    exp_integral_L = torch.exp(integral_L)

    coupled_bound = exp_integral_L * coupled_sq[0]
    coupled_ratio = coupled_sq / coupled_bound.clamp_min(1e-12)
    coupled_holds = coupled_sq <= coupled_bound + 1e-6 * coupled_bound.clamp_min(1.0)

    envelope = {
        "local_L": [float(v.item()) for v in local_L],
        "integral_L": [float(v.item()) for v in integral_L],
        "exp_integral_L": [float(v.item()) for v in exp_integral_L],
        "bound_on_coupled_w2_sq_upper": [float(v.item()) for v in coupled_bound],
        "coupled_ratio": [float(v.item()) for v in coupled_ratio],
        "coupled_holds": [bool(v.item()) for v in coupled_holds],
        "all_coupled_holds": bool(coupled_holds.all().item()),
    }

    if empirical_w2_sq is not None:
        empirical_bound = exp_integral_L * empirical_w2_sq[0]
        empirical_ratio = empirical_w2_sq / empirical_bound.clamp_min(1e-12)
        empirical_holds = empirical_w2_sq <= empirical_bound + 1e-6 * empirical_bound.clamp_min(1.0)
        envelope.update(
            {
                "bound_on_empirical_w2_sq": [float(v.item()) for v in empirical_bound],
                "empirical_ratio": [float(v.item()) for v in empirical_ratio],
                "empirical_holds": [bool(v.item()) for v in empirical_holds],
                "all_empirical_holds": bool(empirical_holds.all().item()),
            }
        )

    return envelope


def _empirical_w2_sq(x: torch.Tensor, y: torch.Tensor) -> float:
    try:
        import ot
    except ImportError as exc:
        raise RuntimeError(
            "POT is required for empirical W2^2 verification. Install `pot` or disable compute_ot."
        ) from exc

    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError("x and y must be 2-D tensors with identical shape")

    n = x.shape[0]
    weights = np.full((n,), 1.0 / n, dtype=np.float64)
    cost = ot.dist(
        x.detach().cpu().numpy(),
        y.detach().cpu().numpy(),
        metric="sqeuclidean",
    )
    return float(ot.emd2(weights, weights, cost))


def _symmetric_local_matrix(
    *,
    x: torch.Tensor,
    t_scalar: torch.Tensor,
    state_idx: int,
    drift_fn=None,
    local_matrix_fn=None,
) -> torch.Tensor:
    if local_matrix_fn is not None:
        matrix = local_matrix_fn(state_idx, x, t_scalar)
        return 0.5 * (matrix + matrix.transpose(-1, -2))

    drift_local = (
        drift_fn.single_state_drift(state_idx)
        if hasattr(drift_fn, "single_state_drift")
        else drift_fn
    )

    def drift_single(z):
        tt = torch.full((1,), t_scalar.item(), device=z.device, dtype=z.dtype)
        return drift_local(z.unsqueeze(0), tt).squeeze(0)

    jacobian = torch.autograd.functional.jacobian(drift_single, x)
    return 0.5 * (jacobian + jacobian.transpose(-1, -2))


def compute_local_jacobian_diagnostic(
    drift,
    xs: torch.Tensor,
    ts: torch.Tensor,
    num_states: int,
    num_times: int,
) -> Dict[str, object]:
    state_indices = _select_indices(xs.shape[0], num_states)
    time_indices = _select_indices(xs.shape[1], num_times)
    if not state_indices or not time_indices:
        return {}

    lambda_values = []
    timewise_values = []
    with torch.enable_grad():
        for time_idx in time_indices:
            t_scalar = ts[time_idx].detach()
            local_values = []
            for state_idx in state_indices:
                drift_local = (
                    drift.single_state_drift(state_idx)
                    if hasattr(drift, "single_state_drift")
                    else drift
                )
                x = xs[state_idx, time_idx].detach().clone().requires_grad_(True)

                def drift_single(z):
                    tt = torch.full((1,), t_scalar.item(), device=z.device, dtype=z.dtype)
                    return drift_local(z.unsqueeze(0), tt).squeeze(0)

                jacobian = torch.autograd.functional.jacobian(drift_single, x)
                sym_jacobian = 0.5 * (jacobian + jacobian.transpose(-1, -2))
                lambda_max = torch.linalg.eigvalsh(sym_jacobian).amax()
                local_values.append(lambda_max.detach())
                lambda_values.append(lambda_max.detach())
            timewise_values.append(torch.stack(local_values))

    lambda_values = torch.stack(lambda_values).float()
    timewise_mean = torch.stack([vals.float().mean() for vals in timewise_values])

    return {
        "num_states": len(state_indices),
        "num_times": len(time_indices),
        "state_indices": state_indices,
        "time_indices": time_indices,
        "times": [float(ts[idx].item()) for idx in time_indices],
        "mean_lambda_max_sym": float(lambda_values.mean().item()),
        "median_lambda_max_sym": float(lambda_values.median().item()),
        "max_lambda_max_sym": float(lambda_values.max().item()),
        "mean_local_L": float((2.0 * lambda_values).mean().item()),
        "timewise_mean_lambda_max_sym": [float(v.item()) for v in timewise_mean],
    }


def compute_timewise_w2_bound(
    drift_x,
    xs: torch.Tensor,
    ys: torch.Tensor,
    ts: torch.Tensor,
    num_states: int,
    compute_ot: bool = True,
    quantile_level: float = 0.9,
    drift_y=None,
    local_matrix_x=None,
    local_matrix_y=None,
    local_operator_description: Optional[str] = None,
) -> Dict[str, object]:
    if xs.shape != ys.shape or xs.ndim != 3:
        raise ValueError("xs and ys must have shape (batch, time, dim)")
    if ts.ndim != 1 or ts.shape[0] != xs.shape[1]:
        raise ValueError("ts must have shape (time,)")

    state_indices = _select_indices(xs.shape[0], num_states)
    if not state_indices:
        return {}

    lambda_mean = []
    lambda_max = []
    lambda_quantile = []
    drift_y = drift_x if drift_y is None else drift_y
    with torch.enable_grad():
        for time_idx, t_scalar in enumerate(ts):
            local_values = []
            for states, drift_fn, local_matrix_fn in (
                (xs, drift_x, local_matrix_x),
                (ys, drift_y, local_matrix_y),
            ):
                for state_idx in state_indices:
                    x = states[state_idx, time_idx].detach().clone().requires_grad_(True)
                    sym_matrix = _symmetric_local_matrix(
                        x=x,
                        t_scalar=t_scalar.detach(),
                        state_idx=state_idx,
                        drift_fn=drift_fn,
                        local_matrix_fn=local_matrix_fn,
                    )
                    local_values.append(torch.linalg.eigvalsh(sym_matrix).amax().detach())
            local_values = torch.stack(local_values).float()
            lambda_mean.append(local_values.mean())
            lambda_max.append(local_values.max())
            lambda_quantile.append(torch.quantile(local_values, quantile_level))

    lambda_mean = torch.stack(lambda_mean)
    lambda_max = torch.stack(lambda_max)
    lambda_quantile = torch.stack(lambda_quantile)
    elapsed = _elapsed_time(ts)

    sq_distances = torch.square(torch.linalg.norm(xs - ys, dim=-1))
    coupled_sq = sq_distances.mean(dim=0)
    empirical_w2_sq = None
    if compute_ot:
        empirical_w2_sq = torch.tensor(
            [_empirical_w2_sq(xs[:, idx], ys[:, idx]) for idx in range(xs.shape[1])],
            dtype=xs.dtype,
            device=xs.device,
        )

    sup_envelope = _build_w2_envelope(
        local_L=2.0 * lambda_max,
        ts=ts,
        coupled_sq=coupled_sq,
        empirical_w2_sq=empirical_w2_sq,
    )
    quantile_envelope = _build_w2_envelope(
        local_L=2.0 * lambda_quantile,
        ts=ts,
        coupled_sq=coupled_sq,
        empirical_w2_sq=empirical_w2_sq,
    )

    result = {
        "num_states_per_process": len(state_indices),
        "state_indices": state_indices,
        "region_description": "sampled rollout states from both coupled processes",
        "local_operator_description": (
            local_operator_description
            or "sym(J(deployed drift)) on sampled rollout states"
        ),
        "physical_times": [float(v.item()) for v in ts],
        "elapsed_time": [float(v.item()) for v in elapsed],
        "lambda_max_sym_mean": [float(v.item()) for v in lambda_mean],
        "lambda_max_sym_sup": [float(v.item()) for v in lambda_max],
        "coupled_w2_sq_upper": [float(v.item()) for v in coupled_sq],
        "lambda_max_sym_quantile_level": float(quantile_level),
        "lambda_max_sym_quantile": [float(v.item()) for v in lambda_quantile],
        "envelopes": {
            "sampled_sup": sup_envelope,
            "sampled_quantile": quantile_envelope,
        },
    }

    if empirical_w2_sq is not None:
        result["empirical_w2_sq"] = [float(v.item()) for v in empirical_w2_sq]

    # Backward-compatible aliases: keep the sampled-sup theorem-style envelope at top level.
    result["local_L_sup"] = sup_envelope["local_L"]
    result["integral_L"] = sup_envelope["integral_L"]
    result["exp_integral_L"] = sup_envelope["exp_integral_L"]
    result["bound_on_coupled_w2_sq_upper"] = sup_envelope["bound_on_coupled_w2_sq_upper"]
    result["coupled_w2_sq_upper_holds"] = sup_envelope["coupled_holds"]
    result["all_coupled_w2_sq_upper_holds"] = sup_envelope["all_coupled_holds"]
    if "bound_on_empirical_w2_sq" in sup_envelope:
        result["bound_on_empirical_w2_sq"] = sup_envelope["bound_on_empirical_w2_sq"]
        result["empirical_w2_sq_holds"] = sup_envelope["empirical_holds"]
        result["all_empirical_w2_sq_holds"] = sup_envelope["all_empirical_holds"]

    return result


def evaluate_directional_contraction(
    drift,
    reference_sampler,
    direction: str,
    sigma: float,
    n_calib: int,
    alpha: float,
    delta0s: Iterable[float],
    nfe: int,
    log_steps: int,
    seed: int,
    device,
    local_matrix_factory=None,
    jacobian_cfg=None,
    w2_bound_cfg=None,
    terminal_certificate_cfg=None,
    save_scores: bool = False,
    save_terminal_distances: bool = False,
    save_histograms: bool = False,
    hist_bins: int = 20,
    verbose: bool = False,
) -> Dict[str, object]:
    device = torch.device(device)
    diffusion = lambda x, t: sigma
    delta0s = [float(delta0) for delta0 in delta0s]

    jacobian_cfg = jacobian_cfg or {}
    jacobian_enabled = bool(jacobian_cfg.get("enabled", False))
    w2_bound_cfg = w2_bound_cfg or {}
    w2_bound_enabled = bool(w2_bound_cfg.get("enabled", False))
    terminal_certificate_cfg = terminal_certificate_cfg or {}
    rollout_log_steps = max(
        int(log_steps),
        int(jacobian_cfg.get("num_times", 0)),
        2 if w2_bound_enabled else 1,
    )

    per_delta = []
    rollout_for_jacobian = None
    for delta_idx, delta0 in enumerate(delta0s):
        base_seed = int(seed + 1000 * delta_idx)
        x0 = _sample_reference(reference_sampler, n_calib, base_seed).to(device)
        noise_generator = build_generator(base_seed + 17, device)
        directions = sample_unit_sphere(x0.shape[0], x0.shape[1], device, noise_generator)
        y0 = x0 + delta0 * directions

        rollout = paired_sdeint(
            x0,
            y0,
            drift=drift,
            diffusion=diffusion,
            direction=direction,
            nfe=nfe,
            log_steps=rollout_log_steps,
            verbose=verbose,
            generator=noise_generator,
        )
        if rollout_for_jacobian is None:
            rollout_for_jacobian = rollout

        scores = rollout["amplification"].detach().cpu()
        terminal_distance = rollout["terminal_distance"].detach().cpu()
        conformal = conformal_quantile(scores, alpha)
        summary = {
            "delta0": float(delta0),
            "n_calib": int(n_calib),
            "alpha": float(alpha),
            "conformal": conformal,
            "amplification": summarize_tensor(scores),
            "terminal_distance": summarize_tensor(terminal_distance),
            "initial_distance": summarize_tensor(rollout["initial_distance"].detach().cpu()),
            "terminal_certificate": build_terminal_certificate(
                delta0=float(delta0),
                alpha=float(alpha),
                n_calib=int(n_calib),
                conformal=conformal,
                baseline_terminal_error=float(
                    terminal_certificate_cfg.get("baseline_terminal_error", 0.0)
                ),
                tau=terminal_certificate_cfg.get("tau"),
            ),
        }
        if save_scores:
            summary["scores"] = [float(v) for v in scores.tolist()]
        if save_terminal_distances:
            summary["terminal_distances"] = [float(v) for v in terminal_distance.tolist()]
        if save_histograms:
            summary["amplification_histogram"] = _histogram(scores, hist_bins)
            summary["terminal_distance_histogram"] = _histogram(
                terminal_distance, hist_bins
            )
        if w2_bound_enabled:
            integration_rollout = _integration_order_rollout(rollout, direction)
            local_matrix_x = local_matrix_y = None
            local_operator_description = None
            if local_matrix_factory is not None:
                (
                    local_matrix_x,
                    local_matrix_y,
                    local_operator_description,
                ) = local_matrix_factory(x0, y0)
            summary["w2_bound"] = compute_timewise_w2_bound(
                drift_x=drift,
                xs=integration_rollout["xs"],
                ys=integration_rollout["ys"],
                ts=integration_rollout["t"],
                num_states=int(w2_bound_cfg.get("num_states", 0)),
                compute_ot=bool(w2_bound_cfg.get("compute_ot", True)),
                quantile_level=float(w2_bound_cfg.get("quantile_level", 0.9)),
                local_matrix_x=local_matrix_x,
                local_matrix_y=local_matrix_y,
                local_operator_description=local_operator_description,
            )
        per_delta.append(summary)

    result = {
        "direction": direction,
        "nfe": int(nfe),
        "log_steps": int(rollout_log_steps),
        "alpha": float(alpha),
        "n_calib": int(n_calib),
        "delta0s": delta0s,
        "per_delta": per_delta,
    }

    if jacobian_enabled and rollout_for_jacobian is not None:
        result["jacobian"] = compute_local_jacobian_diagnostic(
            drift=drift,
            xs=rollout_for_jacobian["xs"],
            ts=rollout_for_jacobian["t"],
            num_states=int(jacobian_cfg.get("num_states", 0)),
            num_times=int(jacobian_cfg.get("num_times", 0)),
        )

    return result
