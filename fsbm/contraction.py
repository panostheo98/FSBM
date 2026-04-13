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


def conformal_quantile(scores: torch.Tensor, alpha: float) -> Dict[str, float]:
    if scores.ndim != 1:
        raise ValueError("scores must be a 1-D tensor")
    n = scores.numel()
    if n == 0:
        raise ValueError("scores must be non-empty")

    sorted_scores = torch.sort(scores).values
    order = int(math.ceil((1 - alpha) * (n + 1)))
    order = min(max(order, 1), n)
    quantile = sorted_scores[order - 1]
    return {
        "order_statistic": order,
        "quantile": float(quantile.item()),
    }


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


def _sample_reference(reference_sampler, n: int, seed: int) -> torch.Tensor:
    sampler = copy.deepcopy(reference_sampler)
    if hasattr(sampler, "set_generator"):
        sampler.set_generator(build_generator(seed))
    samples = sampler(n)
    if not torch.is_tensor(samples):
        samples = torch.as_tensor(samples)
    return samples.float()


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
                x = xs[state_idx, time_idx].detach().clone().requires_grad_(True)

                def drift_single(z):
                    tt = torch.full((1,), t_scalar.item(), device=z.device, dtype=z.dtype)
                    return drift(z.unsqueeze(0), tt).squeeze(0)

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
    jacobian_cfg=None,
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
    rollout_log_steps = max(int(log_steps), int(jacobian_cfg.get("num_times", 0)), 1)

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
        summary = {
            "delta0": float(delta0),
            "n_calib": int(n_calib),
            "alpha": float(alpha),
            "conformal": conformal_quantile(scores, alpha),
            "amplification": summarize_tensor(scores),
            "terminal_distance": summarize_tensor(terminal_distance),
            "initial_distance": summarize_tensor(rollout["initial_distance"].detach().cpu()),
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

