import math
import numpy as np
import torch
from tqdm import trange
from ipdb import set_trace as debug

DIRECTIONS = ["fwd", "bwd"]


def build_basedrift(cfg):
    if cfg.prob.name == "opinion":
        from .opinion import PolarizeDyn

        basedrift = PolarizeDyn(cfg.pdrift)
    else:
        basedrift = ZeroBaseDrift()
    return basedrift


class ZeroBaseDrift(torch.nn.Module):
    def __init__(self):
        super(ZeroBaseDrift, self).__init__()

    def forward(self, xt, t):
        return torch.zeros_like(xt)


def _time_to_index(t: torch.Tensor, T: int) -> int:
    return int((t[0].detach().clamp(0.0, 1.0) * (T - 1)).round().item())


class AnchorGuidancePotential:
    def __init__(self, xt_ks, anchors, tau: float, direction: str):
        assert direction in DIRECTIONS
        if xt_ks is None:
            raise ValueError("xt_ks is required to build the G-guidance potential")

        xt_ks = xt_ks.to(anchors).float()
        anchors = anchors.detach().float()
        _, T_ref, dim = xt_ks.shape
        assert anchors.ndim == 2 and anchors.shape[1] == dim

        anchor_idx = 0 if direction == "fwd" else T_ref - 1
        ref_anchor = xt_ks[:, anchor_idx, :]
        nearest_idx = torch.cdist(anchors, ref_anchor).argmin(dim=1)

        self.ref_paths = xt_ks[nearest_idx]
        self.anchor_distance = (anchors - self.ref_paths[:, anchor_idx, :]).abs()
        self.tau = float(tau)
        self.direction = direction

    def _eval(self, x, t, ref_paths, anchor_distance):
        assert x.shape[0] == ref_paths.shape[0]
        assert t.shape == (x.shape[0],) and torch.allclose(t, t[0] * torch.ones_like(t))

        t_idx = _time_to_index(t, ref_paths.shape[1])
        ref_t = ref_paths[:, t_idx, :].to(x)
        distance = (x - ref_t).abs()
        return self.tau * torch.square(distance - anchor_distance.to(x)).sum(dim=-1)

    def __call__(self, x, t):
        return self._eval(x, t, self.ref_paths, self.anchor_distance)

    def single_state_potential(self, state_idx: int):
        ref_paths = self.ref_paths[state_idx : state_idx + 1]
        anchor_distance = self.anchor_distance[state_idx : state_idx + 1]

        def potential(z, t):
            return self._eval(z, t, ref_paths, anchor_distance)

        return potential


def build_g_guidance_potential(xt_ks, anchors, tau: float, direction: str):
    """
    Build an anchor-conditioned Markov slice of the pathwise G potential.

    The training-time G depends on the whole trajectory and its anchor time.
    For evaluation-time diagnostics we keep the anchor term fixed at the rollout
    anchor, choose a nearest reference path in xt_ks, and evaluate the
    corresponding time-slice potential G_t(x).
    """
    return AnchorGuidancePotential(xt_ks, anchors, tau, direction)


def _assert_increasing(ts: torch.Tensor) -> None:
    assert (ts[1:] > ts[:-1]).all(), "time must be strictly increasing"


def _build_timesteps(direction, T, eps, device):
    assert direction in DIRECTIONS
    if direction == "fwd":
        return torch.linspace(eps, 1 - eps, T, device=device)
    return torch.linspace(1 - eps, eps, T, device=device)


def sdeint(
    xinit,
    drift,
    diffusion,
    direction,
    nfe,
    log_steps=5,
    eps=0,
    verbose=False,
    return_ws=False,
):
    """
    xinit: (B, D)
    drift:     (B, D) + (B,) --> (B, D)
    diffusion: (B, D) + (B,) --> (B, D)
    nfe = T - 1
    ===
    t: (S,)
    xs: (B, S, D)
    us: (B, S, D)
    """
    assert direction in DIRECTIONS

    T, (B, D), S, device = nfe + 1, xinit.shape, log_steps, xinit.device

    # build ts
    timesteps = _build_timesteps(direction, T, eps, device)

    # logging
    steps_to_log = np.linspace(0, T - 1, log_steps).astype(int)
    xs = []
    us = []
    if return_ws:
        ws = []

    x = xinit.detach()
    bar = trange(nfe) if verbose else range(nfe)
    for idx in bar:
        t, tnxt = timesteps[idx], timesteps[idx + 1]
        dt = (tnxt - t).abs()
        dw = math.sqrt(dt) * torch.randn(*x.shape, device=x.device)

        tt = t.repeat(x.shape[0])
        u = drift(x, tt)
        g = diffusion(x, tt)

        if idx in steps_to_log:
            xs.append(x)
            us.append(u)
            if return_ws:
                ws.append(dw)

        x = x + u * dt + g * dw

    assert len(xs) == len(us) == S - 1

    # log last step, u = 0
    xs.append(x)
    us.append(torch.zeros_like(x))
    if return_ws:
        ws.append(torch.zeros_like(x))
    ts = timesteps[steps_to_log]

    if direction == "bwd":
        xs = xs[::-1]
        us = us[::-1]
        ts = ts.flip(
            dims=[
                0,
            ]
        )
        if return_ws:
            ws = ws[::-1]

    xs = torch.stack(xs, dim=1).detach()
    us = torch.stack(us, dim=1).detach()
    assert xs.shape == us.shape == (B, S, D) and ts.shape == (S,)
    _assert_increasing(ts)
    out = {"t": ts, "xs": xs, "us": us}
    if return_ws:
        ws = torch.stack(ws, dim=1).detach()
        assert ws.shape == us.shape
        out["ws"] = ws
    return out


def paired_sdeint(
    xinit,
    yinit,
    drift,
    diffusion,
    direction,
    nfe,
    log_steps=5,
    eps=0,
    verbose=False,
    generator=None,
    return_ws=False,
    drift_y=None,
):
    """
    Roll out two coupled trajectories with identical Brownian increments.

    xinit: (B, D)
    yinit: (B, D)
    drift:     (B, D) + (B,) --> (B, D)
    diffusion: (B, D) + (B,) --> scalar or (B, D)
    ===
    t:                  (S,)
    xs / ys:            (B, S, D)
    distances:          (B, S)
    initial_distance:   (B,)
    terminal_distance:  (B,)
    amplification:      (B,)
    """
    assert direction in DIRECTIONS
    assert xinit.shape == yinit.shape
    drift_y = drift if drift_y is None else drift_y

    T, (B, D), S, device = nfe + 1, xinit.shape, log_steps, xinit.device
    timesteps = _build_timesteps(direction, T, eps, device)

    steps_to_log = np.linspace(0, T - 1, log_steps).astype(int)
    xs = []
    ys = []
    if return_ws:
        ws = []

    x = xinit.detach()
    y = yinit.detach()
    initial_distance = torch.linalg.norm(x - y, dim=-1)

    bar = trange(nfe) if verbose else range(nfe)
    for idx in bar:
        t, tnxt = timesteps[idx], timesteps[idx + 1]
        dt = (tnxt - t).abs()
        dw = math.sqrt(dt) * torch.randn(
            *x.shape, device=x.device, generator=generator
        )

        tt = t.repeat(x.shape[0])
        ux = drift(x, tt)
        uy = drift_y(y, tt)
        g = diffusion(x, tt)

        if idx in steps_to_log:
            xs.append(x)
            ys.append(y)
            if return_ws:
                ws.append(dw)

        x = x + ux * dt + g * dw
        y = y + uy * dt + g * dw

    assert len(xs) == len(ys) == S - 1

    xs.append(x)
    ys.append(y)
    if return_ws:
        ws.append(torch.zeros_like(x))

    ts = timesteps[steps_to_log]
    if direction == "bwd":
        xs = xs[::-1]
        ys = ys[::-1]
        ts = ts.flip(dims=[0])
        if return_ws:
            ws = ws[::-1]

    xs = torch.stack(xs, dim=1).detach()
    ys = torch.stack(ys, dim=1).detach()
    distances = torch.linalg.norm(xs - ys, dim=-1)
    distances_integration = distances.flip(1) if direction == "bwd" else distances
    ts_integration = ts.flip(0) if direction == "bwd" else ts
    terminal_distance = torch.linalg.norm(x - y, dim=-1)
    amplification = terminal_distance / initial_distance.clamp_min(1e-12)

    assert xs.shape == ys.shape == (B, S, D)
    assert distances.shape == (B, S)
    assert ts.shape == (S,)
    _assert_increasing(ts)

    out = {
        "t": ts,
        "t_integration": ts_integration,
        "xs": xs,
        "ys": ys,
        "distances": distances,
        "distances_integration": distances_integration,
        "initial_distance": initial_distance.detach(),
        "terminal_distance": terminal_distance.detach(),
        "amplification": amplification.detach(),
    }
    if return_ws:
        ws = torch.stack(ws, dim=1).detach()
        assert ws.shape == xs.shape
        out["ws"] = ws
    return out
