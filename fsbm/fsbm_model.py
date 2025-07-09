from typing import Any, List
import os
import numpy as np
import math
from datetime import datetime
from rich.console import Console
from easydict import EasyDict as edict
import copy
import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.utils as tu

from .network import build_net
from .sde import build_basedrift, sdeint

from .dataset import PairDataset, SplineDataset
from . import gaussian_path as gpath_lib
from . import utils

from .plotting import (
    save_fig,
    save_xs,
    plot_gpath,
    plot_boundaries,
    plot_xs_opinion,
)

# put gc.collect after io writing to prevent c10::CUDAError in multi-threading
# https://github.com/pytorch/pytorch/issues/67978#issuecomment-1661986812
import gc
from ipdb import set_trace as debug

console = Console()


class FSBMLitModule(pl.LightningModule):
    def __init__(self, cfg, p0, p1, p0_val, p1_val, xt_ks, tau=100, guidance=False):
        super().__init__()

        os.makedirs("figs", exist_ok=True)

        self.cfg = cfg
        self.p0 = p0
        self.p1 = p1
        self.p0_val = p0_val
        self.p1_val = p1_val

        ### Problem
        self.sigma = cfg.prob.sigma
        self.basedrift = build_basedrift(cfg)

        ### SB Model
        self.direction = None
        self.fwd_net = build_net(cfg)
        self.bwd_net = build_net(cfg)

        self.xt_ks = xt_ks
        self.guidance = guidance
        self.tau = tau

    def print(self, content, prefix=True):
        if self.trainer.is_global_zero:
            if prefix:
                now = f"[[cyan]{datetime.now():%Y-%m-%d %H:%M:%S}[/cyan]]"
                if self.direction is None:
                    base = f"[[blue]Init[/blue]] "
                else:
                    base = f"[[blue]Ep {self.current_epoch} ({self.direction})[/blue]] "
                console.print(now, highlight=False, end="")
                console.print(base, end="")
            console.print(f"{content}")

    @property
    def logging_batch_idxs(self):
        return np.linspace(0, self.trainer.num_training_batches - 1, 10).astype(int)

    @property
    def net(self):
        return self.fwd_net if self.direction == "fwd" else self.bwd_net

    @property
    def direction_r(self):
        return "bwd" if self.direction == "fwd" else "fwd"
    
    @staticmethod
    def bm_loss(drift, xt, t, vt):
        pred_vt = drift(xt, t)
        # print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
        # print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
        assert pred_vt.shape == vt.shape == xt.shape
        return torch.square(pred_vt - vt).mean()

    def build_ft(self, direction):
        def ft(x, t):
            """
            x: (B, D)
            t: (B,)
            ===
            out: (B, D)
            """
            B, D = x.shape
            sign = 1.0 if direction == "fwd" else -1.0
            assert t.shape == (B,) and torch.allclose(t, t[0] * torch.ones_like(t))
            return sign * self.basedrift(x.unsqueeze(1), t[0].reshape(1)).squeeze(1)

        return ft

    def build_ut(self, direction, backprop_snet=False):
        """
        ut: x: (B, D), t: (B,) --> (B, D)
        """
        net = self.fwd_net if direction == "fwd" else self.bwd_net
        if self.cfg.field == "vector":
            ut = net
        elif self.cfg.field == "potential":

            def ut(x, t):
                with torch.enable_grad():
                    x = x.detach().clone()
                    x.requires_grad_(True)
                    out = net(x, t)
                    return torch.autograd.grad(
                        out.sum(), x, create_graph=backprop_snet
                    )[0]

        else:
            ValueError(f"Unsupportted field: {self.cfg.field}!")
        return ut

    def build_drift(self, direction, backprop_snet=False):
        ft = self.build_ft(direction)
        ut = self.build_ut(direction, backprop_snet=backprop_snet)
        drift = lambda x, t: ut(x, t) + ft(x, t)
        return drift

    @torch.no_grad()
    def sample(self, xinit, log_steps, direction, drift=None, nfe=None, verbose=False):
        drift = self.build_drift(direction) if drift is None else drift
        diffusion = lambda x, t: self.sigma
        nfe = nfe or self.cfg.nfe
        output = sdeint(
            xinit,
            drift,
            diffusion,
            direction,
            nfe=nfe,
            log_steps=log_steps,
            verbose=verbose,
        )
        return output

    def sample_t(self, batch):
        eps = 1e-4
        t = torch.rand(batch).reshape(-1) * (1 - 2 * eps) + eps
        assert t.shape == (batch,)
        return t

    def sample_gpath(self, batch):
        ### Setup
        gpath = gpath_lib.EndPointGaussianPath(
            batch["mean_t"][0],
            batch["mean_xt"],
            batch["gamma_s"][0],
            batch["gamma_xs"],
            self.sigma,
            self.basedrift,
        )
        x0, x1 = batch["x0"], batch["x1"]
        B, D = x0.shape

        ### Sample t and xt
        T = B
        t = self.sample_t(T).to(x0)
        with torch.no_grad():
            xt = gpath.sample_xt(t, N=1)

        assert t.shape == (T,) and xt.shape == (B, 1, T, D)
        assert B == T
        vt = gpath.ut(t, xt, self.direction)
        xt = xt[torch.arange(B), 0, torch.arange(B)]
        vt = vt[torch.arange(B), 0, torch.arange(B)]
        assert xt.shape == vt.shape == (B, D)

        return x0, x1, t, xt, vt

    def training_step(self, batch: Any, batch_idx: int):
        ### Sample from Gaussian path
        x0, x1, t, xt, vt = self.sample_gpath(batch)
        ### Apply bridge matching orr entropic action matching
        ut = self.build_ut(self.direction, backprop_snet=True)
        loss = self.bm_loss(ut, xt, t, vt)

        if torch.isfinite(loss):
            self.log("train/loss", loss, on_step=True, on_epoch=True)
        else:
            ### Skip step if loss is NaN.
            self.print(f"Skipping iteration because loss is {loss.item()}.")
            return None

        if batch_idx in self.logging_batch_idxs:
            self.print(
                f"[M-step] batch idx: {batch_idx + 1}/{self.trainer.num_training_batches} ..."
            )

        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        ### ** The only place where we modify the direction!! **
        self.direction = self.direction_r
        self.print("", prefix=False)  # change line

    def localize(self, p):
        g = torch.Generator()
        g.manual_seed(g.seed() + self.global_rank)
        local_p = copy.deepcopy(p)
        local_p.set_generator(g)
        return local_p

    def val_dataloader(self):
        totalB, n_device = self.cfg.csoc.B, 1
        B = totalB // n_device
        self.print(f"[Data] Building {totalB} train_data ...")
        self.print(
            f"[Data] Found {n_device} devices, each will generate {B} samples ..."
        )

        x0 = self.localize(self.p0)(B)
        x1 = self.localize(self.p1)(B)
        return DataLoader(
            PairDataset(x0, x1),
            num_workers=self.cfg.optim.num_workers,
            batch_size=self.cfg.csoc.mB,
            persistent_workers=self.cfg.optim.num_workers > 0,
            shuffle=False,
            pin_memory=True,
        )

    def compute_coupling(self, batch, direction, eval_coupling):
        x0, x1, T = batch["x0"], batch["x1"], self.cfg.csoc.T_mean
        if direction is None:
            t = torch.linspace(0, 1, T).to(x0)
            xt = (1 - t[None, :, None]) * x0.unsqueeze(1) + t[
                                                            None, :, None
                                                            ] * x1.unsqueeze(1)
        else:
            xinit = x0 if direction == "fwd" else x1
            output = self.sample(xinit, log_steps=T, direction=direction)
            t, xt = output["t"], output["xs"]

            if eval_coupling:
                metrics = self.evaluator(output)
                for k, v in metrics.items():
                    self.log(f"metrics/{k}", v, on_epoch=True)

        return t, xt

    def validation_step(self, batch: Any, batch_idx: int):
        log_step = batch_idx == 0
        ccfg, direction = self.cfg.csoc, self.direction
        postfix = f"{self.current_epoch:03d}" if direction is not None else "init"

        (B, D), T, S, sigma = batch["x0"].shape, ccfg.T_mean, ccfg.T_gamma, self.sigma
        ### Initialize mean spline (with copuling)
        eval_coupling = log_step and self.cfg.eval_coupling
        self.print(f"[R-step] Simulating {direction or 'init'} coupling ...")
        t, xt = self.compute_coupling(batch, direction, eval_coupling)
        self.print(f"[R-step] Simulated {xt.shape=}!")
        if log_step:
            self.log_coupling(t, xt, direction, f"coupling-{postfix}")
        assert xt.shape == (B, T, D) and t.shape == (T,)

        ### Initialize std spline
        s = torch.linspace(0, 1, S).to(t)
        ys = torch.zeros(B, S, 1).to(xt)

        ### Fit Gaussian paths (update xt, ys)
        gpath = gpath_lib.EndPointGaussianPath(t, xt, s, ys, sigma, self.basedrift)
        loss_fn = gpath_lib.build_loss_fn(sigma, ccfg, self.tau, self.xt_ks)
        with torch.enable_grad():
            verbose = log_step and self.trainer.is_global_zero
            result = gpath_lib.fit(
                ccfg, gpath, direction or "fwd", loss_fn, verbose=verbose
            )

        self.print(f"[R-step] Fit {B} gaussian paths!")
        if log_step:
            self.log_gpath(result, f"gpath-{postfix}")

        ### Built output
        xt = gpath.mean.xt.detach().clone()
        ys = gpath.gamma.xt.detach().clone()
        assert xt.shape == (B, T, D) and ys.shape == (B, S, 1)
        output = {"mean_t": t, "mean_xt": xt, "gamma_s": s, "gamma_xs": ys}

        ### (optional) Handle opinion drift
        if ccfg.name == "opinion":
            tt = torch.linspace(0, 1, self.cfg.pdrift.S).to(t)
            mf_x = gpath.sample_xt(tt, N=1).squeeze(1)
            assert mf_x.shape == (B, len(tt), D)
            output["mf_x"] = mf_x.detach().cpu()

        return output

    def validation_epoch_end(self, outputs: List[Any]):
        ### Handle opinion drift
        if self.cfg.prob.name == "opinion":
            mf_xs = utils.gather(outputs, "mf_x")
            self.basedrift.set_mf_drift(mf_xs)
            self.print(f"[Opinion] Set MF drift shape={mf_xs.shape}!")

        ccfg = self.cfg.csoc
        T, S, D = ccfg.T_mean, ccfg.T_gamma, 512 if self.cfg.name == 'ffhq_gen.yaml' else self.cfg.dim

        ## gather mean_t, gamma_s
        mean_t = outputs[0]["mean_t"].detach().cpu()
        gamma_s = outputs[0]["gamma_s"].detach().cpu()
        assert mean_t.shape == (T,) and gamma_s.shape == (S,)

        ## gather mean_xt, gamma_xs
        mean_xt = utils.gather(outputs, "mean_xt")
        gamma_xs = utils.gather(outputs, "gamma_xs")
        B = mean_xt.shape[0]
        assert mean_xt.shape == (B, T, D)
        assert gamma_xs.shape == (B, S, 1)

        self.train_data = SplineDataset(
            mean_t, mean_xt, gamma_s, gamma_xs, expand_factor=ccfg.epd_fct
        )
        self.print(f"[Data] Fit total {B} gaussian paths as train_data!")

        if self.direction is None:
            self.direction = "fwd"
            self.print("", prefix=False)  # change line

        torch.cuda.empty_cache()

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_data,
            num_workers=self.cfg.optim.num_workers,
            batch_size=self.cfg.optim.batch_size,
            persistent_workers=self.cfg.optim.num_workers > 0,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        return dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.wd,
            eps=self.cfg.optim.eps,
        )

        if self.cfg.optim.get("scheduler", "cosine") == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.optim.num_iterations,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        else:
            return {
                "optimizer": optimizer,
            }

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.net.update_ema()

    def log_coupling(self, t, xs, direction, fn, log_steps=5):
        """
        t: (T,) xs: (B, T, D)
        """
        B, T, D = xs.shape
        assert t.shape == (T,)
        save_xs(t, xs, log_steps, direction, self.cfg.plot, fn)
        gc.collect()

    def log_gpath(self, result, fn):
        plot_gpath(result, self.cfg.plot)
        save_fig(fn)
        gc.collect()

    def log_boundary(self, p0, p1, p0_val, p1_val):
        plot_boundaries(p0, p1, self.cfg.plot)
        save_fig("train_dist")
        plot_boundaries(p0_val, p1_val, self.cfg.plot)
        save_fig("val_dist")
        gc.collect()

    def log_basedrift(self, p0):
        ft = self.build_ft("fwd")
        result = self.sample(p0(512), log_steps=5, direction="fwd", drift=ft, nfe=500)
        plot_xs_opinion(result["t"], result["xs"], 5, "Init", self.cfg.plot)
        save_fig("ft")
