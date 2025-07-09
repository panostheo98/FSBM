from typing import Union, Dict, Any
import os
from glob import glob
import numpy as np
import importlib
from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt
from torch import distributed as dist
from scipy.optimize import fsolve
from sklearn.metrics.pairwise import pairwise_distances
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F


def get_job_directory(file_or_checkpoint: Union[str, Dict[str, Any]]) -> str:
    found = False
    if isinstance(file_or_checkpoint, dict):
        chkpnt = file_or_checkpoint
        key = [x for x in chkpnt["callbacks"].keys() if "Checkpoint" in x][0]
        file = chkpnt["callbacks"][key]["dirpath"]
    else:
        file = file_or_checkpoint

    hydra_files = []
    directory = os.path.dirname(file)
    while not found:
        hydra_files = glob(
            os.path.join(os.path.join(directory, ".hydra/config.yaml")),
            recursive=True,
        )
        if len(hydra_files) > 0:
            break
        directory = os.path.dirname(directory)
        if directory == "":
            raise ValueError("Failed to find hydra config!")
    assert len(hydra_files) == 1, "Found ambiguous hydra config files!"
    job_dir = os.path.dirname(os.path.dirname(hydra_files[0]))
    return job_dir

def restore_model(checkpoint, pl_name="fsbm.fsbm_model", device=None):
    ckpt = torch.load(checkpoint, map_location="cpu")
    job_dir = get_job_directory(checkpoint)
    cfg = OmegaConf.load(os.path.join(job_dir, ".hydra/config.yaml"))
    # print(f"Loaded cfg from {job_dir=}!")

    from .dataset import get_dist_boundary

    p0, p1, p0_val, p1_val = get_dist_boundary(cfg)
    fsbm_model = importlib.import_module(f"{pl_name}")
    xt_ks = None
    model = fsbm_model.FSBMLitModule(cfg, p0, p1, p0_val, p1_val, xt_ks, tau=200, guidance=True)
    model.load_state_dict(ckpt["state_dict"])

    if device is not None:
        model = model.to(device)

    return model, cfg

def get_repo_path():
    curr_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    return curr_dir.parent


def all_gather(tensor: torch.Tensor):
    gathered_tensor = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    with torch.no_grad():
        dist.all_gather(gathered_tensor, tensor)
    gathered_tensor = torch.cat(gathered_tensor, dim=0)
    return gathered_tensor


def gather(outputs, key):
    return torch.cat([o[key].detach().cpu() for o in outputs], dim=0)

def linear_interp1d(t, xt, mask, s):
    """Linear splines.
    B: batch, T: timestep, D: dim, S: query timestep
    Inputs:
        t: (T, B)
        xt: (T, B, D)
        mask: (T, B)
        s: (S, B)
    Outputs:
        xs: (S, B, D)
    """
    T, N, D = xt.shape
    S = s.shape[0]

    if mask is None:
        mask = torch.ones_like(t).bool()

    m = (xt[1:] - xt[:-1]) / (t[1:] - t[:-1] + 1e-10).unsqueeze(-1)

    left = torch.searchsorted(t[1:].T.contiguous(), s.T.contiguous(), side="left").T
    mask_l = F.one_hot(left, T).permute(0, 2, 1).reshape(S, T, N, 1)

    t = t.reshape(1, T, N, 1)
    xt = xt.reshape(1, T, N, D)
    m = m.reshape(1, T - 1, N, D)
    s = s.reshape(S, N, 1)

    x0 = torch.sum(t * mask_l, dim=1)
    p0 = torch.sum(xt * mask_l, dim=1)
    m0 = torch.sum(m * mask_l[:, :-1], dim=1)

    t = s - x0

    return t * m0 + p0


def cubic_interp1d(t, xt, mask, s):
    """
    Inputs:
        t: (T, N)
        xt: (T, N, D)
        mask: (T, N)
        s: (S, N)
    """
    T, N, D = xt.shape
    S = s.shape[0]

    if t.shape == s.shape:
        if torch.linalg.norm(t - s) == 0:
            return xt

    if mask is None:
        mask = torch.ones_like(t).bool()

    mask = mask.unsqueeze(-1)

    fd = (xt[1:] - xt[:-1]) / (t[1:] - t[:-1] + 1e-10).unsqueeze(-1)
    # Set tangents for the interior points.
    m = torch.cat([(fd[1:] + fd[:-1]) / 2, torch.zeros_like(fd[0:1])], dim=0)
    # Set tangent for the right end point.
    m = torch.where(torch.cat([mask[2:], torch.zeros_like(mask[0:1])]), m, fd)
    # Set tangent for the left end point.
    m = torch.cat([fd[[0]], m], dim=0)

    mask = mask.squeeze(-1)

    left = torch.searchsorted(t[1:].T.contiguous(), s.T.contiguous(), side="left").T
    right = (left + 1) % mask.sum(0).long()
    mask_l = F.one_hot(left, T).permute(0, 2, 1).reshape(S, T, N, 1)
    mask_r = F.one_hot(right, T).permute(0, 2, 1).reshape(S, T, N, 1)

    t = t.reshape(1, T, N, 1)
    xt = xt.reshape(1, T, N, D)
    m = m.reshape(1, T, N, D)
    s = s.reshape(S, N, 1)

    x0 = torch.sum(t * mask_l, dim=1)
    x1 = torch.sum(t * mask_r, dim=1)
    p0 = torch.sum(xt * mask_l, dim=1)
    p1 = torch.sum(xt * mask_r, dim=1)
    m0 = torch.sum(m * mask_l, dim=1)
    m1 = torch.sum(m * mask_r, dim=1)

    dx = x1 - x0
    t = (s - x0) / (dx + 1e-10)

    return (
        t**3 * (2 * p0 + m0 - 2 * p1 + m1)
        + t**2 * (-3 * p0 + 3 * p1 - 2 * m0 - m1)
        + t * m0
        + p0
    )


def lin_int(x, y, t):
    Ks, D = x.shape
    time = torch.linspace(0, 1, t).unsqueeze(0).unsqueeze(2).expand(Ks, t, D).to('cuda')
    x = x.to('cuda')
    y = y.to('cuda')
    xt_ks = x.unsqueeze(1) + (y - x).unsqueeze(1) * time
    xt_ks = xt_ks.to('cuda')
    return xt_ks


def mmd(x, y):
    Kxx = pairwise_distances(x, x)
    Kyy = pairwise_distances(y, y)
    Kxy = pairwise_distances(x, y)

    m = x.shape[0]
    n = y.shape[0]

    c1 = 1 / (m * (m - 1))
    A = np.sum(Kxx - np.diag(np.diagonal(Kxx)))

    # Term II
    c2 = 1 / (n * (n - 1))
    B = np.sum(Kyy - np.diag(np.diagonal(Kyy)))

    # Term III
    c3 = 1 / (m * n)
    C = np.sum(Kxy)

    # estimate MMD
    mmd_est = -0.5 * c1 * A - 0.5 * c2 * B + c3 * C
    return mmd_est

class NumpyFolder(Dataset):
    def __init__(self, folder, resize):
        files_grabbed = []
        for tt in ("*.png", "*.jpg"):
            files_grabbed.extend(glob.glob(os.path.join(folder, tt)))
        files_grabbed = sorted(files_grabbed)

        self.img_paths = files_grabbed

        self.transform = transforms.Compose(
            [
                transforms.Resize(resize),
            ]
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image_set = Image.open(self.img_paths[idx])
        image_set = self.transform(image_set)
        image_tensor = np.asarray(image_set)
        return image_tensor
