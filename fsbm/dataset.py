import os
import math
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from fsbm.utils import get_repo_path


def get_sampler(p, gen=None, **kwargs):
    name = p.name

    if name == "gaussian":
        return Gaussian(p.mu, p.var, generator=gen)
    elif name == "opinion":
        return Opinion(p.dim, p.mu, p.var, p.get("var_1st_dim", None), generator=gen)
    elif name == "ffhq_gen":
        dataset = FFHQ_Gender(genders=p.genders, **kwargs)
        return Sampler(dataset)
    elif name == 'ffhq_age':
        dataset = FFHQ_Age(ages=p.ages, **kwargs)
        return Sampler(dataset)
    else:
        raise ValueError(f"Unknown distribution option: {name}")


def get_dist_boundary(cfg):
    p0 = get_sampler(cfg.prob.p0)
    p1 = get_sampler(cfg.prob.p1)
    p0_val = get_sampler(cfg.prob.p0, split="val")
    p1_val = get_sampler(cfg.prob.p1, split="val")
    return p0, p1, p0_val, p1_val


class FFHQ_Gender(Dataset):
    genders = ["male", "female"]

    def __init__(self, genders, split="train"):
        assert split in ("train", "val")
        latents_file = np.load(str(get_repo_path())+"/data/latents.npy")
        gender_file = np.load(str(get_repo_path())+"/data/gender.npy")
        size = 60000
        if split == 'train':
            latents, gen = latents_file[:size], gender_file[:size]
        else:
            latents, gen = latents_file[size:], gender_file[size:]
            size = 10000

        x_inds = np.arange(size)[(gen == genders).reshape(-1)]
        self.x_data = torch.tensor(latents[x_inds])

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, idx):
        latent_vec = self.x_data[idx]
        return latent_vec


class FFHQ_Age(Dataset):
    ages = ["adult", "child"]

    def __init__(self, ages, split="train"):
        assert split in ("train", "val")
        latents_file = np.load(str(get_repo_path())+"/data/latents.npy")
        age_file = np.load(str(get_repo_path())+"/data/age.npy")
        size = 60000
        if split == 'train':
            latents, age = latents_file[:size], age_file[:size]
        else:
            latents, age = latents_file[size:], age_file[size:]
            size = 10000
        x_inds = None
        if ages == ['adult']:
            x_inds = np.arange(size)[(age >= 35).reshape(-1) * (age != -1).reshape(-1)]
        elif ages == ['child']:
            x_inds = np.arange(size)[(age < 15).reshape(-1) * (age != -1).reshape(-1)]

        self.x_data = torch.tensor(latents[x_inds])

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, idx):
        latent_vec = self.x_data[idx]
        return latent_vec


class Sampler:
    def __init__(self, dataset, generator=None):
        self.dataset = dataset
        self.generator = generator

    def set_generator(self, generator):
        self.generator = generator

    def __call__(self, n):
        ii = torch.randint(0, len(self.dataset), (n,), generator=self.generator)
        out = torch.stack([self.dataset[i] for i in ii], dim=0)
        return out.reshape(n, -1)


class Gaussian:
    def __init__(self, mu, var, generator):
        self.mean = torch.tensor(mu).float()
        self.std = torch.tensor(var).sqrt()
        self.generator = generator

    def set_generator(self, generator):
        self.generator = generator

    def __call__(self, n):
        noise_shape = (n,) + self.mean.shape
        return (
            torch.randn(*noise_shape, generator=self.generator).to(self.mean) * self.std
            + self.mean
        )


class Opinion(Gaussian):
    def __init__(self, dim, mu, var, var_1st_dim, generator=None):
        mu = mu * torch.ones(dim)
        var = var * torch.ones(dim)
        if var_1st_dim is not None:
            var[0] = var_1st_dim
        super(Opinion, self).__init__(mu, var, generator)


class PairDataset(Dataset):
    def __init__(self, x0, x1, expand_factor=1):
        assert len(x0) == len(x1)
        self.x0 = x0
        self.x1 = x1
        self.expand_factor = expand_factor

    def __len__(self):
        return len(self.x0) * self.expand_factor

    def __getitem__(self, idx):
        return {"x0": self.x0[idx % len(self.x0)], "x1": self.x1[idx % len(self.x0)]}


class SplineDataset(Dataset):
    def __init__(self, mean_t, mean_xt, gamma_s, gamma_xs, expand_factor=1):
        """
        mean_t: (T,)
        mean_xt: (B, T, D)
        gamma_t: (S,)
        gamma_xt: (B, S, 1)
        """
        (B, T, D), (S,) = mean_xt.shape, gamma_s.shape
        assert T > 3 and S > 3
        assert mean_t.shape == (T,)
        assert gamma_xs.shape == (B, S, 1)

        self.mean_t = mean_t.detach().cpu().clone()
        self.mean_xt = mean_xt.detach().cpu().clone()
        self.gamma_s = gamma_s.detach().cpu().clone()
        self.gamma_xs = gamma_xs.detach().cpu().clone()

        self.expand_factor = expand_factor

    def __len__(self):
        return self.mean_xt.shape[0] * self.expand_factor

    def __getitem__(self, idx):
        _idx = idx % self.mean_xt.shape[0]

        x0 = self.mean_xt[_idx, 0]
        x1 = self.mean_xt[_idx, -1]
        mean_xt = self.mean_xt[_idx]
        gamma_xs = self.gamma_xs[_idx]

        return {
            "x0": x0,
            "x1": x1,
            "mean_t": self.mean_t,
            "mean_xt": mean_xt,
            "gamma_s": self.gamma_s,
            "gamma_xs": gamma_xs,
        }