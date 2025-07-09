

from functools import partial
from scipy.spatial import cKDTree
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as td
from .utils import get_repo_path

from ipdb import set_trace as debug



def obstacle_cfg_stunnel():
    a, b, c = 20, 1, 90
    centers = [[5, 6], [-5, -6]]
    return a, b, c, centers


def obstacle_cfg_vneck():
    c_sq = 0.36
    coef = 5
    return c_sq, coef

