
import os
import sys

sys.path.append("..")
sys.path.append("ALAE/")
import matplotlib.pyplot as plt
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import logging
import json
from glob import glob
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from fsbm.dataset import get_dist_boundary
from fsbm.fsbm_model import FSBMLitModule
from fsbm.optimal_transport_couples import get_couples
from fsbm.plotting import *
from fsbm.sde import build_basedrift
import fsbm.gaussian_path as gpath_lib
from fsbm.utils import get_repo_path
from fsbm.utils import lin_int, NumpyFolder
import random


torch.backends.cudnn.benchmark = True
log = logging.getLogger(__name__)


def init_gpath(x0, x1, ccfg):
    B, D = x0.shape
    assert x1.shape == (B, D)

    T, S = 100, 200

    t = torch.linspace(0, 1, T, device=x0.device)
    tt = t.reshape(1, T, 1)
    xt = (1 - tt) * x0.reshape(B, 1, D) + tt * x1.reshape(B, 1, D)
    assert xt.shape == (B, T, D)

    s = torch.linspace(0, 1, S)
    ys = torch.zeros(B, S, 1)
    basedrift = build_basedrift(ccfg)
    return gpath_lib.EndPointGaussianPath(t, xt, s, ys, 2, basedrift)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    logging.getLogger("pytorch_lightning").setLevel(logging.getLevelName("INFO"))

    hydra_config = HydraConfig.get()

    # Get the number of nodes we are training on
    nnodes = hydra_config.launcher.get("nodes", 1)
    print("nnodes", nnodes)

    if cfg.get("seed", None) is not None:
        pl.utilities.seed.seed_everything(cfg.seed)

    print(cfg)
    print("Found {} CUDA devices.".format(torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(
            "{} \t Memory: {:.2f}GB".format(props.name, props.total_memory / (1024 ** 3))
        )

    keys = [
        "SLURM_NODELIST",
        "SLURM_JOB_ID",
        "SLURM_NTASKS",
        "SLURM_JOB_NAME",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_NODEID",
    ]
    log.info(json.dumps({k: os.environ.get(k, None) for k in keys}, indent=4))

    cmd_str = " \\\n".join([f"python {sys.argv[0]}"] + ["\t" + x for x in sys.argv[1:]])
    with open("cmd.sh", "w") as fout:
        print("#!/bin/bash\n", file=fout)
        print(cmd_str, file=fout)

    log.info(f"CWD: {os.getcwd()}")

    # Construct model
    p0, p1, p0_val, p1_val = get_dist_boundary(cfg)
    xt_ks = torch.load(
        f'/home/ptheodor3/Documents/CODING/generalized-schrodinger-bridge-matching/data/kp_traj_{cfg.name}.pt')
    model = FSBMLitModule(cfg, p0, p1, p0_val, p1_val, xt_ks, tau=cfg.tau, guidance=True)
    if cfg.optim.continue_old_training:
        path = ('')
        assert path is not None
        ckpt = torch.load(path)
        model.load_state_dict(ckpt["state_dict"])

    # model.log_boundary(p0, p1, p0_val, p1_val)
    if cfg.prob.name == "opinion":
        model.log_basedrift(p0)

    # Checkpointing, logging, and other misc.
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="epoch-{epoch:03d}_step-{step}",
            auto_insert_metric_name=False,
            save_top_k=-1,  # save all models whenever callback occurs
            save_last=True,
            every_n_epochs=1,
            verbose=True,
        ),
        LearningRateMonitor(),
    ]

    slurm_plugin = pl.plugins.environments.SLURMEnvironment(auto_requeue=False)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["cwd"] = os.getcwd()
    loggers = [pl.loggers.CSVLogger(save_dir=".")]
    if cfg.use_wandb:
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        loggers.append(
            pl.loggers.WandbLogger(
                save_dir=".",
                name=f"{cfg.prob.name}_{now}",
                project="FSBM",
                log_model=False,
                config=cfg_dict,
                resume=True,
            )
        )

    strategy = "ddp" if torch.cuda.device_count() > 1 else None

    trainer = pl.Trainer(
        max_epochs=cfg.optim.max_epochs,
        accelerator="gpu",
        strategy=strategy,
        logger=loggers,
        num_nodes=nnodes,
        callbacks=callbacks,
        precision=cfg.get("precision", 32),
        gradient_clip_val=cfg.optim.grad_clip,
        plugins=slurm_plugin if slurm_plugin.detect() else None,
        reload_dataloaders_every_n_epochs=1,
        num_sanity_val_steps=-1,
        check_val_every_n_epoch=1,
        replace_sampler_ddp=False,
        enable_progress_bar=False,
    )

    # If we specified a checkpoint to resume from, use it
    checkpoint = cfg.get("resume", None)

    # Check if a checkpoint exists in this working directory.  If so, then we are resuming from a pre-emption
    # This takes precedence over a command line specified checkpoint
    checkpoints = glob("checkpoints/**/*.ckpt", recursive=True)
    if len(checkpoints) > 0:
        # Use the checkpoint with the latest modification time
        checkpoint = sorted(checkpoints, key=os.path.getmtime)[-1]

    # Load dataset (train loader will be generated online)
    trainer.fit(model, ckpt_path=checkpoint)

    metric_dict = trainer.callback_metrics

    for k, v in metric_dict.items():
        metric_dict[k] = float(v)

    with open("metrics.json", "w") as fout:
        print(json.dumps(metric_dict), file=fout)
    return metric_dict


if __name__ == "__main__":
    main()
