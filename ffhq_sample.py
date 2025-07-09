import matplotlib.pyplot as plt
import argparse
import os, sys
from datetime import datetime

sys.path.append("..")
sys.path.append("ALAE/")
from pathlib import Path
import logging
from fsbm.plotting import *
from fsbm.utils import get_repo_path, restore_model
from fsbm.dataset import get_dist_boundary
from alae_ffhq_inference import load_model, encode, decode
from torchvision.utils import save_image

torch.backends.cudnn.benchmark = True
log = logging.getLogger(__name__)
BASE_DIR = Path("outputs//afhq")
date_str = datetime.now().strftime("%Y-%m-%d")
time_str = datetime.now().strftime("%H-%M-%S")

def main(opt, log):

    log(opt)

    ## Load model
    model, cfg = restore_model(opt.ckpt_file, device=opt.device)

    model.eval()

    _, _, p0_val, p1_val = get_dist_boundary(cfg)
    samples = 2048

    default_config = (str(get_repo_path()) + '/ALAE/configs/ffhq.yaml')
    training_artifacts_dir = (str(get_repo_path()) + f"/ALAE/training_artifacts/{cfg.name}/")
    alae_model = load_model(default_config, training_artifacts_dir=training_artifacts_dir)

    output = model.sample(p0_val(samples), log_steps=20, nfe=1000, direction='fwd')
    dir = f'outputs/sample/{opt.experiment}/fsbm_outputs/{date_str}/{time_str}/'
    os.makedirs(dir, exist_ok=True)
    for i in range(samples):
        decoded_inp = decode(alae_model, output['xs'][i, -1].unsqueeze(0).cpu()) * 0.5 + 0.5
        save_image(decoded_inp, os.path.join(dir, f'generated_p1_{i}.png'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_file", type=Path, default=None)
    parser.add_argument("--experiment", type=str, default=None, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    opt = parser.parse_args()
    log = print
    main(opt, log)