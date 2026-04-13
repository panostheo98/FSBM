import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf

from fsbm.utils import restore_model


def to_serializable(obj):
    if OmegaConf.is_config(obj):
        return to_serializable(OmegaConf.to_container(obj, resolve=True))
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_serializable(v) for v in obj]
    if torch.is_tensor(obj):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    return obj


def apply_overrides(cfg, args):
    cfg.contraction.enabled = True
    if args.directions is not None:
        cfg.contraction.directions = args.directions.split(",")
    if args.n_calib is not None:
        cfg.contraction.n_calib = args.n_calib
    if args.alpha is not None:
        cfg.contraction.alpha = args.alpha
    if args.delta0s is not None:
        cfg.contraction.delta0s = [float(x) for x in args.delta0s.split(",")]
    if args.nfe is not None:
        cfg.contraction.nfe = args.nfe
    if args.log_steps is not None:
        cfg.contraction.log_steps = args.log_steps
    if args.seed is not None:
        cfg.contraction.seed = args.seed
    if args.save_scores:
        cfg.contraction.save_scores = True
    if args.save_terminal_distances:
        cfg.contraction.save_terminal_distances = True
    if args.save_histograms:
        cfg.contraction.save_histograms = True
    if args.hist_bins is not None:
        cfg.contraction.hist_bins = args.hist_bins
    if args.verbose:
        cfg.contraction.verbose = True
    if args.jacobian:
        cfg.contraction.jacobian.enabled = True
    if args.jacobian_num_states is not None:
        cfg.contraction.jacobian.num_states = args.jacobian_num_states
    if args.jacobian_num_times is not None:
        cfg.contraction.jacobian.num_times = args.jacobian_num_times
    return cfg


def default_output_path(experiment: str) -> Path:
    now = datetime.now()
    output_dir = (
        Path("outputs")
        / "contraction"
        / experiment
        / now.strftime("%Y.%m.%d")
        / now.strftime("%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "contraction_report.json"


def main(args):
    model, cfg = restore_model(args.ckpt_file, device=args.device)
    cfg = apply_overrides(cfg, args)
    model.eval()

    results = model.evaluate_contraction_suite(cfg.contraction)
    payload = {
        "checkpoint": str(args.ckpt_file),
        "experiment": cfg.name,
        "device": args.device,
        "contraction_cfg": to_serializable(cfg.contraction),
        "results": to_serializable(results),
    }

    output_path = Path(args.output) if args.output is not None else default_output_path(cfg.name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fout:
        json.dump(payload, fout, indent=2)

    print(f"Saved contraction report to {output_path}")
    for direction, result in results.items():
        first_delta = result["per_delta"][0]
        s_hat_alpha = first_delta["conformal"]["quantile"]
        print(f"{direction}: delta0={first_delta['delta0']} s_hat_alpha={s_hat_alpha:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_file", type=Path, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--directions", type=str, default=None)
    parser.add_argument("--n_calib", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--delta0s", type=str, default=None)
    parser.add_argument("--nfe", type=int, default=None)
    parser.add_argument("--log_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_scores", action="store_true")
    parser.add_argument("--save_terminal_distances", action="store_true")
    parser.add_argument("--save_histograms", action="store_true")
    parser.add_argument("--hist_bins", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--jacobian", action="store_true")
    parser.add_argument("--jacobian_num_states", type=int, default=None)
    parser.add_argument("--jacobian_num_times", type=int, default=None)
    main(parser.parse_args())
