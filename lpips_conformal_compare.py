"""Paired LPIPS conformal comparison for image-translation methods.

The experimental unit is a source identity.  For every identity this script
compares the same source image with one output from each method, then reports
separate upper conformal thresholds and an upper threshold on the paired
difference ``LPIPS(FSBM) - LPIPS(baseline)``.

Example
-------
PYTHONPATH=/path/to/PerceptualSimilarity python lpips_conformal_compare.py \
    --input-dir /path/to/fsbm_inputs \
    --fsbm-dir /path/to/fsbm_outputs \
    --baseline-dir /path/to/lsbm_outputs \
    --baseline-name LSBM --task gender \
    --output-dir reports/artifacts/lpips_conformal/gender
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def conformal_upper_threshold(
    scores: Sequence[float], alpha: float
) -> Dict[str, object]:
    """Return the finite-sample split-conformal upper threshold."""
    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("scores must be non-empty")
    if not np.isfinite(values).all():
        raise ValueError("scores must all be finite")
    if not 0.0 <= float(alpha) < 1.0:
        raise ValueError("alpha must satisfy 0 <= alpha < 1")

    n = int(values.size)
    order = int(math.ceil((n + 1) * (1.0 - float(alpha))))
    threshold = (
        math.inf
        if order > n
        else float(np.partition(values, order - 1)[order - 1])
    )
    return {
        "alpha": float(alpha),
        "coverage": float(1.0 - alpha),
        "n_calibration": n,
        "order_statistic": order,
        "threshold": threshold,
        "finite_threshold": bool(math.isfinite(threshold)),
        "minimum_finite_alpha": float(1.0 / (n + 1)),
    }


def _natural_key(text: str) -> Tuple[object, ...]:
    pieces: List[object] = []
    current = ""
    numeric = text[:1].isdigit()
    for char in text:
        char_numeric = char.isdigit()
        if current and char_numeric != numeric:
            pieces.append(int(current) if numeric else current.lower())
            current = ""
        current += char
        numeric = char_numeric
    if current:
        pieces.append(int(current) if numeric else current.lower())
    return tuple(pieces)


def indexed_images(directory: Path) -> Mapping[str, Path]:
    """Index images by filename stem, rejecting ambiguous identities."""
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {directory}")
    result: Dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in result:
            raise ValueError(
                f"duplicate sample identity {path.stem!r} in {directory}"
            )
        result[path.stem] = path.resolve()
    if not result:
        raise ValueError(f"no supported images found in {directory}")
    return result


def matched_image_paths(
    input_dir: Path, fsbm_dir: Path, baseline_dir: Path
) -> List[Tuple[str, Path, Path, Path]]:
    """Return exact three-way identity matches, failing on any mismatch."""
    inputs = indexed_images(input_dir)
    fsbm = indexed_images(fsbm_dir)
    baseline = indexed_images(baseline_dir)
    identity_sets = {
        "input": set(inputs),
        "fsbm": set(fsbm),
        "baseline": set(baseline),
    }
    union = set().union(*identity_sets.values())
    if any(ids != union for ids in identity_sets.values()):
        details = {
            name: sorted(union - ids, key=_natural_key)
            for name, ids in identity_sets.items()
            if ids != union
        }
        raise ValueError(f"image identities are not exactly matched: {details}")
    return [
        (sample_id, inputs[sample_id], fsbm[sample_id], baseline[sample_id])
        for sample_id in sorted(union, key=_natural_key)
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_batch(paths: Iterable[Path], resolution: int):
    import torch
    import torch.nn.functional as functional
    from PIL import Image

    tensors = []
    for path in paths:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
        tensors.append(torch.from_numpy(array).permute(2, 0, 1) / 255.0)
    batch = torch.stack(tensors, dim=0)
    if batch.shape[-2:] != (resolution, resolution):
        batch = functional.interpolate(
            batch, size=(resolution, resolution), mode="area"
        )
    return batch


def compute_lpips_scores(
    matches: Sequence[Tuple[str, Path, Path, Path]],
    *,
    net: str,
    version: str,
    resolution: int,
    batch_size: int,
    device_name: str,
):
    """Compute both methods in the same batches and on the same device."""
    import lpips
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if resolution < 16:
        raise ValueError("resolution must be at least 16")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    model = lpips.LPIPS(net=net, version=version, verbose=True)
    model = model.to(device).eval()

    records: List[Dict[str, object]] = []
    with torch.inference_mode():
        for start in range(0, len(matches), batch_size):
            subset = matches[start : start + batch_size]
            source = _load_batch((item[1] for item in subset), resolution).to(device)
            fsbm = _load_batch((item[2] for item in subset), resolution).to(device)
            baseline = _load_batch((item[3] for item in subset), resolution).to(device)
            fsbm_values = (
                model(source, fsbm, normalize=True).reshape(-1).detach().cpu().numpy()
            )
            baseline_values = (
                model(source, baseline, normalize=True)
                .reshape(-1)
                .detach()
                .cpu()
                .numpy()
            )
            for item, fsbm_score, baseline_score in zip(
                subset, fsbm_values, baseline_values
            ):
                delta = float(fsbm_score - baseline_score)
                records.append(
                    {
                        "sample_id": item[0],
                        "input_path": str(item[1]),
                        "fsbm_path": str(item[2]),
                        "baseline_path": str(item[3]),
                        "lpips_fsbm": float(fsbm_score),
                        "lpips_baseline": float(baseline_score),
                        "delta_fsbm_minus_baseline": delta,
                        "fsbm_lower_lpips": bool(delta < 0.0),
                    }
                )
    return records, device


def metric_summary(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "maximum": float(array.max()),
    }


def build_summary(
    records: Sequence[Mapping[str, object]],
    *,
    task: str,
    baseline_name: str,
    alpha: float,
    net: str,
    version: str,
    resolution: int,
    device,
) -> Dict[str, object]:
    fsbm = np.asarray([row["lpips_fsbm"] for row in records], dtype=float)
    baseline = np.asarray([row["lpips_baseline"] for row in records], dtype=float)
    delta = fsbm - baseline
    paired_bound = conformal_upper_threshold(delta, alpha)
    return {
        "schema": "fsbm-paired-lpips-conformal-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "method_names": {"primary": "FSBM", "baseline": baseline_name},
        "metric": {
            "name": "LPIPS",
            "reference": "same source image for both translated outputs",
            "interpretation": (
                "perceptual change from source; lower means less source-image "
                "change, not necessarily better translation success"
            ),
            "network": net,
            "version": version,
            "input_normalization": "RGB [0,1], LPIPS normalize=True",
            "evaluation_resolution": [resolution, resolution],
            "resize": "torch area interpolation from original resolution",
        },
        "calibration": {
            "experimental_unit": "matched source identity",
            "alpha": float(alpha),
            "nominal_marginal_coverage": float(1.0 - alpha),
            "n": int(len(records)),
            "all_available_matched_identities_used_for_calibration": True,
            "heldout_coverage_audit_performed": False,
        },
        "fsbm": {
            **metric_summary(fsbm),
            "conformal_upper_bound": conformal_upper_threshold(fsbm, alpha),
        },
        "baseline": {
            **metric_summary(baseline),
            "conformal_upper_bound": conformal_upper_threshold(baseline, alpha),
        },
        "paired_difference_fsbm_minus_baseline": {
            **metric_summary(delta),
            "conformal_upper_bound": paired_bound,
            "fsbm_lower_count": int(np.sum(delta < 0.0)),
            "tie_count": int(np.sum(delta == 0.0)),
            "fsbm_lower_fraction": float(np.mean(delta < 0.0)),
            "certifies_fsbm_lower_lpips_at_nominal_coverage": bool(
                float(paired_bound["threshold"]) < 0.0
            ),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "device": str(device),
        },
    }


def write_results(
    output_dir: Path,
    records: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "per_image_lpips.csv"
    with score_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--fsbm-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--net", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument("--version", choices=("0.0", "0.1"), default="0.1")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    matches = matched_image_paths(args.input_dir, args.fsbm_dir, args.baseline_dir)
    records, device = compute_lpips_scores(
        matches,
        net=args.net,
        version=args.version,
        resolution=args.resolution,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    summary = build_summary(
        records,
        task=args.task,
        baseline_name=args.baseline_name,
        alpha=args.alpha,
        net=args.net,
        version=args.version,
        resolution=args.resolution,
        device=device,
    )
    summary["provenance"] = {
        "input_directory": str(args.input_dir.resolve()),
        "fsbm_directory": str(args.fsbm_dir.resolve()),
        "baseline_directory": str(args.baseline_dir.resolve()),
        "evaluator": str(Path(__file__).resolve()),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_results(args.output_dir, records, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
