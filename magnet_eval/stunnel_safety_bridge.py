"""Bridge the FSBM stunnel evaluator into a MAGNET card.

The bridge deliberately returns the evaluator's complete JSON report.  The
MAGNET card keeps the empirical acceptance criteria in its own ``claim`` block
so that an independent evaluator can read the asserted claim without having to
inspect this module.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Mapping


DEFAULT_CHECKPOINT = Path(
    "outputs/runs/stunnel/2026.04.13/150806/checkpoints/last.ckpt"
)


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the FSBM root from a path inside the checkout."""

    current = Path.cwd() if start is None else Path(start)
    current = current.expanduser().resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "stunnel_safety_eval.py").is_file() and (
            candidate / "fsbm"
        ).is_dir():
            return candidate
    raise RuntimeError(
        "FSBM project root not found; run the card from inside the FSBM "
        "checkout or set FSBM_PROJECT_ROOT"
    )


def resolve_checkpoint(
    project_root: str | Path, checkpoint: str | Path = DEFAULT_CHECKPOINT
) -> Path:
    """Resolve and validate the frozen stunnel checkpoint path."""

    root = Path(project_root).expanduser().resolve()
    path = Path(checkpoint).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"stunnel checkpoint not found: {path}. Supply the frozen asset "
            "or set FSBM_STUNNEL_CHECKPOINT."
        )
    return path


def run_evaluation(
    *,
    project_root: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    direction: str = "fwd",
    delta: float = 0.1,
    alpha: float = 0.1,
    n_calib: int = 128,
    n_test: int = 256,
    n_probes: int = 8,
    nfe: int = 100,
    batch_size: int = 64,
    seed: int = 20260731,
    target_mass: float = 0.99,
    risk_points: int = 21,
    examples: int = 6,
    torch_threads: int = 4,
) -> Dict[str, Any]:
    """Run the frozen FSBM evaluation and return its machine-readable report."""

    root = find_project_root(project_root)
    checkpoint_path = resolve_checkpoint(root, checkpoint)
    destination = Path(output_dir).expanduser()
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()

    # The evaluator uses project-local imports and restores the Hydra snapshot
    # adjacent to the checkpoint.  Import only after the root is established.
    import sys

    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    import stunnel_safety_eval

    args = Namespace(
        ckpt_file=checkpoint_path,
        output=destination,
        device=str(device),
        direction=str(direction),
        delta=float(delta),
        alpha=float(alpha),
        n_calib=int(n_calib),
        n_test=int(n_test),
        n_probes=int(n_probes),
        nfe=int(nfe),
        batch_size=int(batch_size),
        seed=int(seed),
        target_radius=None,
        target_mass=float(target_mass),
        risk_points=int(risk_points),
        examples=int(examples),
        torch_threads=int(torch_threads),
    )
    stunnel_safety_eval.main(args)

    report_path = destination / "stunnel_safety_report.json"
    if not report_path.is_file():
        raise RuntimeError(f"FSBM evaluator did not create {report_path}")
    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    report["magnet_bridge"] = {
        "project_root": str(root),
        "checkpoint": str(checkpoint_path),
        "output_dir": str(destination),
        "report_path": str(report_path),
    }
    return report


def summarize_evaluation(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract the small set of values used by the MAGNET claim."""

    heldout = report["heldout"]
    operating = report["alarm_performance"]["operating_point"]
    retained_risk = operating["collision_risk_among_trusted"]
    return {
        "covered_predicted_safe_collision_count": int(
            heldout["covered_predicted_safe_collision_count_invariant"]
        ),
        "heldout_marginal_coverage": float(
            heldout["marginal_conformal_coverage"]
        ),
        "target_marginal_coverage": float(heldout["target_marginal_coverage"]),
        "overall_sampled_collision_risk": float(
            heldout["realized_perturbed_discrete_collision_item_fraction"]
        ),
        "retained_fraction": float(operating["retained_fraction"]),
        "sampled_collision_risk_among_retained": (
            None if retained_risk is None else float(retained_risk)
        ),
    }
