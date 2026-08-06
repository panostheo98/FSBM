"""Shared project-root discovery for FSBM MAGNET evaluation cards."""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: str | Path | None = None) -> Path:
    """Find an FSBM checkout from a path located anywhere below its root."""

    current = Path.cwd() if start is None else Path(start)
    current = current.expanduser().resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "fsbm" / "fsbm_model.py").is_file() and (
            candidate / "configs"
        ).is_dir():
            return candidate
    raise RuntimeError(
        "FSBM project root not found; run the card from inside the FSBM "
        "checkout or set FSBM_PROJECT_ROOT"
    )
