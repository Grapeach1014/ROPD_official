from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_CKPT_ROOT = PROJECT_ROOT / "checkpoints" / "ropd"

__all__ = ["resolve_ropd_ckpt_dir", "resolve_ropd_ckpt_dir_with_source"]


def resolve_ropd_ckpt_dir_with_source() -> tuple[str, str]:
    """Resolve the checkpoint root used by ROPD training entrypoints.

    Resolution order:
    1. ``ROPD_CKPT_DIR`` environment variable (explicit override).
    2. ``<repo>/checkpoints/ropd`` (in-repo default).
    """

    explicit_ckpt_dir = os.environ.get("ROPD_CKPT_DIR")
    if explicit_ckpt_dir:
        return explicit_ckpt_dir, "ROPD_CKPT_DIR"

    return str(PROJECT_CKPT_ROOT.resolve()), "project_root_fallback"


def resolve_ropd_ckpt_dir() -> str:
    resolved_path, _ = resolve_ropd_ckpt_dir_with_source()
    return resolved_path
