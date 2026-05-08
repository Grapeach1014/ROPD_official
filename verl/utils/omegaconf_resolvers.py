from __future__ import annotations

from omegaconf import OmegaConf

from .ropd_paths import resolve_ropd_ckpt_dir


def register_omegaconf_resolvers() -> None:
    if not OmegaConf.has_resolver("ropd_ckpt_dir"):
        OmegaConf.register_new_resolver("ropd_ckpt_dir", resolve_ropd_ckpt_dir, use_cache=False)
