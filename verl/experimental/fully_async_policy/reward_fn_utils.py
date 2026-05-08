from __future__ import annotations

from typing import Any

from verl.trainer.main_ppo import build_val_reward_fn
from verl.trainer.ppo.reward import load_reward_manager


def build_fully_async_reward_fns(config: Any, tokenizer: Any) -> tuple[Any, Any]:
    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    val_reward_fn = build_val_reward_fn(config, tokenizer)
    return reward_fn, val_reward_fn
