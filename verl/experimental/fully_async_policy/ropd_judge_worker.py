from __future__ import annotations

from typing import Any

from verl.experimental.fully_async_policy.detach_utils import RolloutSample, ScoredRolloutSample
from verl.trainer.ppo.reward import load_reward_manager


class BlackOPDJudgeWorker:
    def __init__(self, config: Any, tokenizer: Any) -> None:
        self.reward_fn = load_reward_manager(
            config,
            tokenizer,
            num_examine=0,
            **config.reward_model.get("reward_kwargs", {}),
        )

    def score(self, rollout_sample: RolloutSample) -> ScoredRolloutSample:
        result = self.reward_fn(rollout_sample.full_batch, return_dict=True)
        return ScoredRolloutSample(
            rollout_sample=rollout_sample,
            token_level_scores=result["reward_tensor"],
            reward_extra_info=result["reward_extra_info"],
            reward_control=result["reward_control"],
        )
