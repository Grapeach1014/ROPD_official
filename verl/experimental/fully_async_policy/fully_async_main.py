# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
import socket
import threading
import time
from pprint import pprint
from typing import Any

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.ropd_judge_worker import BlackOPDJudgeWorker
from verl.experimental.fully_async_policy.black_opd_queue import BoundedGroupQueue
#from verl.experimental.fully_async_policy.ropd_queue import BoundedGroupQueue
from verl.experimental.fully_async_policy.black_opd_window_coordinator import WindowCoordinator
#from verl.experimental.fully_async_policy.ropd_window_coordinator import WindowCoordinator
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, need_reference_policy
from verl.utils.fs import copy_to_local

STRICT_ASYNC_RESOURCE_POOL_MODE_SPLIT = "split"
STRICT_ASYNC_RESOURCE_POOL_MODE_SHARED = "shared"


def is_ropd_strict_async(config) -> bool:
    ropd_execution = config.get("ropd_execution")
    if ropd_execution is None:
        return False
    return ropd_execution.get("mode", "sync") == "strict_async"


def get_strict_async_resource_pool_mode(config) -> str:
    if not is_ropd_strict_async(config):
        return STRICT_ASYNC_RESOURCE_POOL_MODE_SPLIT

    strict_cfg = config.ropd_execution.get("strict_async", {})
    mode = str(strict_cfg.get("resource_pool_mode", STRICT_ASYNC_RESOURCE_POOL_MODE_SPLIT))
    if mode not in {STRICT_ASYNC_RESOURCE_POOL_MODE_SPLIT, STRICT_ASYNC_RESOURCE_POOL_MODE_SHARED}:
        raise ValueError(
            "ropd_execution.strict_async.resource_pool_mode must be one of "
            f"{STRICT_ASYNC_RESOURCE_POOL_MODE_SPLIT}, {STRICT_ASYNC_RESOURCE_POOL_MODE_SHARED}; got {mode}"
        )
    return mode


def should_share_strict_async_resource_pool(config) -> bool:
    return get_strict_async_resource_pool_mode(config) == STRICT_ASYNC_RESOURCE_POOL_MODE_SHARED


@ray.remote(num_cpus=1)
class BlackOPDJudgeActor:
    def __init__(self, config: Any, tokenizer: Any) -> None:
        self.worker = BlackOPDJudgeWorker(config=config, tokenizer=tokenizer)

    def score(self, payload: Any) -> tuple[str, bytes]:
        rollout_sample = ray.cloudpickle.loads(payload) if isinstance(payload, (bytes, bytearray)) else payload
        scored = self.worker.score(rollout_sample)
        return scored.rollout_sample.group_id, ray.cloudpickle.dumps(scored)


@ray.remote(num_cpus=1)
class StrictAsyncJudgeDispatcher:
    """Pump rollout samples through judge workers and publish scored samples to the trainer queue."""

    def __init__(
        self,
        config: Any,
        raw_queue: Any,
        scored_queue: Any,
        window_coordinator: Any,
        judge_workers: list[Any],
    ) -> None:
        self.config = config
        self.raw_queue = raw_queue
        self.scored_queue = scored_queue
        self.window_coordinator = window_coordinator
        self.judge_workers = judge_workers
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        fail_policy = self.config.ropd_execution.strict_async.get("fail_policy", "stop")
        pending: dict[Any, str] = {}
        worker_index = 0
        upstream_done = False

        try:
            while self._running:
                while not upstream_done and self._running:
                    queue_size = ray.get(self.raw_queue.qsize.remote())
                    if queue_size <= 0:
                        break

                    payload = ray.get(self.raw_queue.get_nowait.remote())
                    if payload is None:
                        upstream_done = True
                        break

                    judge_worker = self.judge_workers[worker_index % len(self.judge_workers)]
                    worker_index += 1
                    pending[judge_worker.score.remote(payload)] = "judge"

                if pending:
                    done_refs, _ = ray.wait(list(pending.keys()), num_returns=1, timeout=1.0)
                    if not done_refs:
                        continue

                    done_ref = done_refs[0]
                    pending.pop(done_ref, None)
                    try:
                        group_id, scored_payload = ray.get(done_ref)
                        self._put_scored_payload(scored_payload)
                        ray.get(self.window_coordinator.mark_reward_done.remote(group_id))
                    except Exception:
                        if fail_policy == "stop":
                            raise
                else:
                    if upstream_done:
                        break
                    time.sleep(0.1)
        finally:
            self._put_scored_payload(None, force=True)

    def _put_scored_payload(self, payload: bytes | None, force: bool = False) -> None:
        while self._running or force:
            if ray.get(self.scored_queue.put_nowait.remote(payload)):
                return
            time.sleep(0.1)


def create_resource_pool_manager(config, roles: list) -> ResourcePoolManager:
    """
    Create resource pool manager

    Args:
        config: Configuration object
        roles: List of roles that need to create resource pools

    Returns:
        ResourcePoolManager: Resource pool manager
    """
    resource_pool_spec = {}
    mapping = {}

    if should_share_strict_async_resource_pool(config):
        assert config.trainer.n_gpus_per_node > 0, "config.trainer.n_gpus_per_node must be greater than 0"
        assert config.trainer.nnodes > 0, "config.trainer.nnodes must be greater than 0"
        if config.rollout.n_gpus_per_node != config.trainer.n_gpus_per_node or config.rollout.nnodes != config.trainer.nnodes:
            raise ValueError(
                "strict_async shared resource pool requires rollout and trainer to share the same GPU topology: "
                f"trainer=({config.trainer.nnodes}, {config.trainer.n_gpus_per_node}) "
                f"rollout=({config.rollout.nnodes}, {config.rollout.n_gpus_per_node})"
            )

        shared_pool = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        resource_pool_spec["shared_pool"] = shared_pool
        for role in roles:
            mapping[role] = "shared_pool"
        return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    # Actor/Critic resource pool
    if any(role in roles for role in [Role.Actor, Role.ActorRollout, Role.Critic, Role.RefPolicy, Role.RewardModel]):
        assert config.trainer.n_gpus_per_node > 0, "config.trainer.n_gpus_per_node must be greater than 0"
        assert config.trainer.nnodes > 0, "config.trainer.nnodes must be greater than 0"

        trainer_pool = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        resource_pool_spec["trainer_pool"] = trainer_pool

        # Map training-related roles to the same resource pool
        for role in [Role.Actor, Role.ActorRollout, Role.Critic, Role.RefPolicy, Role.RewardModel]:
            if role in roles:
                mapping[role] = "trainer_pool"

    # Rollout resource pool
    if Role.Rollout in roles:
        assert config.rollout.n_gpus_per_node > 0, "config.rollout.n_gpus_per_node must be greater than 0"
        assert config.rollout.nnodes > 0, "config.rollout.nnodes must be greater than 0"

        rollout_pool = [config.rollout.n_gpus_per_node] * config.rollout.nnodes
        resource_pool_spec["rollout_pool"] = rollout_pool
        mapping[Role.Rollout] = "rollout_pool"

    return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)


def create_role_worker_mapping(config):
    """
    Create mapping from roles to worker classes

    Args:
        config: Configuration object

    Returns:
        dict: Mapping from roles to worker classes
    """
    # Select worker class based on strategy
    if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.experimental.fully_async_policy.fsdp_workers import (
            CriticWorker,
            DetachActorWorker,
            DetachAsyncRolloutWorker,
        )
        from verl.single_controller.ray import RayWorkerGroup

        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == "megatron":
        assert config.critic.strategy == "megatron"
        from verl.experimental.fully_async_policy.megatron_worker import (
            CriticWorker,
            DetachActorWorker,
            DetachAsyncRolloutWorker,
        )
        from verl.single_controller.ray import RayWorkerGroup

        ray_worker_group_cls = RayWorkerGroup
    else:
        raise NotImplementedError(f"Unsupported strategy: {config.actor_rollout_ref.actor.strategy}")

    train_role = Role.ActorRollout if config.async_training.use_trainer_do_validate else Role.Actor
    role_worker_mapping = {
        train_role: ray.remote(DetachActorWorker),
        Role.Rollout: ray.remote(DetachAsyncRolloutWorker),
        Role.Critic: ray.remote(CriticWorker),
    }

    if config.reward_model.enable:
        if config.reward_model.strategy in ["fsdp", "fsdp2"]:
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == "megatron":
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError

        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)

    # Add reference policy (if KL loss or reward is required)
    if need_reference_policy(config):
        role_worker_mapping[Role.RefPolicy] = ray.remote(DetachActorWorker)

    return role_worker_mapping, ray_worker_group_cls


@ray.remote(num_cpus=1)
class FullyAsyncTaskRunner:
    """
    Ray remote class for executing distributed PPO training tasks.
    """

    def __init__(self):
        self.running = False
        self.components = {}
        self.shutdown_event = threading.Event()

    def run(self, config):
        print("[ASYNC MAIN] Starting fully async PPO training...")
        self._initialize_components(config)
        self._run_training_loop()

    def _initialize_components(self, config) -> None:
        print(f"[ASYNC MAIN] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        print("[ASYNC MAIN] Initializing model and tokenizer...")
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        self.components["tokenizer"] = tokenizer
        self.components["processor"] = processor
        self.components["config"] = config

        print("[ASYNC MAIN] Creating worker mapping and resource pools...")
        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
        self.components["role_worker_mapping"] = role_worker_mapping
        self.components["ray_worker_group_cls"] = ray_worker_group_cls
        self._initialize_shared_resource_pool_manager(config)

        print("[ASYNC MAIN] Creating FullyAsyncRollouter...")
        self._create_rollouter(config)

        print("[ASYNC MAIN] Creating FullyAsyncTrainer...")
        self._create_trainer(config)

        # sync total_train_steps between rollouter and trainer
        total_train_steps = ray.get(self.components["rollouter"].get_total_train_steps.remote())
        print(f"total_train_steps {total_train_steps}")
        ray.get(self.components["trainer"].set_total_train_steps.remote(total_train_steps))

        print("[ASYNC MAIN] Creating runtime queues...")
        self._initialize_runtime_queues(config)

        print("[ASYNC MAIN] Setting up parameter synchronization...")
        from verl.experimental.fully_async_policy.param_sync import ParameterSynchronizer

        param_synchronizer = ParameterSynchronizer.remote(
            config=config,
            trainer=self.components["trainer"],
            rollouter=self.components["rollouter"],
            mq=self.components["message_queue_client"],
            window_coordinator=self.components.get("window_coordinator"),
        )
        ray.get(self.components["trainer"].set_parameter_synchronizer.remote(param_synchronizer))

        # load checkpoint and sync parameter before doing anything
        val_before_train = config.trainer.get("val_before_train", True)
        # param_version resume from ckpt or default 0
        param_version = ray.get(self.components["trainer"].load_checkpoint.remote())
        ray.get(self.components["rollouter"].load_checkpoint.remote())
        ray.get(
            param_synchronizer.sync_weights.remote(
                version=param_version,
                validate=val_before_train,
                use_trainer_do_validate=config.async_training.use_trainer_do_validate,
            )
        )
        ray.get(param_synchronizer.wait_last_valid.remote())

        self.components["param_synchronizer"] = param_synchronizer
        print("[ASYNC MAIN] All components initialized successfully")

    def _initialize_shared_resource_pool_manager(self, config) -> None:
        if not should_share_strict_async_resource_pool(config):
            return

        shared_manager = create_resource_pool_manager(
            config,
            roles=list(self.components["role_worker_mapping"].keys()),
        )
        # Pre-create placement groups on the driver and reuse them across trainer/rollouter copies.
        shared_manager.create_resource_pool()
        self.components["shared_resource_pool_manager"] = shared_manager

    def _get_resource_pool_manager(self, config, roles: list[Role]) -> ResourcePoolManager:
        shared_manager = self.components.get("shared_resource_pool_manager")
        if shared_manager is not None:
            return shared_manager
        return create_resource_pool_manager(config, roles=roles)

    def _initialize_runtime_queues(self, config) -> None:
        max_queue_size = ray.get(self.components["rollouter"].get_max_queue_size.remote())
        print(f"[ASYNC MAIN] Creating MessageQueue... max_queue_size {max_queue_size}")
        message_queue = MessageQueue.remote(config, max_queue_size)
        message_queue_client = MessageQueueClient(message_queue)
        self.components["message_queue"] = message_queue
        self.components["message_queue_client"] = message_queue_client

        ray.get(self.components["rollouter"].set_message_queue_client.remote(message_queue_client))
        ray.get(self.components["trainer"].set_message_queue_client.remote(message_queue_client))

        if not is_ropd_strict_async(config):
            return

        strict_cfg = config.ropd_execution.strict_async
        raw_queue = BoundedGroupQueue.remote(max_queue_size=int(strict_cfg.raw_group_queue_size))
        scored_queue = BoundedGroupQueue.remote(max_queue_size=int(strict_cfg.scored_group_queue_size))
        window_coordinator = WindowCoordinator.remote(target_group_count=int(strict_cfg.target_group_count))

        self.components["raw_queue"] = raw_queue
        self.components["scored_queue"] = scored_queue
        self.components["window_coordinator"] = window_coordinator

        self._wire_strict_async_runtime(config, raw_queue, scored_queue, window_coordinator)

    def _wire_strict_async_runtime(self, config, raw_queue, scored_queue, window_coordinator) -> None:
        judge_worker_count = int(config.ropd_execution.strict_async.judge_worker_count)
        judge_workers = [
            BlackOPDJudgeActor.remote(config=config, tokenizer=self.components["tokenizer"])
            for _ in range(judge_worker_count)
        ]
        judge_dispatcher = StrictAsyncJudgeDispatcher.remote(
            config=config,
            raw_queue=raw_queue,
            scored_queue=scored_queue,
            window_coordinator=window_coordinator,
            judge_workers=judge_workers,
        )

        self.components["judge_workers"] = judge_workers
        self.components["judge_dispatcher"] = judge_dispatcher

        ray.get(self.components["rollouter"].set_ropd_async_context.remote(raw_queue, window_coordinator))
        ray.get(self.components["trainer"].set_ropd_async_context.remote(scored_queue, window_coordinator))

    def _create_rollouter(self, config) -> None:
        rollouter = FullyAsyncRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping={Role.Rollout: self.components["role_worker_mapping"][Role.Rollout]},
            resource_pool_manager=self._get_resource_pool_manager(config, roles=[Role.Rollout]),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )

        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())

        self.components["rollouter"] = rollouter
        print("[ASYNC MAIN] Rollouter created and initialized successfully")

    def _create_trainer(self, config) -> None:
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }

        trainer = FullyAsyncTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=self._get_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )

        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[ASYNC MAIN] FullyAsyncTrainer created and initialized successfully")

    def _run_training_loop(self):
        self.running = True

        print("[ASYNC MAIN] Starting Rollouter and Trainer...")
        rollouter_future = self.components["rollouter"].fit.remote()
        trainer_future = self.components["trainer"].fit.remote()
        futures = [rollouter_future, trainer_future]
        judge_dispatcher_future = None
        if "judge_dispatcher" in self.components:
            judge_dispatcher_future = self.components["judge_dispatcher"].run.remote()
            futures.append(judge_dispatcher_future)

        try:
            while futures:
                # Use ray.wait to monitor all futures and return when any one is completed.
                done_futures, remaining_futures = ray.wait(futures, num_returns=1, timeout=None)

                for future in done_futures:
                    try:
                        ray.get(future)
                        print("[ASYNC MAIN] One component completed successfully")
                    except Exception as e:
                        print(f"[ASYNC MAIN] Component failed with error: {e}")
                        for remaining_future in remaining_futures:
                            ray.cancel(remaining_future)
                        raise e

                futures = remaining_futures

        except Exception as e:
            print(f"[ASYNC MAIN] Training failed: {e}")
            for future in futures:
                ray.cancel(future)
            raise
        finally:
            self._shutdown_runtime()
            print("[ASYNC MAIN] Training completed or interrupted")

    def _shutdown_runtime(self) -> None:
        if "judge_dispatcher" in self.components:
            ray.get(self.components["judge_dispatcher"].stop.remote())
        if "message_queue_client" in self.components:
            queue_actor = self.components["message_queue_client"].queue_actor
            ray.get(queue_actor.clear_queue.remote())
            ray.get(queue_actor.shutdown.remote())
        if "raw_queue" in self.components:
            self._clear_group_queue(self.components["raw_queue"])
        if "scored_queue" in self.components:
            self._clear_group_queue(self.components["scored_queue"])

    @staticmethod
    def _clear_group_queue(queue_actor: Any) -> None:
        while ray.get(queue_actor.qsize.remote()) > 0:
            ray.get(queue_actor.get_nowait.remote())


@hydra.main(config_path="config", config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    from verl.trainer.main_ppo import run_ppo

    # Ensure async training config exists
    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")
    from time import time

    start_time = time()
    run_ppo(config, task_runner_class=FullyAsyncTaskRunner)
    print(f"total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
