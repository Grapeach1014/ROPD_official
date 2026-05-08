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

from enum import Enum
from typing import Any

import ray


class WindowState(str, Enum):
    OPEN = "OPEN"
    SEALED = "SEALED"
    READY_TO_UPDATE = "READY_TO_UPDATE"


class WindowCoordinatorState:
    """Tracks strict on-policy window membership and version advancement."""

    def __init__(self, target_group_count: int):
        if target_group_count <= 0:
            raise ValueError(f"target_group_count must be positive, got: {target_group_count}")

        self.target_group_count = int(target_group_count)
        self.current_param_version = 0
        self.window_id = 0
        self.window_state = WindowState.OPEN

        self._next_group_index = 0
        self._admitted_group_ids: set[str] = set()
        self._reward_done_group_ids: set[str] = set()
        self._consumed_group_ids: set[str] = set()

    def request_admission(self) -> dict[str, Any]:
        if self.window_state is not WindowState.OPEN:
            return {
                "admitted": False,
                "reason": self.window_state.value,
                "param_version": self.current_param_version,
                "window_id": self.window_id,
            }

        group_id = f"v{self.current_param_version}-g{self._next_group_index}"
        self._next_group_index += 1
        self._admitted_group_ids.add(group_id)

        if len(self._admitted_group_ids) >= self.target_group_count:
            self.window_state = WindowState.SEALED

        return {
            "admitted": True,
            "group_id": group_id,
            "param_version": self.current_param_version,
            "window_id": self.window_id,
        }

    def mark_reward_done(self, group_id: str) -> None:
        self._ensure_group_admitted(group_id)
        self._reward_done_group_ids.add(group_id)
        if self.window_state is WindowState.SEALED and self._reward_done_group_ids == self._admitted_group_ids:
            self.window_state = WindowState.READY_TO_UPDATE

    def mark_consumed(self, group_id: str) -> None:
        self._ensure_group_admitted(group_id)
        self._consumed_group_ids.add(group_id)

    def ready_for_sync(self) -> bool:
        return self.window_state is WindowState.READY_TO_UPDATE and self._consumed_group_ids == self._admitted_group_ids

    def advance_version(self) -> int:
        if not self.ready_for_sync():
            raise RuntimeError("window is not ready for version advance")

        self.current_param_version += 1
        self.window_id += 1
        self.window_state = WindowState.OPEN

        self._next_group_index = 0
        self._admitted_group_ids.clear()
        self._reward_done_group_ids.clear()
        self._consumed_group_ids.clear()

        return self.current_param_version

    def snapshot(self) -> dict[str, int | str]:
        return {
            "current_param_version": self.current_param_version,
            "window_id": self.window_id,
            "window_state": self.window_state.value,
            "admitted_count": len(self._admitted_group_ids),
            "reward_done_count": len(self._reward_done_group_ids),
            "consumed_count": len(self._consumed_group_ids),
        }

    def _ensure_group_admitted(self, group_id: str) -> None:
        if group_id not in self._admitted_group_ids:
            raise KeyError(f"group_id is not admitted in the current window: {group_id}")


@ray.remote(num_cpus=1)
class WindowCoordinator:
    """Ray actor wrapper around ``WindowCoordinatorState``."""

    def __init__(self, target_group_count: int):
        self._state = WindowCoordinatorState(target_group_count=target_group_count)

    def request_admission(self) -> dict[str, Any]:
        return self._state.request_admission()

    def mark_reward_done(self, group_id: str) -> None:
        self._state.mark_reward_done(group_id)

    def mark_consumed(self, group_id: str) -> None:
        self._state.mark_consumed(group_id)

    def ready_for_sync(self) -> bool:
        return self._state.ready_for_sync()

    def advance_version(self) -> int:
        return self._state.advance_version()

    def snapshot(self) -> dict[str, int | str]:
        return self._state.snapshot()
