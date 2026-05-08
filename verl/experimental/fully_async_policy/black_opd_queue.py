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

from collections import deque
from threading import Lock
from typing import Any

import ray


class BoundedGroupQueueState:
    """A bounded FIFO queue that rejects overflow instead of dropping existing items."""

    def __init__(self, max_queue_size: int):
        if max_queue_size <= 0:
            raise ValueError(f"max_queue_size must be positive, got: {max_queue_size}")

        self.max_queue_size = int(max_queue_size)
        self._queue: deque[Any] = deque()

    def put_nowait(self, item: Any) -> bool:
        if len(self._queue) >= self.max_queue_size:
            return False

        self._queue.append(item)
        return True

    def get_nowait(self) -> Any:
        return self._queue.popleft()

    def qsize(self) -> int:
        return len(self._queue)


@ray.remote(num_cpus=1)
class BoundedGroupQueue:
    """Ray actor wrapper around ``BoundedGroupQueueState``."""

    def __init__(self, max_queue_size: int):
        self._state = BoundedGroupQueueState(max_queue_size=max_queue_size)
        self._lock = Lock()

    def put_nowait(self, item: Any) -> bool:
        with self._lock:
            return self._state.put_nowait(item)

    def get_nowait(self) -> Any:
        with self._lock:
            return self._state.get_nowait()

    def qsize(self) -> int:
        with self._lock:
            return self._state.qsize()
