# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import ipaddress
import logging
import os
import socket

import uvicorn
from fastapi import FastAPI

logger = logging.getLogger(__file__)


def get_max_position_embeddings(hf_config) -> int:
    max_len = getattr(hf_config, "max_position_embeddings", None)
    if max_len is None:
        text_config = getattr(hf_config, "text_config", None)
        if text_config is not None:
            max_len = getattr(text_config, "max_position_embeddings", None)

    if max_len is None:
        raise ValueError("max_position_embeddings not found in HFModelConfig!")
    return int(max_len)


def resolve_rollout_max_model_len(
    configured_max_model_len: int | None,
    model_max_position_embeddings: int,
    *,
    model_name_or_path: str | None = None,
) -> int:
    """Resolve the effective rollout max length without silently widening user config.

    Args:
        configured_max_model_len: The configured rollout max length. When ``None``,
            the model's native limit is used.
        model_max_position_embeddings: The model's native maximum context length.
        model_name_or_path: Optional model identifier to include in validation errors.

    Returns:
        The effective rollout max length.

    Raises:
        ValueError: If the configured max length exceeds the model's native limit.
    """

    model_max_position_embeddings = int(model_max_position_embeddings)
    if configured_max_model_len is None:
        return model_max_position_embeddings

    configured_max_model_len = int(configured_max_model_len)
    if configured_max_model_len > model_max_position_embeddings:
        model_ref = model_name_or_path or "<unknown model>"
        raise ValueError(
            "Configured rollout max_model_len "
            f"({configured_max_model_len}) exceeds model max_position_embeddings "
            f"({model_max_position_embeddings}) for {model_ref}. "
            "Lower actor_rollout_ref.rollout.max_model_len / max_model_len, or explicitly "
            "increase the model's max_position_embeddings via actor_rollout_ref.model.override_config."
        )

    return configured_max_model_len


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def get_free_port(address: str) -> tuple[int, socket.socket]:
    family = socket.AF_INET
    if is_valid_ipv6_address(address):
        family = socket.AF_INET6

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((address, 0))

    port = sock.getsockname()[1]
    return port, sock


async def run_unvicorn(app: FastAPI, server_args, server_address, max_retries=5) -> tuple[int, asyncio.Task]:
    server_port, server_task = None, None

    for i in range(max_retries):
        try:
            server_port, sock = get_free_port(server_address)
            app.server_args = server_args
            config = uvicorn.Config(app, host=server_address, port=server_port, log_level="warning")
            server = uvicorn.Server(config)
            server.should_exit = True
            await server.serve()
            server_task = asyncio.create_task(server.main_loop())
            break
        except (OSError, SystemExit) as e:
            logger.error(f"Failed to start HTTP server on port {server_port} at try {i}, error: {e}")
    else:
        logger.error(f"Failed to start HTTP server after {max_retries} retries, exiting...")
        os._exit(-1)

    logger.info(f"HTTP server started on port {server_port}")
    return server_port, server_task
