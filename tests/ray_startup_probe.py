#!/usr/bin/env python3
"""Minimal local-Ray health probe for a multi-GPU ROPD host.

This deliberately does not import ROPD, vLLM, or load a model.  It verifies
that Ray can start a local cluster and schedule one CUDA task per requested
GPU.  Run from the repository root, for example:

    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --no-sync python tests/ray_startup_probe.py --expected-gpus 4
"""

from __future__ import annotations

import argparse
import os
import socket

import ray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-gpus",
        type=int,
        default=None,
        help="Minimum number of GPUs Ray must expose (defaults to Ray's detected count).",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


@ray.remote(num_cpus=1)
def cpu_probe() -> dict[str, str]:
    return {"host": socket.gethostname(), "pid": str(os.getpid())}


@ray.remote(num_gpus=1)
def gpu_probe() -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Ray scheduled a GPU task but torch.cuda.is_available() is False")
    return {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "ray_visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_visible_gpu_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
    }


def main() -> None:
    args = parse_args()
    print(f"launcher CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print("Starting local Ray cluster...", flush=True)
    ray.init(include_dashboard=False, log_to_driver=True, runtime_env={"working_dir": None})
    try:
        cluster = ray.cluster_resources()
        available = ray.available_resources()
        detected_gpus = int(cluster.get("GPU", 0))
        expected_gpus = detected_gpus if args.expected_gpus is None else args.expected_gpus

        print(f"Ray cluster resources: {cluster}")
        print(f"Ray available resources: {available}")
        if detected_gpus < expected_gpus:
            raise RuntimeError(f"Ray exposes {detected_gpus} GPU(s), expected at least {expected_gpus}")
        if expected_gpus < 1:
            raise RuntimeError("No GPU was detected by Ray")

        print(f"CPU probe: {ray.get(cpu_probe.remote(), timeout=args.timeout_seconds)}")
        results = ray.get(
            [gpu_probe.remote() for _ in range(expected_gpus)], timeout=args.timeout_seconds
        )
        for rank, result in enumerate(results):
            print(f"GPU probe {rank}: {result}")
        print("PASS: Ray started and scheduled all requested GPU tasks.")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
