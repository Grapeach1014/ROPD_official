#!/usr/bin/env python3
"""Benchmark end-to-end latency for models routed through an Anthropic gateway.

The benchmark is deliberately sequential: it measures one user-facing request
at a time, avoiding cross-model contention caused by local client concurrency.
Credentials are read from the environment or requested interactively and are
never written to the result files.
"""

from __future__ import annotations

import argparse
import getpass
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from anthropic import Anthropic
import httpx


DEFAULT_MODELS = (
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
)

PROMPT = """请用简体中文写一段 180 到 220 个汉字的说明，主题是“为什么在比较多个大模型 API 路由时，除了平均响应时间，还应记录尾延迟和成功率”。要求：只输出一段正文；不要使用标题、列表、Markdown、代码、引用或公式；不要复述本题要求；内容应具体、通顺，并在结尾给出一句简短建议。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model IDs to route to; defaults to the seven requested Claude IDs.",
    )
    parser.add_argument("--runs", type=int, default=5, help="Measured sequential requests per model (default: 5).")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Unreported warm-up requests per model (default: 1).")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum generated tokens per request (default: 512).")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-request timeout in seconds (default: 180).")
    parser.add_argument(
        "--base-url",
        default=os.getenv("ANTHROPIC_BASE_URL"),
        help="Gateway base URL; defaults to ANTHROPIC_BASE_URL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/model_route_latency"),
        help="Directory for JSON and Markdown results.",
    )
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2 so the result is not a one-off measurement.")
    if args.warmup_runs < 0 or args.max_tokens < 1 or args.timeout <= 0:
        parser.error("warmup-runs must be >= 0, max-tokens must be >= 1, and timeout must be > 0.")
    if not args.base_url:
        parser.error("Set ANTHROPIC_BASE_URL or pass --base-url.")
    return args


def percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile without requiring NumPy."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def response_text(response: Any) -> str:
    return "".join(getattr(block, "text", "") for block in response.content if getattr(block, "type", None) == "text")


def request_once(
    client: Anthropic,
    gateway_client: httpx.Client,
    *,
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        if model.startswith("gpt-"):
            # The gateway accepts GPT routes on its Messages endpoint but
            # requires the OpenAI-style completion-token field. Anthropic's
            # SDK always emits max_tokens, so use the same endpoint directly
            # for this route family.
            http_response = gateway_client.post(
                "messages",
                json={
                    "model": model,
                    "max_completion_tokens": max_tokens,
                    "messages": [{"role": "user", "content": PROMPT}],
                },
            )
            http_response.raise_for_status()
            payload = http_response.json()
            text = "".join(
                str(block.get("text", ""))
                for block in payload.get("content", [])
                if isinstance(block, dict) and block.get("type") == "text"
            )
            usage = payload.get("usage") or {}
            return {
                "ok": True,
                "latency_seconds": round(time.perf_counter() - started, 4),
                "output_characters": len(text),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "stop_reason": payload.get("stop_reason"),
                "response_preview": text[:160],
            }

        request_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": PROMPT}],
        }
        # The gateway rejects temperature for these newer Opus routes.
        if model not in {"claude-opus-4-8", "claude-opus-4-7"}:
            request_kwargs["temperature"] = 0
        response = client.messages.create(**request_kwargs)
    except Exception as exc:  # Record gateway and model routing failures per model.
        return {
            "ok": False,
            "latency_seconds": round(time.perf_counter() - started, 4),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    text = response_text(response)
    usage = getattr(response, "usage", None)
    return {
        "ok": True,
        "latency_seconds": round(time.perf_counter() - started, 4),
        "output_characters": len(text),
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "stop_reason": getattr(response, "stop_reason", None),
        "response_preview": text[:160],
    }


def summarize(model: str, attempts: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [item for item in attempts if item["ok"]]
    summary: dict[str, Any] = {
        "model": model,
        "attempts": len(attempts),
        "successes": len(successful),
        "success_rate": len(successful) / len(attempts),
    }
    if successful:
        latencies = [float(item["latency_seconds"]) for item in successful]
        summary.update(
            {
                "avg_latency_seconds": statistics.fmean(latencies),
                "median_latency_seconds": statistics.median(latencies),
                "p95_latency_seconds": percentile(latencies, 0.95),
                "min_latency_seconds": min(latencies),
                "max_latency_seconds": max(latencies),
                "avg_output_characters": statistics.fmean(int(item["output_characters"]) for item in successful),
                "avg_output_tokens": statistics.fmean(
                    int(item["output_tokens"]) for item in successful if item.get("output_tokens") is not None
                )
                if any(item.get("output_tokens") is not None for item in successful)
                else None,
            }
        )
    return summary


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Anthropic Gateway 路由延迟基准",
        "",
        f"- 时间（UTC）：{payload['started_at_utc']}",
        f"- 每模型预热：{payload['warmup_runs']} 次（不计入统计）",
        f"- 每模型计量：{payload['runs']} 次，串行执行",
        f"- 输出约束：中文约 200 字，completion token 上限={payload['max_tokens']}；在路由支持时使用 `temperature=0`",
        "",
        "| 模型 | 成功/尝试 | 成功率 | Avg (s) | P50 (s) | P95 (s) | Max (s) | 平均输出字符 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["summaries"]:
        if item["successes"]:
            lines.append(
                "| {model} | {successes}/{attempts} | {rate:.0%} | {avg:.2f} | {p50:.2f} | {p95:.2f} | {max_:.2f} | {chars:.0f} |".format(
                    model=item["model"],
                    successes=item["successes"],
                    attempts=item["attempts"],
                    rate=item["success_rate"],
                    avg=item["avg_latency_seconds"],
                    p50=item["median_latency_seconds"],
                    p95=item["p95_latency_seconds"],
                    max_=item["max_latency_seconds"],
                    chars=item["avg_output_characters"],
                )
            )
        else:
            lines.append(f"| {item['model']} | 0/{item['attempts']} | 0% | — | — | — | — | — |")
    lines.extend(["", "详细的每次请求耗时、错误与输出预览保存在同名 JSON 文件。"])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    token = os.getenv("ANTHROPIC_AUTH_TOKEN") or getpass.getpass("ANTHROPIC_AUTH_TOKEN: ")
    if not token:
        print("ANTHROPIC_AUTH_TOKEN is required.", file=sys.stderr)
        return 2

    client = Anthropic(api_key=token, base_url=args.base_url, timeout=args.timeout)
    gateway_client = httpx.Client(
        base_url=args.base_url.rstrip("/") + "/",
        timeout=args.timeout,
        headers={
            "x-api-key": token,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    started_at = datetime.now(timezone.utc)
    all_results: dict[str, list[dict[str, Any]]] = {}
    for model in args.models:
        print(f"\n=== {model}: warmup={args.warmup_runs}, measured={args.runs} ===", flush=True)
        for warmup_index in range(args.warmup_runs):
            result = request_once(client, gateway_client, model=model, max_tokens=args.max_tokens)
            state = "ok" if result["ok"] else f"failed: {result['error_type']}"
            print(f"warmup {warmup_index + 1}/{args.warmup_runs}: {result['latency_seconds']:.2f}s ({state})", flush=True)

        measured: list[dict[str, Any]] = []
        for run_index in range(args.runs):
            result = request_once(client, gateway_client, model=model, max_tokens=args.max_tokens)
            measured.append(result)
            state = "ok" if result["ok"] else f"failed: {result['error_type']}"
            print(f"run {run_index + 1}/{args.runs}: {result['latency_seconds']:.2f}s ({state})", flush=True)
        all_results[model] = measured

    gateway_client.close()
    payload: dict[str, Any] = {
        "started_at_utc": started_at.isoformat(),
        "base_url": args.base_url,
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "prompt": PROMPT,
        "results": all_results,
        "summaries": [summarize(model, all_results[model]) for model in args.models],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    json_path = args.output_dir / f"anthropic_route_latency_{stamp}.json"
    markdown_path = args.output_dir / f"anthropic_route_latency_{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(payload), encoding="utf-8")

    print("\n" + markdown_report(payload))
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
