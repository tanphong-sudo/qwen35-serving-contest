#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from math import ceil
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qwen35_serving_bench.trace import iter_jsonl


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, ceil(q * len(ordered)) - 1))
    return ordered[idx]


def stat_block(values: list[float]) -> dict[str, float | None]:
    return {
        "min": percentile(values, 0.0),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": percentile(values, 1.0),
    }


def replay_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    failed_ids = [
        row.get("request_id")
        for row in results
        if row.get("error") or not row.get("output_tokens") or row.get("ttft_ms") is None or row.get("tpot_ms") is None
    ]
    failed_id_set = set(failed_ids)
    successful = [row for row in results if row.get("request_id") not in failed_id_set]
    ttft = [float(row["ttft_ms"]) for row in successful]
    tpot = [float(row["tpot_ms"]) for row in successful]
    latency = [float(row["latency_ms"]) for row in results if row.get("latency_ms") is not None]
    return {
        "request_count": len(results),
        "failed_count": len(failed_ids),
        "failed_request_ids": failed_ids,
        "ttft_ms": stat_block(ttft),
        "tpot_ms": stat_block(tpot),
        "latency_ms": stat_block(latency),
        "note": "TPOT/output_tokens use streamed content chunks, not guaranteed tokenizer tokens.",
    }


def parse_sse_payload(line: str) -> dict[str, Any] | None:
    if not line.startswith("data: "):
        return None
    data = line[len("data: ") :].strip()
    if not data or data == "[DONE]":
        return None
    return json.loads(data)


def chunk_has_content(payload: dict[str, Any]) -> bool:
    choices = payload.get("choices") or []
    if not choices:
        return False
    delta = choices[0].get("delta") or {}
    return bool(delta.get("content"))


async def replay_one(client: httpx.AsyncClient, row: dict[str, Any], start_time: float, url: str) -> dict[str, Any]:
    target_time = start_time + float(row["timestamp_ms"]) / 1000.0
    await asyncio.sleep(max(0.0, target_time - time.perf_counter()))

    body = dict(row["body"])
    body["stream"] = True
    request_start = time.perf_counter()
    first_token_at: float | None = None
    token_event_times: list[float] = []
    error: str | None = None

    try:
        async with client.stream("POST", url, json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                try:
                    payload = parse_sse_payload(line)
                except json.JSONDecodeError:
                    continue
                if payload is None:
                    continue
                if chunk_has_content(payload):
                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now
                    token_event_times.append(now)
    except Exception as exc:  # noqa: BLE001 - keep replay resilient.
        error = repr(exc)

    completed_at = time.perf_counter()
    output_events = len(token_event_times)
    ttft_ms = None if first_token_at is None else (first_token_at - request_start) * 1000.0
    if len(token_event_times) >= 2:
        tpot_ms = ((token_event_times[-1] - token_event_times[0]) / (len(token_event_times) - 1)) * 1000.0
    else:
        tpot_ms = None

    return {
        "request_id": row.get("request_id"),
        "scheduled_timestamp_ms": row.get("timestamp_ms"),
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "latency_ms": (completed_at - request_start) * 1000.0,
        "output_tokens": output_events,
        "error": error,
        "note": "output_tokens counts streamed content chunks, not guaranteed tokenizer tokens",
    }


async def replay(args: argparse.Namespace) -> None:
    rows = list(iter_jsonl(args.trace))
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    timeout = httpx.Timeout(args.timeout, read=args.timeout)
    limits = httpx.Limits(max_connections=args.max_connections, max_keepalive_connections=args.max_connections)
    start_time = time.perf_counter()

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        tasks = [replay_one(client, row, start_time, url) for row in rows]
        results = await asyncio.gather(*tasks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for result in sorted(results, key=lambda row: row["request_id"]):
            handle.write(json.dumps(result) + "\n")
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(replay_summary(results), indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay the trace against an OpenAI-compatible chat completions endpoint.")
    parser.add_argument("--trace", type=Path, default=Path("input/trace-round1.jsonl"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("results/replay.jsonl"))
    parser.add_argument("--summary-output", type=Path, help="Optional path to write replay latency summary JSON.")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-connections", type=int, default=256)
    args = parser.parse_args()

    asyncio.run(replay(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
