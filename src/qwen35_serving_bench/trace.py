from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Iterable


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def message_text(body: dict[str, Any]) -> str:
    return "\n".join(message.get("content", "") for message in body.get("messages", []))


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("No values")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(q * len(ordered)) - 1))
    return ordered[idx]


def summarize_trace(path: str | Path) -> dict[str, Any]:
    rows = list(iter_jsonl(path))
    timestamps = sorted(int(row["timestamp_ms"]) for row in rows)
    gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
    bodies = [row["body"] for row in rows]
    texts = [message_text(body) for body in bodies]
    chars = [len(text) for text in texts]
    words = [len(text.split()) for text in texts]
    approx_tokens = [max(char_count / 4.0, word_count * 1.35) for char_count, word_count in zip(chars, words)]
    max_tokens = [int(body.get("max_tokens") or 0) for body in bodies]

    return {
        "request_count": len(rows),
        "workload_types": dict(Counter(row.get("workload_type") for row in rows)),
        "models": dict(Counter(body.get("model") for body in bodies)),
        "timestamp_min_ms": min(timestamps),
        "timestamp_max_ms": max(timestamps),
        "inter_arrival_ms": {
            "min": min(gaps) if gaps else 0,
            "p50": median(gaps) if gaps else 0,
            "p90": percentile(gaps, 0.90) if gaps else 0,
            "max": max(gaps) if gaps else 0,
        },
        "message_counts": dict(Counter(len(body.get("messages", [])) for body in bodies)),
        "max_tokens": dict(Counter(max_tokens)),
        "temperature": dict(Counter(body.get("temperature") for body in bodies)),
        "seed": dict(Counter(body.get("seed") for body in bodies)),
        "chars": stat_block(chars),
        "words": stat_block(words),
        "approx_input_tokens": stat_block(approx_tokens),
        "approx_total_tokens": stat_block([prompt + output for prompt, output in zip(approx_tokens, max_tokens)]),
        "prefix_uniques": {
            str(width): len(Counter(text[:width] for text in texts))
            for width in (128, 512, 2048, 8192)
        },
    }


def stat_block(values: list[float]) -> dict[str, float]:
    return {
        "min": round(min(values), 2),
        "p50": round(percentile(values, 0.50), 2),
        "p75": round(percentile(values, 0.75), 2),
        "p90": round(percentile(values, 0.90), 2),
        "p95": round(percentile(values, 0.95), 2),
        "max": round(max(values), 2),
    }

