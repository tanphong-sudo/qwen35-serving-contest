#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from math import ceil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qwen35_serving_bench.scoring import accuracy_factor, final_score, record_is_failed, summarize_scores
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Score replay metrics using the contest ERS formula.")
    parser.add_argument("--metrics", type=Path, required=True, help="JSONL with ttft_ms, tpot_ms or tbt_mean_ms.")
    parser.add_argument("--accuracy", type=float, help="Optional observed GPQA accuracy as a fraction, e.g. 0.34.")
    parser.add_argument("--baseline-accuracy", type=float, default=0.4)
    args = parser.parse_args()

    records = list(iter_jsonl(args.metrics))
    summary = summarize_scores(records)
    failed_ids = [row.get("request_id") for row in records if record_is_failed(row)]
    successful = [row for row in records if not record_is_failed(row)]
    ttft = [float(row["ttft_ms"]) for row in successful]
    tpot = [
        float(row.get("tpot_ms", row.get("tbt_mean_ms")))
        for row in successful
    ]
    latency = [float(row["latency_ms"]) for row in records if row.get("latency_ms") is not None]
    result = {
        "request_count": summary.request_count,
        "failed_count": summary.failed_count,
        "failed_request_ids": failed_ids,
        "ers": summary.ers,
        "min_request_score": summary.min_score,
        "max_request_score": summary.max_score,
        "ttft_ms": stat_block(ttft),
        "tpot_ms": stat_block(tpot),
        "latency_ms": stat_block(latency),
        "note": "Local metrics use streamed content chunks, not guaranteed official tokenizer tokens.",
    }
    if args.accuracy is not None:
        multiplier = accuracy_factor(baseline_accuracy=args.baseline_accuracy, observed_accuracy=args.accuracy)
        result["accuracy_multiplier"] = multiplier
        result["final_score"] = final_score(summary.ers, multiplier)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
