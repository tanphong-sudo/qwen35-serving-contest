from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


F_TTFT_MS = 100.0
C_TTFT_MS = 1500.0
F_TPOT_MS = 20.0
C_TPOT_MS = 45.0
GAMMA = 2.0
TTFT_WEIGHT = 0.5


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def latency_component(value_ms: float, floor_ms: float, ceiling_ms: float, gamma: float = GAMMA) -> float:
    x = clamp((ceiling_ms - value_ms) / (ceiling_ms - floor_ms))
    return x**gamma


def request_score(
    ttft_ms: float,
    tpot_ms: float,
    *,
    weight: float = TTFT_WEIGHT,
    gamma: float = GAMMA,
) -> float:
    s_ttft = latency_component(ttft_ms, F_TTFT_MS, C_TTFT_MS, gamma)
    s_tpot = latency_component(tpot_ms, F_TPOT_MS, C_TPOT_MS, gamma)
    return weight * s_ttft + (1.0 - weight) * s_tpot


def accuracy_factor(*, baseline_accuracy: float = 0.4, observed_accuracy: float | None = None, delta: float | None = None) -> float:
    if delta is None:
        if observed_accuracy is None:
            raise ValueError("Provide observed_accuracy or delta")
        delta = baseline_accuracy - observed_accuracy
    if delta <= 0.10:
        return 1.0
    if delta >= 0.16:
        return 0.0
    return 1.0 - (delta - 0.10) / 0.06


def final_score(ers: float, accuracy_multiplier: float) -> float:
    return 100.0 * ers * accuracy_multiplier


def record_is_failed(record: Mapping[str, object]) -> bool:
    if record.get("error") or record.get("timeout"):
        return True
    if record.get("ttft_ms") is None or record.get("tpot_ms", record.get("tbt_mean_ms")) is None:
        return True
    output_tokens = record.get("output_tokens", record.get("generated_tokens"))
    if output_tokens is not None:
        try:
            return int(output_tokens) <= 0
        except (TypeError, ValueError):
            return True
    return False


def score_record(record: Mapping[str, object]) -> float:
    if record_is_failed(record):
        return 0.0

    ttft = record.get("ttft_ms")
    tpot = record.get("tpot_ms", record.get("tbt_mean_ms"))
    if ttft is None or tpot is None:
        raise ValueError(f"Record missing ttft_ms or tpot_ms/tbt_mean_ms: {record}")
    return request_score(float(ttft), float(tpot))


@dataclass(frozen=True)
class ErsSummary:
    request_count: int
    failed_count: int
    ers: float
    min_score: float
    max_score: float


def summarize_scores(records: Iterable[Mapping[str, object]]) -> ErsSummary:
    rows = list(records)
    if not rows:
        raise ValueError("No records to score")
    scores = [score_record(row) for row in rows]
    failed = sum(1 for row in rows if record_is_failed(row))
    return ErsSummary(
        request_count=len(rows),
        failed_count=failed,
        ers=sum(scores) / len(scores),
        min_score=min(scores),
        max_score=max(scores),
    )
