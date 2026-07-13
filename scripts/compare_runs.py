#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_score(run_dir: Path) -> dict[str, Any] | None:
    score_path = run_dir / "score.json"
    if not score_path.exists():
        return None
    data = json.loads(score_path.read_text(encoding="utf-8"))
    data["run"] = run_dir.name
    data["path"] = str(run_dir)
    return data


def metric(data: dict[str, Any], path: tuple[str, ...], default: float = float("inf")) -> float:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or current.get(key) is None:
            return default
        current = current[key]
    try:
        return float(current)
    except (TypeError, ValueError):
        return default


def sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    ranking_score = row.get("final_score")
    if ranking_score is None:
        ranking_score = float(row.get("ers", 0.0)) * 100.0
    return (
        float(row.get("failed_count", 10**9)),
        -float(ranking_score),
        metric(row, ("tpot_ms", "p95")),
        metric(row, ("ttft_ms", "p95")),
    )


def format_number(value: Any, precision: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.{precision}f}"


def render_table(rows: list[dict[str, Any]]) -> str:
    header = "| Rank | Run | ERS | Final | Failed | TTFT p95 | TPOT p95 | Path |\n| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |"
    lines = [header]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {run} | {ers} | {final} | {failed} | {ttft:.2f} | {tpot:.2f} | {path} |".format(
                rank=idx,
                run=str(row["run"]).replace("|", "\\|"),
                ers=format_number(row.get("ers")),
                final=format_number(row.get("final_score"), 4),
                failed=int(row.get("failed_count", 0)),
                ttft=metric(row, ("ttft_ms", "p95"), 0.0),
                tpot=metric(row, ("tpot_ms", "p95"), 0.0),
                path=str(row["path"]).replace("|", "\\|"),
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare saved candidate run score.json files.")
    parser.add_argument("--runs-dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a Markdown table.")
    args = parser.parse_args()

    rows = [score for path in sorted(args.runs_dir.glob("*")) if path.is_dir() and (score := load_score(path))]
    rows.sort(key=sort_key)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(render_table(rows) if rows else f"No score.json files found under {args.runs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
