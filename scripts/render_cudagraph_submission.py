from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qwen35_serving_bench.cudagraph import cohort_capture_sizes


BASE_COMPOSE = Path("configs/vllm/submission-score-65_06.compose.yml")
INSERT_AFTER = "      - --max-num-batched-tokens=2048"


def render_cudagraph_compose(
    max_capture_size: int,
    *,
    cohort_size: int,
    max_batch_size: int,
) -> str:
    capture_sizes = cohort_capture_sizes(
        max_capture_size,
        cohort_size=cohort_size,
        max_batch_size=max_batch_size,
    )
    compilation_config = json.dumps(
        {"cudagraph_capture_sizes": capture_sizes},
        separators=(",", ":"),
    )

    base_lines = BASE_COMPOSE.read_text().splitlines()
    if base_lines.count(INSERT_AFTER) != 1:
        raise ValueError("base compose must contain exactly one insertion anchor")

    rendered: list[str] = []
    for line in base_lines:
        rendered.append(line)
        if line == INSERT_AFTER:
            rendered.append(
                f"      - '--compilation-config={compilation_config}'"
            )
    return "\n".join(rendered) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-capture-size", type=int, default=512)
    parser.add_argument("--cohort-size", type=int, default=20)
    parser.add_argument("--max-batch-size", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        render_cudagraph_compose(
            args.max_capture_size,
            cohort_size=args.cohort_size,
            max_batch_size=args.max_batch_size,
        )
    )


if __name__ == "__main__":
    main()
