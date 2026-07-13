from __future__ import annotations

import argparse
from pathlib import Path


BASE_COMPOSE = Path("configs/vllm/submission-score-65_06.compose.yml")


def render_adaptive_compose(image: str) -> str:
    if "@sha256:" not in image:
        raise ValueError("image must be pinned by sha256 digest")

    compose = BASE_COMPOSE.read_text()
    lines = compose.splitlines()
    rendered: list[str] = []

    for line in lines:
        if line.startswith("    image: "):
            rendered.append(f"    image: {image}")
            continue

        rendered.append(line)
        if line == '      PYTHONUNBUFFERED: "1"':
            rendered.append('      QWEN35_DECODE_WINDOW: "20"')
        elif line == "      - --max-num-batched-tokens=2048":
            rendered.append(
                "      - --scheduler-cls="
                "qwen35_adaptive.scheduler.CompletionCohortAsyncScheduler"
            )

    return "\n".join(rendered) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render_adaptive_compose(args.image))


if __name__ == "__main__":
    main()
