from __future__ import annotations

import argparse
import re
from pathlib import Path


BASE_COMPOSE = Path("configs/vllm/submission-score-65_06.compose.yml")
BASE_IMAGE = (
    "vllm/vllm-openai@sha256:"
    "3de11aaf1d2aa1c6245a93e9279cc10af6d0b9f5eb3b34704fbd099a8ac42c7d"
)
IMAGE_PATTERN = re.compile(r"vllm/vllm-openai@sha256:[0-9a-f]{64}")


def render_image_upgrade(image: str) -> str:
    if IMAGE_PATTERN.fullmatch(image) is None:
        raise ValueError("image must be a digest-pinned official vLLM image")

    base = BASE_COMPOSE.read_text()
    image_line = f"    image: {BASE_IMAGE}"
    if base.count(image_line) != 1:
        raise ValueError("base compose must contain exactly one expected image")
    return base.replace(image_line, f"    image: {image}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render_image_upgrade(args.image))


if __name__ == "__main__":
    main()
