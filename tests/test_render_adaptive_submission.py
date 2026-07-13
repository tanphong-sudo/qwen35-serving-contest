from pathlib import Path

import pytest

from scripts.render_adaptive_submission import render_adaptive_compose


def test_renders_digest_pinned_custom_scheduler_compose():
    image = "ghcr.io/example/qwen35-adaptive@sha256:" + "a" * 64

    compose = render_adaptive_compose(image)

    assert f"    image: {image}" in compose
    assert '      QWEN35_DECODE_WINDOW: "20"' in compose
    assert (
        "      - --scheduler-cls="
        "qwen35_adaptive.scheduler.CompletionCohortAsyncScheduler"
    ) in compose
    assert "      - --max-num-batched-tokens=2048" in compose
    assert "--max-num-seqs=" not in compose


def test_rejects_mutable_image_tag():
    with pytest.raises(ValueError, match="pinned by sha256"):
        render_adaptive_compose("ghcr.io/example/qwen35-adaptive:latest")


def test_base_compose_exists():
    assert Path("configs/vllm/submission-score-65_06.compose.yml").is_file()
