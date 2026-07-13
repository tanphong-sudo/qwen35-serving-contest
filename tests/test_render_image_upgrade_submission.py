import subprocess
import sys
from pathlib import Path

import pytest

from qwen35_serving_bench.scoring import request_score
from scripts.render_image_upgrade_submission import BASE_IMAGE, render_image_upgrade


V025_IMAGE = (
    "vllm/vllm-openai@sha256:"
    "1a62fd4ad863259ec206e0d2b9fb24eb5d67b4deff87a1b2ae7889fc7f9ab23e"
)


def test_renderer_changes_only_the_image():
    rendered = render_image_upgrade(V025_IMAGE)
    base = Path("configs/vllm/submission-score-65_06.compose.yml").read_text()

    assert rendered != base
    assert rendered.replace(V025_IMAGE, BASE_IMAGE) == base


def test_renderer_rejects_mutable_or_non_official_images():
    with pytest.raises(ValueError, match="digest-pinned official"):
        render_image_upgrade("vllm/vllm-openai:v0.25.0")
    with pytest.raises(ValueError, match="digest-pinned official"):
        render_image_upgrade("example/image@sha256:" + "a" * 64)


def test_upstream_tpot_ratio_clears_submission_gate():
    projected_tbt_ms = 26 * 9.68 / 10.33
    projected_gain = 100 * (
        request_score(266, projected_tbt_ms) - request_score(266, 26)
    )

    assert projected_tbt_ms == pytest.approx(24.363988, abs=0.000001)
    assert projected_gain == pytest.approx(5.187598, abs=0.000001)
    assert 65.06 + projected_gain > 70


def test_cli_writes_exact_snapshot(tmp_path):
    output = tmp_path / "candidate.yml"
    subprocess.run(
        [
            sys.executable,
            "scripts/render_image_upgrade_submission.py",
            "--image",
            V025_IMAGE,
            "--output",
            str(output),
        ],
        check=True,
    )

    assert output.read_text() == Path(
        "configs/vllm/submission-v025-upgrade.compose.yml"
    ).read_text()
