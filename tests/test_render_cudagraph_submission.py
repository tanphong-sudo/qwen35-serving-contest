import json
import subprocess
import sys

import pytest

import scripts.render_cudagraph_submission as renderer
from scripts.render_cudagraph_submission import render_cudagraph_compose


def test_renderer_adds_only_explicit_capture_grid():
    rendered = render_cudagraph_compose(
        max_capture_size=512,
        cohort_size=20,
        max_batch_size=120,
    )

    assert "vllm/vllm-openai@sha256:" in rendered
    assert "qwen35_adaptive" not in rendered
    assert rendered.count("--compilation-config=") == 1

    config_line = next(
        line.strip().removeprefix("- ").strip("'")
        for line in rendered.splitlines()
        if "--compilation-config=" in line
    )
    payload = json.loads(config_line.split("=", 1)[1])
    sizes = payload["cudagraph_capture_sizes"]

    assert sizes[:5] == [1, 2, 4, 8, 16]
    assert sizes[-1] == 512
    assert len(sizes) == 54
    assert all(size in sizes for size in (20, 40, 60, 80, 100, 120))


def test_renderer_is_deterministic():
    assert render_cudagraph_compose(
        512, cohort_size=20, max_batch_size=120
    ) == render_cudagraph_compose(
        512, cohort_size=20, max_batch_size=120
    )


def test_renderer_rejects_base_without_exact_anchor(tmp_path, monkeypatch):
    base = tmp_path / "base.yml"
    base.write_text("services: {}\n")
    monkeypatch.setattr(renderer, "BASE_COMPOSE", base)

    with pytest.raises(ValueError, match="exactly one insertion anchor"):
        render_cudagraph_compose(512, cohort_size=20, max_batch_size=120)


def test_cli_writes_candidate(tmp_path):
    output = tmp_path / "candidate.yml"

    subprocess.run(
        [
            sys.executable,
            "scripts/render_cudagraph_submission.py",
            "--output",
            str(output),
        ],
        check=True,
    )

    assert output.read_text() == render_cudagraph_compose(
        512, cohort_size=20, max_batch_size=120
    )


def test_cli_requires_explicit_output():
    result = subprocess.run(
        [sys.executable, "scripts/render_cudagraph_submission.py"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--output" in result.stderr
