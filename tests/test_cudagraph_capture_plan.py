import pytest

from qwen35_serving_bench.cudagraph import (
    cohort_capture_sizes,
    default_capture_sizes,
    padded_capture_size,
    project_tbt_ms,
)
from qwen35_serving_bench.scoring import request_score


def test_default_capture_grid_pads_twenty_request_decode_batch():
    capture_sizes = default_capture_sizes(max_capture_size=128)

    assert padded_capture_size(20, capture_sizes) == 24
    assert padded_capture_size(40, capture_sizes) == 40
    assert padded_capture_size(60, capture_sizes) == 64
    assert padded_capture_size(100, capture_sizes) == 104
    assert padded_capture_size(120, capture_sizes) == 120


def test_cohort_capture_grid_adds_only_missing_twenty_request_sizes():
    default_sizes = default_capture_sizes(max_capture_size=512)
    capture_sizes = cohort_capture_sizes(
        max_capture_size=512,
        cohort_size=20,
        max_batch_size=120,
    )

    assert capture_sizes[-1] == 512
    assert set(capture_sizes) - set(default_sizes) == {20, 60, 100}
    assert len(default_sizes) == 51
    assert len(capture_sizes) == 54
    assert all(
        padded_capture_size(batch_size, capture_sizes) == batch_size
        for batch_size in (20, 40, 60, 80, 100, 120)
    )


def test_default_large_graph_coverage_is_preserved():
    capture_sizes = cohort_capture_sizes(512, cohort_size=20, max_batch_size=120)

    assert padded_capture_size(129, capture_sizes) == 136
    assert padded_capture_size(500, capture_sizes) == 512


def test_capture_sizes_reject_invalid_bounds():
    with pytest.raises(ValueError, match="max_capture_size"):
        cohort_capture_sizes(0, cohort_size=20, max_batch_size=120)
    with pytest.raises(ValueError, match="cohort_size"):
        cohort_capture_sizes(512, cohort_size=0, max_batch_size=120)
    with pytest.raises(ValueError, match="max_batch_size"):
        cohort_capture_sizes(512, cohort_size=20, max_batch_size=0)


def test_half_graph_sensitive_projection_moves_tbt_below_24_ms():
    projected = project_tbt_ms(
        base_tbt_ms=26,
        batch_size=20,
        base_padded_size=24,
        tuned_padded_size=20,
        graph_sensitive_fraction=0.5,
    )

    assert projected == pytest.approx(23.8333333333)
    assert 100 * (
        request_score(266, projected) - request_score(266, 26)
    ) > 6


def test_projection_validates_inputs():
    with pytest.raises(ValueError, match="graph_sensitive_fraction"):
        project_tbt_ms(26, 20, 24, 20, 1.1)
    with pytest.raises(ValueError, match="batch sizes"):
        project_tbt_ms(26, 20, 0, 20, 0.5)
