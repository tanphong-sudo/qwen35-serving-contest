from qwen35_serving_bench.scoring import accuracy_factor, request_score, summarize_scores


def test_request_score_best_case():
    assert request_score(100, 20) == 1.0


def test_request_score_ceiling_case():
    assert request_score(1500, 45) == 0.0


def test_accuracy_factor_piecewise():
    assert accuracy_factor(delta=0.10) == 1.0
    assert accuracy_factor(delta=0.16) == 0.0
    assert round(accuracy_factor(delta=0.13), 6) == 0.5


def test_summarize_scores_counts_failures():
    summary = summarize_scores(
        [
            {"ttft_ms": 100, "tpot_ms": 20, "output_tokens": 10},
            {"timeout": True, "output_tokens": 0},
        ]
    )
    assert summary.request_count == 2
    assert summary.failed_count == 1
    assert summary.ers == 0.5


def test_summarize_scores_treats_missing_tpot_as_failed():
    summary = summarize_scores(
        [
            {"request_id": 0, "ttft_ms": 100, "tpot_ms": 20, "output_tokens": 10},
            {"request_id": 1, "ttft_ms": 100, "output_tokens": 1},
        ]
    )

    assert summary.request_count == 2
    assert summary.failed_count == 1
    assert summary.ers == 0.5
