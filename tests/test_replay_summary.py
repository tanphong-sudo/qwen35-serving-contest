from scripts.replay_trace import replay_summary


def test_replay_summary_reports_tail_metrics_and_failures():
    rows = [
        {"request_id": 0, "ttft_ms": 100.0, "tpot_ms": 20.0, "latency_ms": 300.0, "output_tokens": 10, "error": None},
        {"request_id": 1, "ttft_ms": 200.0, "tpot_ms": 25.0, "latency_ms": 500.0, "output_tokens": 10, "error": None},
        {"request_id": 2, "ttft_ms": None, "tpot_ms": None, "latency_ms": 50.0, "output_tokens": 0, "error": "boom"},
    ]

    summary = replay_summary(rows)

    assert summary["request_count"] == 3
    assert summary["failed_count"] == 1
    assert summary["failed_request_ids"] == [2]
    assert summary["ttft_ms"]["p95"] == 200.0
    assert summary["tpot_ms"]["p50"] == 20.0
