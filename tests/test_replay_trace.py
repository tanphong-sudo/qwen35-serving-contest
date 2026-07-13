import importlib.util
from pathlib import Path


def load_replay_trace():
    path = Path(__file__).resolve().parents[1] / "scripts" / "replay_trace.py"
    spec = importlib.util.spec_from_file_location("replay_trace", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_replay_summary_reports_failed_ids_and_success_latency_stats():
    module = load_replay_trace()

    summary = module.replay_summary(
        [
            {"request_id": 0, "ttft_ms": 100.0, "tpot_ms": 20.0, "latency_ms": 300.0, "output_tokens": 10},
            {"request_id": 1, "ttft_ms": 100.0, "tpot_ms": None, "latency_ms": 150.0, "output_tokens": 1},
            {"request_id": 2, "ttft_ms": None, "tpot_ms": None, "latency_ms": 200.0, "output_tokens": 0},
        ]
    )

    assert summary["failed_count"] == 2
    assert summary["failed_request_ids"] == [1, 2]
    assert summary["ttft_ms"]["p95"] == 100.0
    assert summary["tpot_ms"]["p95"] == 20.0
    assert summary["latency_ms"]["max"] == 300.0


def test_parse_sse_payload_ignores_done_and_parses_data():
    module = load_replay_trace()

    assert module.parse_sse_payload("data: [DONE]") is None
    assert module.parse_sse_payload('data: {"choices":[{"delta":{"content":"x"}}]}') == {
        "choices": [{"delta": {"content": "x"}}]
    }
