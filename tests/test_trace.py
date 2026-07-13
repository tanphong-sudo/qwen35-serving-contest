from pathlib import Path

from qwen35_serving_bench.trace import summarize_trace


def test_summarize_trace(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        '{"request_id":0,"timestamp_ms":0,"workload_type":"conversation","body":{"model":"Qwen3.5-2B","messages":[{"role":"user","content":"hello"}],"max_tokens":200,"temperature":0,"seed":42}}\n'
        '{"request_id":1,"timestamp_ms":25,"workload_type":"conversation","body":{"model":"Qwen3.5-2B","messages":[{"role":"user","content":"hello again"}],"max_tokens":200,"temperature":0,"seed":42}}\n',
        encoding="utf-8",
    )
    summary = summarize_trace(trace)
    assert summary["request_count"] == 2
    assert summary["timestamp_max_ms"] == 25
    assert summary["max_tokens"] == {200: 2}

