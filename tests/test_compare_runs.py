import importlib.util
from pathlib import Path


def load_compare_runs():
    path = Path(__file__).resolve().parents[1] / "scripts" / "compare_runs.py"
    spec = importlib.util.spec_from_file_location("compare_runs", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sort_key_prioritizes_failures_before_score():
    module = load_compare_runs()
    clean_lower_score = {"failed_count": 0, "ers": 0.7, "ttft_ms": {"p95": 100}, "tpot_ms": {"p95": 20}}
    failed_higher_score = {"failed_count": 1, "ers": 0.9, "ttft_ms": {"p95": 100}, "tpot_ms": {"p95": 20}}

    assert module.sort_key(clean_lower_score) < module.sort_key(failed_higher_score)


def test_sort_key_uses_final_score_when_accuracy_is_available():
    module = load_compare_runs()
    higher_ers_lower_accuracy = {
        "failed_count": 0,
        "ers": 0.9,
        "final_score": 60.0,
        "ttft_ms": {"p95": 100},
        "tpot_ms": {"p95": 20},
    }
    lower_ers_higher_accuracy = {
        "failed_count": 0,
        "ers": 0.8,
        "final_score": 70.0,
        "ttft_ms": {"p95": 100},
        "tpot_ms": {"p95": 20},
    }

    assert module.sort_key(lower_ers_higher_accuracy) < module.sort_key(higher_ers_lower_accuracy)


def test_render_table_escapes_markdown_pipes():
    module = load_compare_runs()

    rendered = module.render_table(
        [
            {
                "run": "a|b",
                "path": "results/a|b",
                "ers": 0.5,
                "final_score": 50.0,
                "failed_count": 0,
                "ttft_ms": {"p95": 100},
                "tpot_ms": {"p95": 20},
            }
        ]
    )

    assert "a\\|b" in rendered
    assert "results/a\\|b" in rendered
