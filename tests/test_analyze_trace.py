import importlib.util
from collections import UserDict
from pathlib import Path


def load_analyze_trace():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_trace.py"
    spec = importlib.util.spec_from_file_location("analyze_trace", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_token_count_handles_batch_encoding_like_mapping():
    module = load_analyze_trace()
    tokenized = UserDict({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]})

    assert module.token_count(tokenized) == 3


def test_token_count_handles_plain_token_list():
    module = load_analyze_trace()

    assert module.token_count([1, 2, 3, 4]) == 4


def test_common_prefix_length():
    module = load_analyze_trace()

    assert module.common_prefix_length([[1, 2, 3], [1, 2, 4], [1, 2]]) == 2
