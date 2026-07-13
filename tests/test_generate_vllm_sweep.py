import importlib.util
import sys
from pathlib import Path


def load_generate_vllm_sweep():
    path = Path(__file__).resolve().parents[1] / "scripts" / "generate_vllm_sweep.py"
    spec = importlib.util.spec_from_file_location("generate_vllm_sweep", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_core_profile_has_controlled_number_of_candidates():
    module = load_generate_vllm_sweep()

    candidates = module.candidate_set("core")

    assert len(candidates) == 11
    assert len({candidate.name for candidate in candidates}) == 11


def test_full_profile_matches_declared_grid_size():
    module = load_generate_vllm_sweep()

    assert len(module.candidate_set("full")) == 3 * 4 * 4 * 3


def test_compose_text_contains_required_vllm_flags():
    module = load_generate_vllm_sweep()
    candidate = module.VllmSweepCandidate("0.95", 8192, 32, 4, 1)

    text = module.compose_text(candidate)

    assert "--model=/model" in text
    assert "--served-model-name=Qwen3.5-2B" in text
    assert "--max-model-len=65536" in text
    assert "--enable-prefix-caching" in text
    assert "--enable-chunked-prefill" in text
    assert "--max-num-batched-tokens=8192" in text
    assert "--max-num-seqs=32" in text
