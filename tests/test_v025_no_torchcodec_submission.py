from pathlib import Path


OFFICIAL_V025_IMAGE = (
    "vllm/vllm-openai@sha256:"
    "1a62fd4ad863259ec206e0d2b9fb24eb5d67b4deff87a1b2ae7889fc7f9ab23e"
)
SANITIZED_V025_IMAGE = (
    "ghcr.io/tanphong-sudo/qwen35-adaptive@sha256:"
    "a80e8468a978aba2e39ebb3dfa18d858fc3af3cefa17619736cd2852947c7c11"
)


def test_root_matches_sanitized_v025_snapshot():
    root = Path("docker-compose.yml").read_text()
    snapshot = Path(
        "configs/vllm/submission-v025-no-torchcodec.compose.yml"
    ).read_text()

    assert root == snapshot


def test_candidate_only_replaces_failed_v025_image():
    candidate = Path("docker-compose.yml").read_text()
    failed_v025 = Path("configs/vllm/submission-v025-upgrade.compose.yml").read_text()

    assert candidate.replace(SANITIZED_V025_IMAGE, OFFICIAL_V025_IMAGE) == failed_v025


def test_repair_image_removes_only_broken_optional_package():
    dockerfile = Path("custom_runtime/Dockerfile.v025-no-torchcodec").read_text()

    assert f"FROM {OFFICIAL_V025_IMAGE}" in dockerfile
    assert "python3 -m pip uninstall -y torchcodec" in dockerfile
    assert 'find_spec("torchcodec") is None' in dockerfile
    assert 'torch.version.cuda == "12.9"' in dockerfile
    assert 'vllm.__version__ == "0.25.0"' in dockerfile
