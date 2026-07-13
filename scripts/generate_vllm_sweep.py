#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path


IMAGE = "vllm/vllm-openai@sha256:953d3a06d5e64ab582985cd7401289d3abf2a2c14ef2158e9a84313daeec77d7"


@dataclass(frozen=True)
class VllmSweepCandidate:
    gpu_memory_utilization: str
    max_num_batched_tokens: int
    max_num_seqs: int
    max_num_partial_prefills: int
    max_long_partial_prefills: int

    @property
    def name(self) -> str:
        gpu = self.gpu_memory_utilization.replace(".", "")
        return (
            f"vllm-gpu{gpu}-bt{self.max_num_batched_tokens}-seq{self.max_num_seqs}-"
            f"pp{self.max_num_partial_prefills}-{self.max_long_partial_prefills}"
        )


CORE_CANDIDATES = [
    VllmSweepCandidate("0.95", 4096, 32, 4, 1),
    VllmSweepCandidate("0.95", 8192, 32, 4, 1),
    VllmSweepCandidate("0.95", 12288, 32, 4, 1),
    VllmSweepCandidate("0.95", 16384, 32, 4, 1),
    VllmSweepCandidate("0.95", 8192, 16, 4, 1),
    VllmSweepCandidate("0.95", 8192, 48, 4, 1),
    VllmSweepCandidate("0.95", 8192, 64, 4, 1),
    VllmSweepCandidate("0.92", 8192, 32, 4, 1),
    VllmSweepCandidate("0.97", 8192, 32, 4, 1),
    VllmSweepCandidate("0.95", 8192, 32, 2, 1),
    VllmSweepCandidate("0.95", 8192, 32, 4, 2),
]


def full_candidates() -> list[VllmSweepCandidate]:
    return [
        VllmSweepCandidate(str(gpu), batched, seqs, partial, long_partial)
        for gpu, batched, seqs, (partial, long_partial) in product(
            ("0.92", "0.95", "0.97"),
            (4096, 8192, 12288, 16384),
            (16, 32, 48, 64),
            ((2, 1), (4, 1), (4, 2)),
        )
    ]


def compose_text(candidate: VllmSweepCandidate) -> str:
    return f"""services:
  model:
    image: {IMAGE}
    environment:
      VLLM_NO_USAGE_STATS: "1"
      HF_HUB_ENABLE_HF_TRANSFER: "1"
      PYTHONUNBUFFERED: "1"
    entrypoint:
      - python3
      - -m
      - vllm.entrypoints.openai.api_server
    command:
      - --model=/model
      - --served-model-name=Qwen3.5-2B
      - --host=0.0.0.0
      - --port=8000
      - --max-model-len=65536
      - --gpu-memory-utilization={candidate.gpu_memory_utilization}
      - --tensor-parallel-size=1
      - --enable-prefix-caching
      - --enable-chunked-prefill
      - --max-num-batched-tokens={candidate.max_num_batched_tokens}
      - --max-num-seqs={candidate.max_num_seqs}
      - --max-num-partial-prefills={candidate.max_num_partial_prefills}
      - --max-long-partial-prefills={candidate.max_long_partial_prefills}
      - --long-prefill-token-threshold=4096
      - --disable-uvicorn-access-log
    ports:
      - "8000:8000"
    shm_size: "2g"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
"""


def candidate_set(profile: str) -> list[VllmSweepCandidate]:
    if profile == "core":
        return CORE_CANDIDATES
    if profile == "full":
        return full_candidates()
    raise ValueError(f"Unknown profile: {profile}")


def write_candidates(candidates: list[VllmSweepCandidate], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for candidate in candidates:
        path = output_dir / f"{candidate.name}.compose.yml"
        path.write_text(compose_text(candidate), encoding="utf-8")
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate vLLM compose files for controlled sweep runs.")
    parser.add_argument("--profile", choices=("core", "full"), default="core")
    parser.add_argument("--output-dir", type=Path, default=Path("configs/generated/vllm-sweep"))
    args = parser.parse_args()

    paths = write_candidates(candidate_set(args.profile), args.output_dir)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
