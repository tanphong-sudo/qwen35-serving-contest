# Qwen3.5-2B Serving Contest

Workspace for phase 1 of the LLM inference serving contest.

The goal is to serve `Qwen/Qwen3.5-2B` on one H200 MIG instance with 18 GB VRAM, 3 CPU cores, and 8 GB RAM. The benchmark replays `input/trace-round1.jsonl` and scores every request using TTFT and TPOT, then multiplies by the GPQA accuracy gate.

## Current Trace Facts

- Requests: 120
- Time window: 0 ms to 25,475 ms
- Inter-arrival: mostly 25 ms, with a few larger gaps
- Workload type: `conversation`
- Model name in payload: `Qwen3.5-2B`
- `max_tokens`: 200 for every request
- `temperature`: 0 for every request
- Message counts: 2, 4, 6, 8, 10, and 12 messages, 20 requests each
- Approx input length: about 20k to 42k tokens by rough character/word estimate
- Exact input length with `Qwen/Qwen3.5-2B` tokenizer: 12,936 to 27,398 tokens
- Exact total length with 200 output budget: max 27,598 tokens
- Prefix: first 2,048 characters are identical across all 120 requests; token-level common prefix is 6,396 tokens with the Qwen tokenizer

The trace is six bursts of 20 requests, spaced five seconds apart. Do not duplicate current
bottleneck conclusions here; they change after each portal result and are maintained in
`docs/current-findings.md`.

## Current State

`README.md` intentionally does not duplicate mutable score, candidate or hash information.
Use `docs/current-findings.md` for the exact root artifact and next experiment, and
`docs/submission-results-log.md` for every portal result and boot failure.

## Layout

```text
.
├── docker-compose.yml
├── Dockerfile.vllm
├── Makefile
├── configs/
│   ├── sglang/
│   ├── trtllm/
│   └── vllm/
├── docs/
├── input/
│   ├── docker-compose-baseline.yml
│   └── trace-round1.jsonl
├── results/
├── scripts/
├── src/qwen35_serving_bench/
└── tests/
```

## Docs

Start here:

- [docs/contest-requirements.md](docs/contest-requirements.md) — official rules and artifact contract.
- [docs/submission-results-log.md](docs/submission-results-log.md) — scored versions and lessons.
- [docs/current-findings.md](docs/current-findings.md) — current findings and optimization order.

## Useful Commands

Analyze the trace:

```bash
make analyze
```

Run unit tests:

```bash
make test
```

Score a replay metrics file:

```bash
python3 scripts/score_ers.py --metrics results/replay.jsonl
```

Replay the trace against a local OpenAI-compatible endpoint:

```bash
python3 scripts/replay_trace.py \
  --trace input/trace-round1.jsonl \
  --base-url http://127.0.0.1:8000 \
  --output results/replay.jsonl
```

Start the default vLLM candidate:

```bash
docker compose up
```

Run one candidate end-to-end and save artifacts:

```bash
scripts/run_candidate.sh configs/vllm/submission-score-48_67.compose.yml
scripts/compare_runs.py --runs-dir results/runs
```

Generate vLLM sweep compose files:

```bash
scripts/generate_vllm_sweep.py --profile core
# Use --profile full only on a machine ready to run the larger grid.
```

## Submission Discipline

1. Change one variable per portal submission.
2. Snapshot every scored root compose before editing it.
3. Keep offline env and prefix caching in all candidates.
4. Roll back on failures or accuracy penalty.
5. Promote only when final portal score improves the confirmed best by at least 1 point.
