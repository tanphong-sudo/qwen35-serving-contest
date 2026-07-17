# Legacy Custom Runtime Archive

This directory contains rejected Qwen3.5-round experiments. The leaderboard, model, trace,
scoring bounds, host runtime and registry contract were reset on 17 July 2026.

Do not build, publish or submit these runtimes for the current LFM2.5 round. In particular,
`Dockerfile.adaptive`, the completion-cohort scheduler and the old GHCR images are historical
artifacts only.

The only retained lesson is procedural: custom images must be built for `linux/amd64`, pin an
immutable base digest, smoke-import the exact vLLM API module, verify public anonymous pull,
and demonstrate a workload-specific gain before portal submission. Current policy and the new
candidate roadmap live in `docs/current-findings.md`.
