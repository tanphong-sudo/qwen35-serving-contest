# Submission Results Log — Reset Round

## Reset Boundary

Leaderboard and task specification were reset on **17 July 2026**. The previous Qwen3.5-2B
scores, metrics, image digests and tuning sequence are not comparable to the new
`LiquidAI/LFM2.5-1.2B-Instruct` workload and are removed from the canonical log.

Old compose snapshots remain under `configs/` only as repository history. They must not be used
as a baseline, projection source, finalist or submission for the new round.

## Current Best

```text
No valid new-round submission yet.
```

## New-Round Submission Table

| Run | Date | Candidate | Compose SHA | Image digest | ERS | Failed | TTFT p50/p95 | TPOT mean/p50/p95 | Decision |
| --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| — | — | — | — | — | — | — | — | — | Waiting for migrated BF16 baseline |

## Post-Online Finalist Shortlist

BTC allows at most five submissions for validity review and GPQA Diamond full evaluation.
Image/digest cannot be changed after selection.

| Slot | Role | Submission | ERS | Accuracy risk | Reason retained |
| ---: | --- | --- | ---: | --- | --- |
| 1 | BF16 accuracy-safe | Empty | — | Lowest | Required safety anchor |
| 2 | Best overall ERS | Empty | — | Unknown | — |
| 3 | Best p95 TTFT | Empty | — | Unknown | Tie-break hedge |
| 4 | Best TPOT | Empty | — | Unknown | Generation-speed hedge |
| 5 | Best quantized | Empty | — | Medium/high | Performance/accuracy trade-off |

## Submission Record Template

```markdown
## Run N — Candidate Name — Result

Date:
Compose snapshot:
Compose SHA-256:
Docker Hub image digest:
Base candidate:
Single changed variable:
Hypothesis:

Validation:
- linux/amd64 image:
- anonymous/public pull:
- API module import:
- model boot/healthcheck:
- trace/scorer version:

Online metrics:
- ERS:
- scored requests:
- failed/timeout/zero-token:
- TTFT p50/p95:
- TPOT mean/p50/p95:
- primer behavior:

Accuracy risk:
Expected signature vs actual:
Decision: promote / hold finalist / reject
Rollback artifact:
```

## Legacy Lessons Retained

Only these process lessons survive the reset:

- Pin immutable image and compose hashes.
- Exact API import/boot smoke matters; optional packages can crash startup.
- Public registry visibility must be verified anonymously.
- Runner compatibility is more important than local Compose semantics.
- One-variable A/B and rollback snapshots prevent ambiguous conclusions.
- Fixed concurrency caps, custom scheduling and graph micro-tuning require workload evidence.
- Version upgrades can matter materially, but every model/runtime pair must be revalidated.
- Do not spend submissions on known rollback artifacts when leaderboard keeps historical best.

No numeric Qwen result is retained as a performance prior for LFM2.5.
