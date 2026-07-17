# LFM2.5-1.2B Inference Serving Contest

Workspace for the reset online round of the LLM inference optimization contest.

The current target is `LiquidAI/LFM2.5-1.2B-Instruct` on one H200 MIG slice with roughly
18 GB VRAM, 3 CPU cores, and 8 GB RAM. The official text describes a deterministic
Poisson-arrival multi-turn workload with 70 conversations, 330 scored requests, and 15 warm-up
primer conversations. Whether the primer conversations are additional to or included in the 70
must be confirmed from the new public trace. Online submissions are ranked by ERS; accuracy is
evaluated only after the online round on at most five submissions selected by the team.

## Reset Status — 17 July 2026

The repository still contains implementation artifacts from the retired Qwen3.5 round.
They are preserved only for engineering history and must not be submitted unchanged.

Known-invalid current artifacts:

- `docker-compose.yml` serves `Qwen3.5-2B`, uses a GHCR image, and is not a valid new-round submission.
- `input/trace-round1.jsonl` contains the old 120-request trace, not `trace_grading_public.jsonl`.
- `src/qwen35_serving_bench/scoring.py` still uses the old 100–1500 ms TTFT and 20–45 ms TPOT bounds.
- Existing scored compose snapshots and custom runtimes belong to the retired leaderboard.

The next implementation task is a clean migration to the new model, trace, scoring constants,
Docker Hub artifact contract, and Ubuntu 24.04 / driver 590.x runtime.

## Documentation

Start here:

- `docs/official-source-2026-07-17.txt` — raw textual snapshot of the official task/rules and overview supplied on 17 July 2026.
- `docs/contest-requirements.md` — normalized current rules and submission contract.
- `docs/current-findings.md` — reset audit, retained lessons, invalid assumptions, and optimization plan.
- `docs/submission-results-log.md` — new-round submission history and five-finalist shortlist.

## Repository Layout

```text
.
├── docker-compose.yml                 # legacy until migration is completed
├── configs/                           # historical snapshots; not current candidates
├── custom_runtime/                    # archived Qwen experiments
├── docs/
├── input/                             # currently contains the retired trace
├── results/
├── scripts/                           # some scripts still encode old-round assumptions
├── src/qwen35_serving_bench/          # legacy package name and scorer
└── tests/
```

## Safe Commands

Run the current regression suite while migrating:

```bash
make test
```

Validate a newly rendered compose file:

```bash
docker compose -f docker-compose.yml config -q
```

Do not use the current trace analyzer, replay score, sweep output, or root compose as official
new-round evidence until the migration checklist in `docs/current-findings.md` is complete.
