# Adaptive Scheduler Runtime — Historical Rejected Experiment

Portal score for completion cohort window 20 was 49.12 versus the 65.06 base. The policy
kept ERC at 1 but increased TTFT p50 to 371 ms and TBT to 32 ms. Do not reuse this image or
sweep nearby window/defer values for submission; the implementation remains only as an
experiment record and as input to later scheduler research.

This image keeps the contest-required vLLM module path and adds a custom
`AsyncScheduler` implementation.

The policy always permits prompt prefill and first-token generation. After the first output
token, it prioritizes the most-complete decode cohort so older requests finish before newer
cohorts expand the steady-state decode batch.

Build locally:

```bash
docker build \
  -f custom_runtime/Dockerfile.adaptive \
  -t qwen35-adaptive:v0.24.0 .
```

The exact historical image is:

```text
ghcr.io/tanphong-sudo/qwen35-adaptive@sha256:8a18315745a39d54085e1d99bfbb7e5ae55e5b6fb320132c7261abfa4dfc18db
```

GitHub Actions run `29231204106` built it for `linux/amd64`, smoke-imported the scheduler,
and pushed it to GHCR. An anonymous registry manifest request returned HTTP 200 with the
same digest. The required engine flag is:

```text
--scheduler-cls=qwen35_adaptive.scheduler.CompletionCohortAsyncScheduler
```

Do not place this image into the root compose until the module import, exact vLLM API
compatibility check, compose validation, public visibility check, and anonymous manifest
inspection pass. A full GPU endpoint boot remains a portal gate because the hosted build
runner has no compatible GPU.

The manual GitHub Actions workflow `.github/workflows/build-adaptive-image.yml` builds the
amd64 image, smoke-tests the scheduler import inside the exact vLLM base, pushes to GHCR and
uploads the immutable digest. After downloading that digest, render the candidate with:

```bash
python3 scripts/render_adaptive_submission.py \
  --image ghcr.io/OWNER/qwen35-adaptive@sha256:DIGEST
```

The current package is public. Repeat the build, import smoke test, public visibility check,
and anonymous manifest inspection whenever scheduler code or the base image changes.
