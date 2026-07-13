# Adaptive Scheduler Runtime

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

The submission image must be pushed to a public registry and pinned by digest before a
compose candidate is created. The required engine flag is:

```text
--scheduler-cls=qwen35_adaptive.scheduler.CompletionCohortAsyncScheduler
```

Do not place this image into the root compose until the module import and OpenAI endpoint
smoke tests pass against the exact pinned image.

The manual GitHub Actions workflow `.github/workflows/build-adaptive-image.yml` builds the
amd64 image, smoke-tests the scheduler import inside the exact vLLM base, pushes to GHCR and
uploads the immutable digest. After downloading that digest, render the candidate with:

```bash
python3 scripts/render_adaptive_submission.py \
  --image ghcr.io/OWNER/qwen35-adaptive@sha256:DIGEST
```

GHCR packages may be private after the first push. Set the package visibility to public and
verify an anonymous manifest inspection before using the digest in a contest submission.
