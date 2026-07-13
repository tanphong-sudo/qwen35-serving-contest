#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <compose-file> [output-dir]" >&2
  exit 2
fi

COMPOSE_FILE="$1"
if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "Compose file not found: $COMPOSE_FILE" >&2
  exit 2
fi

TRACE="${TRACE:-input/trace-round1.jsonl}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
TOKENIZER="${TOKENIZER:-/model}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"
REPLAY_TIMEOUT="${REPLAY_TIMEOUT:-300}"
MAX_CONNECTIONS="${MAX_CONNECTIONS:-256}"
PYTHON="${PYTHON:-python3}"

RUN_NAME="$(date -u +%Y%m%dT%H%M%SZ)-$(basename "$COMPOSE_FILE" .yml | tr -cs '[:alnum:]-' '-')"
OUT_DIR="${2:-results/runs/$RUN_NAME}"
PROJECT_NAME="qwen35-${RUN_NAME//[^a-zA-Z0-9]/}"

mkdir -p "$OUT_DIR"
cp "$COMPOSE_FILE" "$OUT_DIR/compose.yml"

{
  echo "# Candidate Run"
  echo
  echo "- Compose: $COMPOSE_FILE"
  echo "- Output: $OUT_DIR"
  echo "- Trace: $TRACE"
  echo "- Base URL: $BASE_URL"
  echo "- Tokenizer: $TOKENIZER"
  echo "- Started UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT_DIR/notes.md"

docker compose -f "$COMPOSE_FILE" config -q

if [[ -e "$TOKENIZER" || "$TOKENIZER" != "/model" ]]; then
  "$PYTHON" scripts/analyze_trace.py --trace "$TRACE" --tokenizer "$TOKENIZER" --output "$OUT_DIR/trace-summary.json"
else
  "$PYTHON" scripts/analyze_trace.py --trace "$TRACE" --output "$OUT_DIR/trace-summary.json"
  echo "Tokenizer path $TOKENIZER not found; exact /model token check skipped." >> "$OUT_DIR/notes.md"
fi

cleanup() {
  status=$?
  {
    echo "- Completed UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "- Exit status: $status"
  } >> "$OUT_DIR/notes.md"
  if [[ "${KEEP_SERVER:-0}" != "1" ]]; then
    docker compose -p "$PROJECT_NAME" -f "$COMPOSE_FILE" down --remove-orphans >> "$OUT_DIR/server.log" 2>&1 || true
  fi
  exit "$status"
}
trap cleanup EXIT

docker compose -p "$PROJECT_NAME" -f "$COMPOSE_FILE" down --remove-orphans >> "$OUT_DIR/server.log" 2>&1 || true
if curl -fsS "$BASE_URL/v1/models" >> "$OUT_DIR/health.log" 2>&1 || curl -fsS "$BASE_URL/health" >> "$OUT_DIR/health.log" 2>&1; then
  echo "A server is already responding at $BASE_URL before this candidate starts. Stop it or set BASE_URL to an isolated endpoint." >&2
  exit 1
fi
docker compose -p "$PROJECT_NAME" -f "$COMPOSE_FILE" up >> "$OUT_DIR/server.log" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + HEALTH_TIMEOUT))
until curl -fsS "$BASE_URL/v1/models" >> "$OUT_DIR/health.log" 2>&1 || curl -fsS "$BASE_URL/health" >> "$OUT_DIR/health.log" 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server process exited before health check passed. See $OUT_DIR/server.log" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Server did not become healthy within ${HEALTH_TIMEOUT}s. See $OUT_DIR/server.log" >&2
    exit 1
  fi
  sleep 5
done

"$PYTHON" scripts/replay_trace.py \
  --trace "$TRACE" \
  --base-url "$BASE_URL" \
  --output "$OUT_DIR/replay.jsonl" \
  --summary-output "$OUT_DIR/replay-summary.json" \
  --timeout "$REPLAY_TIMEOUT" \
  --max-connections "$MAX_CONNECTIONS"

"$PYTHON" scripts/score_ers.py --metrics "$OUT_DIR/replay.jsonl" > "$OUT_DIR/score.json"
"$PYTHON" scripts/compare_runs.py --runs-dir "$(dirname "$OUT_DIR")" > "$(dirname "$OUT_DIR")/comparison.md" || true

echo "Run complete: $OUT_DIR"
