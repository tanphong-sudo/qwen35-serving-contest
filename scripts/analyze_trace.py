#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qwen35_serving_bench.trace import iter_jsonl, stat_block, summarize_trace


def token_count(tokenized: object) -> int:
    return len(input_ids(tokenized))


def input_ids(tokenized: object) -> list[int]:
    if isinstance(tokenized, Mapping) and "input_ids" in tokenized:
        return list(tokenized["input_ids"])
    return list(tokenized)  # type: ignore[arg-type]


def common_prefix_length(sequences: list[list[int]]) -> int:
    if not sequences:
        return 0
    prefix = sequences[0]
    for sequence in sequences[1:]:
        idx = 0
        limit = min(len(prefix), len(sequence))
        while idx < limit and prefix[idx] == sequence[idx]:
            idx += 1
        prefix = prefix[:idx]
    return len(prefix)


def exact_token_counts(trace: Path, tokenizer_path: str) -> dict[str, object]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install transformers or omit --tokenizer") from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    counts: list[int] = []
    output_budgets: list[int] = []
    token_sequences: list[list[int]] = []
    for row in iter_jsonl(trace):
        body = row["body"]
        messages = body["messages"]
        if hasattr(tokenizer, "apply_chat_template"):
            token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        else:
            text = "\n".join(message.get("content", "") for message in messages)
            token_ids = tokenizer.encode(text)
        ids = input_ids(token_ids)
        token_sequences.append(ids)
        counts.append(len(ids))
        output_budgets.append(int(body.get("max_tokens") or 0))

    return {
        "exact_input_tokens": stat_block(counts),
        "exact_total_tokens": stat_block([count + output for count, output in zip(counts, output_budgets)]),
        "exact_common_prefix_tokens": common_prefix_length(token_sequences),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze contest trace shape.")
    parser.add_argument("--trace", type=Path, default=Path("input/trace-round1.jsonl"))
    parser.add_argument("--tokenizer", help="Optional local tokenizer path or HF repo id for exact token counts.")
    parser.add_argument("--output", type=Path, help="Optional path to write JSON summary.")
    args = parser.parse_args()

    summary = summarize_trace(args.trace)
    if args.tokenizer:
        summary.update(exact_token_counts(args.trace, args.tokenizer))

    payload = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
