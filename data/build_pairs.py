"""Build preference pairs from scored rollouts.

One pair per prompt: shortest correct response beats longest correct response.
Both are correct, so length is the only difference the model can learn from.

    .venv/bin/python -m data.build_pairs --in data/rollouts/pilot.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evaluation.grade import grade


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="source", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/pairs/pairs.jsonl"))
    # Pairs whose two sides are nearly the same length teach nothing about
    # brevity. 1.0 keeps everything.
    parser.add_argument("--min-ratio", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    by_prompt = defaultdict(list)
    for line in args.source.open():
        row = json.loads(line)
        # Re-grade rather than trust the stored field, which may predate a
        # grader fix. Truncated responses are excluded: their length is set by
        # max_new_tokens, not by the model.
        if row["truncated"] or not grade(row["completion"], row["gold"]).correct:
            continue
        by_prompt[row["prompt_index"]].append(row)

    written = skipped = 0
    with args.out.open("w") as handle:
        for prompt_index, rows in sorted(by_prompt.items()):
            if len(rows) < 2:
                skipped += 1
                continue

            rows.sort(key=lambda r: r["total_tokens"])
            shortest, longest = rows[0], rows[-1]

            if longest["total_tokens"] / shortest["total_tokens"] < args.min_ratio:
                skipped += 1
                continue

            handle.write(
                json.dumps(
                    {
                        "prompt": shortest["question"],
                        "chosen": shortest["completion"],
                        "rejected": longest["completion"],
                        "chosen_tokens": shortest["total_tokens"],
                        "rejected_tokens": longest["total_tokens"],
                    }
                )
                + "\n"
            )
            written += 1

    print(f"{written} pairs written to {args.out} ({skipped} prompts skipped)")


if __name__ == "__main__":
    main()
