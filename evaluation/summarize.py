"""Aggregate eval runs into the reported numbers, with bootstrap CIs.

Reads the JSONL written by run_eval.py -- one file per condition -- and reports
accuracy and token counts side by side. Separate from generation because
generation is expensive and aggregation is not: re-analyse as often as needed
without regenerating anything.

Intervals are percentile bootstrap over problems. Decoding is greedy, so the
uncertainty being quantified is "which problems ended up in the eval set", not
sampling noise. n is printed with every number.

    python -m evaluation.summarize data/eval/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def bootstrap_ci(values: list[float], rng: random.Random) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean."""
    n = len(values)
    means = []
    for _ in range(BOOTSTRAP_SAMPLES):
        means.append(sum(rng.choices(values, k=n)) / n)
    means.sort()
    lo = int((1 - CONFIDENCE) / 2 * BOOTSTRAP_SAMPLES)
    hi = int((1 + CONFIDENCE) / 2 * BOOTSTRAP_SAMPLES) - 1
    return means[lo], means[hi]


def summarize(path: Path, rng: random.Random) -> dict:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    correct = [float(r["correct"]) for r in rows]
    total = [float(r["total_tokens"]) for r in rows]
    reasoning = [float(r["reasoning_tokens"]) for r in rows]

    accuracy_lo, accuracy_hi = bootstrap_ci(correct, rng)
    total_lo, total_hi = bootstrap_ci(total, rng)

    return {
        "condition": rows[0]["condition"],
        "n": len(rows),
        "accuracy": statistics.mean(correct),
        "accuracy_ci": (accuracy_lo, accuracy_hi),
        "total_tokens": statistics.mean(total),
        "total_tokens_ci": (total_lo, total_hi),
        "reasoning_tokens": statistics.mean(reasoning),
        "answer_tokens": statistics.mean(float(r["answer_tokens"]) for r in rows),
        # Reported per condition: a model that spirals less is improving for a
        # different reason than one that reasons more concisely.
        "truncated": statistics.mean(float(r["truncated"]) for r in rows),
        "no_answer": statistics.mean(float(r["no_answer"]) for r in rows),
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    results = [summarize(path, rng) for path in args.files]

    header = (f"{'condition':<18} {'n':>5} {'accuracy':>22} {'total tok':>20} "
              f"{'reason':>7} {'answer':>7} {'trunc':>6} {'no_ans':>7}")
    print(header)
    print("-" * len(header))
    for r in results:
        acc = f"{r['accuracy']:.3f} [{r['accuracy_ci'][0]:.3f},{r['accuracy_ci'][1]:.3f}]"
        tok = f"{r['total_tokens']:.0f} [{r['total_tokens_ci'][0]:.0f},{r['total_tokens_ci'][1]:.0f}]"
        print(f"{r['condition']:<18} {r['n']:>5} {acc:>22} {tok:>20} "
              f"{r['reasoning_tokens']:>7.0f} {r['answer_tokens']:>7.0f} "
              f"{r['truncated']:>6.1%} {r['no_answer']:>7.1%}")

    # Relative change against the first file, which is assumed to be the base.
    if len(results) > 1:
        base = results[0]
        print(f"\nrelative to {base['condition']} (n={base['n']}):")
        for r in results[1:]:
            d_tok = 100 * (r["total_tokens"] - base["total_tokens"]) / base["total_tokens"]
            d_acc = r["accuracy"] - base["accuracy"]
            print(f"  {r['condition']:<18} tokens {d_tok:+.1f}%   "
                  f"accuracy {d_acc:+.3f} ({base['accuracy']:.3f} -> {r['accuracy']:.3f})")


if __name__ == "__main__":
    main()
