"""Generate and score one completion per GSM8K test problem for one condition.

A condition is a model (base, SFT, or a DPO/SimPO adapter merged into the base)
plus an optional concise-prompt instruction. Every condition runs through this
same script with the same decoding settings, so differences between them come
from the model rather than the harness.

Decoding is greedy: deterministic, reproducible, and identical across
conditions. Confidence intervals come from bootstrapping over problems in
summarize.py, not from sampling variance.

    python -m evaluation.run_eval --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \\
        --condition base --out data/eval/base.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset

from evaluation.grade import grade
from evaluation.tokens import count_tokens

BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Three variants rather than one: a single string makes the zero-training
# baseline hostage to prompt luck. Report the best of the three.
CONCISE_PROMPTS = {
    "none": None,
    "concise-a": "Answer concisely. Keep your reasoning brief.",
    "concise-b": (
        "Solve this using as few reasoning steps as possible. "
        "Do not restate the problem or verify your work."
    ),
    "concise-c": (
        "Think efficiently. Reason only as much as the problem requires, "
        "then give the answer."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=BASE_MODEL, help="model path or hub id")
    parser.add_argument("--condition", required=True, help="label recorded in every row")
    parser.add_argument("--prompt-variant", choices=sorted(CONCISE_PROMPTS), default="none")
    # Full test split by default. A smaller value takes a fixed-seed subset that
    # stays a true subset as n grows, so results remain comparable.
    parser.add_argument("--n-problems", type=int, default=0, help="0 = all 1319")
    # Settled decision: identical across every condition.
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def build_prompt(tokenizer, question: str, instruction: str | None) -> str:
    if instruction is not None:
        question = f"{instruction}\n\n{question}"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    close_id = tokenizer.convert_tokens_to_ids("</think>")
    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed)
    # Greedy: temperature 0 makes the run reproducible and removes sampling
    # noise from a comparison that is already noisy enough at n=1319.
    params = SamplingParams(n=1, temperature=0.0, max_tokens=args.max_new_tokens)

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if args.n_problems:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.n_problems))

    instruction = CONCISE_PROMPTS[args.prompt_variant]
    prompts = [build_prompt(tokenizer, row["question"], instruction) for row in dataset]

    print(f"condition={args.condition} model={args.model} "
          f"prompt={args.prompt_variant} n={len(prompts)}")

    started = time.time()
    outputs = llm.generate(prompts, params)

    with args.out.open("w") as handle:
        for index, (row, output) in enumerate(zip(dataset, outputs)):
            completion_out = output.outputs[0]
            token_ids = list(completion_out.token_ids)
            truncated = completion_out.finish_reason == "length"
            completion = tokenizer.decode(token_ids, skip_special_tokens=False)

            result = grade(completion, row["answer"])
            counts = count_tokens(token_ids, close_id, truncated=truncated)

            handle.write(json.dumps({
                "condition": args.condition,
                "prompt_variant": args.prompt_variant,
                "model": args.model,
                "problem_index": index,
                "question": row["question"],
                "gold": row["answer"],
                "completion": completion,
                "correct": result.correct,
                "no_answer": result.no_answer,
                "grade_method": result.method,
                "predicted": str(result.predicted_value),
                "reasoning_tokens": counts.reasoning,
                "answer_tokens": counts.answer,
                "total_tokens": counts.total,
                "truncated": counts.truncated,
            }) + "\n")

    print(f"done in {(time.time() - started) / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
