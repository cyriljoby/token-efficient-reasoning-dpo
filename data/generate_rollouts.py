"""Sample rollouts from the base model and score them.

Writes one JSONL row per rollout: the completion text plus everything the
pair constructor needs -- correctness, token counts, truncation. Generation is
the expensive half and pair construction is cheap, so they are separate passes:
rollouts are produced once and re-paired as often as needed (different rules,
the same-correctness ablation) without regenerating anything.

Rows stream to disk as they finish, so an interrupted run is still usable.

    .venv/bin/python -m data.generate_rollouts --n-prompts 15
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.grade import grade
from evaluation.tokens import count_tokens

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-prompts", type=int, default=15)
    parser.add_argument("--n-rollouts", type=int, default=6)
    # Settled decision: >=2048, identical across every condition.
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/rollouts/pilot.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Explicit placement, not device_map="auto": accelerate was offloading
    # layers to disk, which silently made generation many times slower.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    model.to(device)
    model.eval()

    close_id = tokenizer.convert_tokens_to_ids("</think>")
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    # Fixed-seed sample so the same prompts recur across runs.
    dataset = load_dataset("openai/gsm8k", "main", split=args.split)
    dataset = dataset.shuffle(seed=args.seed).select(range(args.n_prompts))

    print(f"model on {model.device}; {args.n_prompts} prompts x {args.n_rollouts} rollouts")
    print(f"writing to {args.out}")

    started = time.time()
    with args.out.open("w") as handle:
        for prompt_index, row in enumerate(dataset):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["question"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            prompt_length = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    num_return_sequences=args.n_rollouts,
                    pad_token_id=pad_id,
                )

            for rollout_index, sequence in enumerate(outputs):
                generated = sequence[prompt_length:].tolist()
                # This tokenizer uses one id for both pad and eos, so trailing
                # ids are ambiguous. Cut at the first eos instead: everything
                # after it is padding from longer siblings in the batch, and its
                # absence is what truncation actually means. The eos itself is
                # dropped so counts are content only.
                if eos_id in generated:
                    generated = generated[: generated.index(eos_id)]
                    truncated = False
                else:
                    truncated = True
                completion = tokenizer.decode(generated, skip_special_tokens=False)

                result = grade(completion, row["answer"])
                counts = count_tokens(generated, close_id, truncated=truncated)

                handle.write(
                    json.dumps(
                        {
                            "prompt_index": prompt_index,
                            "rollout_index": rollout_index,
                            "question": row["question"],
                            "gold": row["answer"],
                            "completion": completion,
                            "correct": result.correct,
                            "grade_method": result.method,
                            "predicted": str(result.predicted_value),
                            "reasoning_tokens": counts.reasoning,
                            "answer_tokens": counts.answer,
                            "total_tokens": counts.total,
                            "truncated": counts.truncated,
                            "has_think_close": counts.has_think_close,
                        }
                    )
                    + "\n"
                )
            handle.flush()

            correct = sum(
                grade(tokenizer.decode(s[prompt_length:], skip_special_tokens=False), row["answer"]).correct
                for s in outputs
            )
            elapsed = time.time() - started
            print(
                f"[{prompt_index + 1}/{args.n_prompts}] {correct}/{args.n_rollouts} correct  "
                f"({elapsed / 60:.1f} min elapsed)",
                flush=True,
            )

    print(f"\ndone in {(time.time() - started) / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
