"""Sample rollouts from the base model and score them.

Writes one JSONL row per rollout: the completion text plus everything the pair
constructor needs: correctness, token counts, truncation. Generation is the
expensive half and pair construction is cheap, so they are separate passes.

Two backends. `hf` runs anywhere including Apple silicon, and is for small local
checks. `vllm` is for the real run on a CUDA box. It batches continuously
instead of waiting for each prompt's slowest rollout.

    .venv/bin/python -m data.generate_rollouts --n-prompts 15
    python -m data.generate_rollouts --backend vllm --n-prompts 2000 --resume
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset

from evaluation.grade import grade
from evaluation.tokens import count_tokens

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Ada (RTX 4090) and Apple silicon both support bfloat16.
DTYPE = "bfloat16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # number of prompts to sample from the dataset, and number of rollouts per prompt
    parser.add_argument("--n-prompts", type=int, default=15)
    parser.add_argument("--n-rollouts", type=int, default=6)

    # Settled decision: >=2048, identical across every condition.
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    # Higher value to ensure length variation
    parser.add_argument("--temperature", type=float, default=0.8)
    # split to pull data from (train/test)
    parser.add_argument("--split", default="train")
    # for reproducibility
    parser.add_argument("--seed", type=int, default=0)
    # hf for local tests, vllm for real runs on a cuda box
    parser.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    # vLLM returns a chunk only when all its prompts finish; smaller chunks
    # checkpoint more often, larger ones batch better.
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--resume", action="store_true", help="skip prompts already in --out")
    parser.add_argument("--out", type=Path, default=Path("data/rollouts/pilot.jsonl"))
    return parser.parse_args()


def done_prompts(path: Path) -> set[int]:
    """Prompt indices already written, so an interrupted run can continue.

    A run killed mid-chunk can leave a half-written final line. Skip it rather
    than crash: resume exists for exactly that situation.
    """
    if not path.exists():
        return set()
    done = set()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["prompt_index"])
            except json.JSONDecodeError:
                continue
    return done


class HFBackend:
    """transformers.generate: portable, slow. One prompt at a time."""

    def __init__(self, args):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(args.seed)
        self.torch = torch
        # loads correct tokenizer for the model
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Explicit placement, not device_map="auto" because accelerate was offloading to disk
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        # loads correct model for the model, with correct dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=getattr(torch, DTYPE)
        )
        self.model.to(self.device)
        self.model.eval()
        self.args = args

    def generate(self, prompts: list[str]) -> list[list[tuple[list[int], bool]]]:
        eos_id = self.tokenizer.eos_token_id
        results = []
        # one prompt at a time
        for prompt in prompts:
            # tokenize the prompt and move to the correct device
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            # save the prompt length
            prompt_length = inputs["input_ids"].shape[1]
            with self.torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.args.max_new_tokens,
                    do_sample=True,
                    temperature=self.args.temperature,
                    num_return_sequences=self.args.n_rollouts,
                    pad_token_id=self.tokenizer.pad_token_id or eos_id,
                )
            rollouts = []
            for sequence in outputs:
                # generated tokens are everything after the prompt
                generated = sequence[prompt_length:].tolist()
                # if the EOS token is present, truncate the generated tokens at the EOS token
                if eos_id in generated:
                    rollouts.append((generated[: generated.index(eos_id)], False))
                # if the EOS token is not present, the generation was truncated due to max_new_tokens
                else:
                    rollouts.append((generated, True))
            results.append(rollouts)
        return results


class VLLMBackend:
    """vLLM: continuous batching, and finish_reason instead of EOS-sniffing."""

    def __init__(self, args):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        # select correct tokenizer for the model
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # select correct model for the model, with correct dtype and seed
        self.llm = LLM(model=MODEL_NAME, dtype=DTYPE, seed=args.seed)
        # bundle all the sampling parameter together
        self.params = SamplingParams(
            n=args.n_rollouts,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
        )

    def generate(self, prompts: list[str]) -> list[list[tuple[list[int], bool]]]:
        # All prompts submitted at once
        outputs = self.llm.generate(prompts, self.params)
        # vllm provides a finish_reason for each rollout, so no need to sniff for EOS
        return [
            [
                (list(completion.token_ids), completion.finish_reason == "length")
                for completion in output.outputs
            ]
            for output in outputs
        ]


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    backend = VLLMBackend(args) if args.backend == "vllm" else HFBackend(args)
    tokenizer = backend.tokenizer
    close_id = tokenizer.convert_tokens_to_ids("</think>")

    dataset = load_dataset("openai/gsm8k", "main", split=args.split)
    dataset = dataset.shuffle(seed=args.seed).select(range(args.n_prompts))

    already = done_prompts(args.out) if args.resume else set()
    todo = [i for i in range(len(dataset)) if i not in already]
    if already:
        print(f"resuming: {len(already)} prompts already done, {len(todo)} to go")

    print(f"backend={args.backend} {len(todo)} prompts "
          f"x {args.n_rollouts} rollouts -> {args.out}")

    started = time.time()
    with args.out.open("a" if args.resume else "w") as handle:
        for chunk_start in range(0, len(todo), args.chunk_size):
            chunk = todo[chunk_start : chunk_start + args.chunk_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": dataset[i]["question"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for i in chunk
            ]

            for prompt_index, rollouts in zip(chunk, backend.generate(prompts)):
                row = dataset[prompt_index]
                for rollout_index, (generated, truncated) in enumerate(rollouts):
                    completion = tokenizer.decode(generated, skip_special_tokens=False)
                    result = grade(completion, row["answer"])
                    counts = count_tokens(generated, close_id, truncated=truncated)
                    handle.write(json.dumps({
                        "prompt_index": prompt_index,
                        "rollout_index": rollout_index,
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
                        "has_think_close": counts.has_think_close,
                    }) + "\n")
            handle.flush()
            print(f"[{chunk_start + len(chunk)}/{len(todo)}] "
                  f"{(time.time() - started) / 60:.1f} min elapsed", flush=True)

    print(f"\ndone in {(time.time() - started) / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
