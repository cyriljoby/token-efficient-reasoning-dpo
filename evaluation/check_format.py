"""Validate the evaluation layer against real model output.

`grade.py` and `tokens.py` are tested against hand-written fixtures, which
encode assumptions about how DeepSeek-R1-Distill emits reasoning. This script
checks those assumptions on actual generations before any of them are relied on
to build preference data:

  1. Does `</think>` tokenize to a single id? (`tokens.py` assumes it does.)
  2. Does the chat template inject `<think>` into the prompt, so generations
     begin mid-reasoning with no opening tag?
  3. Do completions land on `\\boxed{}`, or on the `last_number` fallback?
  4. What does the reasoning/answer split look like on real text?

    .venv/bin/python evaluation/check_format.py
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.grade import grade
from evaluation.tokens import count_tokens

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

N_PROMPTS = 5
# Real runs use >=2048 (a settled decision). Held lower here only so this check
# finishes quickly on a laptop; truncation is expected and is itself informative.
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.8


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    print(f"model: {MODEL_NAME}")
    print(f"device: {model.device}   dtype: {model.dtype}")

    # --- assumption 1: is </think> a single token? -------------------------
    close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    close_id = tokenizer.convert_tokens_to_ids("</think>")
    print(f"\n</think> encode -> {close_ids}  (single token: {len(close_ids) == 1})")
    print(f"</think> convert_tokens_to_ids -> {close_id}")
    print(f"<think>  encode -> {tokenizer.encode('<think>', add_special_tokens=False)}")

    # --- assumption 2: does the chat template inject <think>? --------------
    test = load_dataset("openai/gsm8k", "main", split="test")
    rows = [test[i] for i in range(N_PROMPTS)]

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": rows[0]["question"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    print("\n--- templated prompt (repr, tail) ---")
    print(repr(prompt[-200:]))
    print(f"contains '<think>': {'<think>' in prompt}")

    # --- assumptions 3 and 4: grade and count real completions -------------
    print(f"\n--- {N_PROMPTS} completions ---")
    for row in rows:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated_ids = output[0][inputs["input_ids"].shape[1] :].tolist()
        completion = tokenizer.decode(generated_ids, skip_special_tokens=False)

        # No EOS at the end means generation stopped at the length cap.
        truncated = generated_ids[-1] != tokenizer.eos_token_id

        result = grade(completion, row["answer"])
        counts = count_tokens(generated_ids, close_id, truncated=truncated)

        print(
            f"correct={result.correct!s:5} method={result.method:12} "
            f"pred={str(result.predicted_value):8} gold={str(result.gold_value):8} "
            f"reasoning={counts.reasoning:5} answer={counts.answer:4} "
            f"total={counts.total:5} trunc={counts.truncated!s:5} "
            f"close={counts.has_think_close}"
        )

    # --- one full completion, to eyeball the format ------------------------
    print("\n--- raw completion (last prompt, first 1200 chars) ---")
    print(completion[:1200])


if __name__ == "__main__":
    main()
