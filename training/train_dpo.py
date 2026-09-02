"""Train the policy with vanilla DPO (LoRA) on the preference pairs.

Uses TRL's DPOTrainer. TRL owns batching, the reference-model forward, and checkpointing.

LoRA means the reference model is the same weights with adapters disabled, so
only one copy of the base model is held in memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from datasets import Dataset
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer
from transformers import AutoTokenizer

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# This model's stop token. vLLM keeps it on generated text and TRL appends its
# own, so it is stripped when pairs are loaded.
EOS_TOKEN = "<｜end▁of▁sentence｜>"

# Appended by the chat template, so it belongs to the prompt, never the completion.
THINK_OPEN = "<think>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("data/pairs/train.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dpo"))
    # Higher beta keeps the policy nearer the reference.
    parser.add_argument("--beta", type=float, default=0.1)
    # LoRA tolerates roughly 10x a full fine-tuning rate: ~1% of params train.
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    # DPO overfits quickly on ~1.8k pairs.
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2560)
    parser.add_argument("--warmup-steps", type=int, default=10)
    # Rank is the honest limitation: a weak result and too small an r are not
    # separable from one run.
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_pairs(path: Path, tokenizer) -> Dataset:
    """Preference pairs -> a DPO-ready Dataset.

    Three plain-string columns: the prompt rendered through the chat template
    exactly as generation rendered it, plus the `chosen` and `rejected`
    completions verbatim.

    The prompt must reach the model in the same shape it had at generation
    time. Rollouts were sampled after the chat template appended `<think>`,
    so training on the bare question would teach a format that never occurs
    at eval.

    Conversational format (message lists) cannot be used here: this model's
    template renders an assistant turn by DROPPING everything up to and
    including `</think>`, which would strip the reasoning out of every
    completion and delete the length signal entirely. Rendering the prompt
    here and passing strings keeps TRL to plain concatenation.
    """
    rows = []
    with path.open() as handle:
        for line in handle:
            pair = json.loads(line)
            if pair["chosen_tokens"] > pair["rejected_tokens"]:
                raise ValueError(
                    f"chosen is longer than rejected: {pair['chosen_tokens']} "
                    f"> {pair['rejected_tokens']}"
                )
            # explicitly enforce the template: the model must see the same <think> it was sampled under.
            rows.append({
                "prompt": tokenizer.apply_chat_template(
                    [{"role": "user", "content": pair["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                "chosen": pair["chosen"].removesuffix(EOS_TOKEN),
                "rejected": pair["rejected"].removesuffix(EOS_TOKEN),
            })
    return Dataset.from_list(rows)


def build_lora_config(args) -> LoraConfig:
    """Adapters on attention and MLP projections.

    alpha = 2r keeps the a/r scaling at 2 across rank sweeps.
    """
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.05,
        # include all weight matrices in attention and MLP
        # skip RMSNorm (tiny, and adapting norms tends to destabilize training)
        # skip lm_head because adapting it reshapes output token distribution directly
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_training_config(args) -> DPOConfig:
    """DPOConfig.

    max_length must clear the longest completion — rejected responses are the
    long ones by construction, so truncation would strip the length signal
    right where it lives. Rollouts capped at 2048 + ~120 prompt tokens =
    2560 leaves headroom. Verify the trainer logs zero truncated examples.
    """
    return DPOConfig(
        output_dir=args.output_dir,
        # vanilla DPO -- validated against tests/test_dpo_loss_vs_trl.py
        loss_type="sigmoid",
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        # not the 1024 default: rejected completions hit ~2045 tokens and are
        # the long side of every pair by construction. Longest observed
        # sequence ~2165 (2045 completion + ~120 prompt).
        max_length=args.max_length,
        # two models in the graph, long sequences -- recompute over storing
        gradient_checkpointing=True,
        bf16=True,
        # ~10% of total optimizer steps (113 at default batch settings)
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.eos_token != EOS_TOKEN:
        raise ValueError(
            f"EOS_TOKEN {EOS_TOKEN!r} does not match the tokenizer's "
            f"{tokenizer.eos_token!r}; stop tokens would survive into training text"
        )

    dataset = load_pairs(args.pairs, tokenizer)
    print(f"{len(dataset)} pairs from {args.pairs}")

    trainer = DPOTrainer(
        model=MODEL_NAME,
        args=build_training_config(args),
        train_dataset=dataset,
        processing_class=tokenizer,
        # No reference model is passed: with LoRA the reference is the same
        # weights with the adapters disabled, so only one copy is in memory.
        peft_config=build_lora_config(args),
    )


    batch = next(iter(trainer.get_train_dataloader()))
    example = tokenizer.decode(batch["input_ids"][0])
    print(f"\nfirst training sequence begins:\n{example[:220]!r}\n")
    if THINK_OPEN not in example:
        raise ValueError("chat template was not applied: no <think> in the training text")

    trainer.train()

    trainer.save_model(str(args.output_dir))
    print(f"\nadapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
