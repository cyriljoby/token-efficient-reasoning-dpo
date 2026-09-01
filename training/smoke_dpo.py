"""End-to-end DPO pipeline check on a tiny model. CPU-only, seconds to run.

Build-order step 1: confirm TRL's DPOTrainer runs start to finish before any
real model or data is involved. Proves the plumbing, not the method -- a 2.4M
parameter random model learns nothing here, and the loss it reports is
meaningless as a result.

    .venv/bin/python training/smoke_dpo.py
"""

from datasets import Dataset
from trl import DPOConfig, DPOTrainer

MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

# Shaped like the real thing: same prompt, one shorter and one longer correct
# answer, preferring the shorter. Content is irrelevant at this scale.
PAIRS = [
    {
        "prompt": "What is 2 + 2?",
        "chosen": "4",
        "rejected": "Let me think step by step. First 2, then add 2 more, giving 4.",
    },
    {
        "prompt": "What is 10 - 3?",
        "chosen": "7",
        "rejected": "Starting from 10 and counting down 3 gives 9, 8, 7. The answer is 7.",
    },
    {
        "prompt": "What is 5 * 6?",
        "chosen": "30",
        "rejected": "5 sixes is 6 + 6 + 6 + 6 + 6, which sums to 30.",
    },
    {
        "prompt": "What is 12 / 4?",
        "chosen": "3",
        "rejected": "Dividing 12 into groups of 4 gives 3 groups, so the answer is 3.",
    },
]


def main() -> None:
    dataset = Dataset.from_list(PAIRS)

    config = DPOConfig(
        output_dir="outputs/smoke_dpo",
        per_device_train_batch_size=2,
        max_steps=2,
        learning_rate=1e-4,
        beta=0.1,
        loss_type="sigmoid",
        logging_steps=1,
        report_to=[],
        use_cpu=True,
        seed=0,
    )

    trainer = DPOTrainer(
        model=MODEL_ID,
        args=config,
        train_dataset=dataset,
    )
    trainer.train()

    losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    print(f"\npipeline ran end to end; {len(losses)} logged steps, losses: {losses}")


if __name__ == "__main__":
    main()
