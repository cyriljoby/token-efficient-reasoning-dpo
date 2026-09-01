"""Numerical equivalence between the custom DPO objective and TRL's.

Drives a real DPOTrainer on a tiny model, takes the batch it built, and checks
that the hand-written loss reproduces TRL's to floating-point tolerance.

This is the test that makes "custom DPO objective, validated against TRL" a
claim rather than an assertion. It downloads a 2.4M-parameter model on first
run; everything is CPU.
"""

import pytest
import torch

from training.dpo_loss import dpo_loss, sequence_logprobs

datasets = pytest.importorskip("datasets")
trl = pytest.importorskip("trl")

MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
BETA = 0.1

PAIRS = [
    {
        "prompt": "What is 2 + 2?",
        "chosen": "4",
        "rejected": "Let me think step by step. First 2, then add 2 more, giving 4.",
    },
    {
        "prompt": "What is 10 - 3?",
        "chosen": "7",
        "rejected": "Starting from 10 and counting down 3 gives 9, 8, 7, so 7.",
    },
]


@pytest.fixture(scope="module")
def trainer():
    from trl import DPOConfig, DPOTrainer

    config = DPOConfig(
        output_dir="outputs/test_equivalence",
        per_device_train_batch_size=2,
        beta=BETA,
        loss_type="sigmoid",
        report_to=[],
        use_cpu=True,
        seed=0,
    )
    return DPOTrainer(
        model=MODEL_ID,
        args=config,
        train_dataset=datasets.Dataset.from_list(PAIRS),
    )


@pytest.fixture(scope="module")
def batch(trainer):
    return next(iter(trainer.get_train_dataloader()))


def _custom_loss(trainer, inputs):
    """Recompute the loss with the hand-written objective on TRL's own batch."""
    model_kwargs = {
        k: v
        for k, v in inputs.items()
        if k not in {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}
    }
    input_ids = inputs["input_ids"]
    completion_mask = inputs["completion_mask"].float()

    with torch.no_grad():
        policy_logits = trainer.model(**model_kwargs).logits
        reference_logits = trainer.ref_model(**model_kwargs).logits

    policy_logps = sequence_logprobs(policy_logits, input_ids, completion_mask)
    reference_logps = sequence_logprobs(reference_logits, input_ids, completion_mask)

    # TRL stacks the batch as [chosen; rejected] along dim 0.
    policy_chosen, policy_rejected = policy_logps.chunk(2, dim=0)
    reference_chosen, reference_rejected = reference_logps.chunk(2, dim=0)

    losses, _, _ = dpo_loss(
        policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=BETA
    )
    return losses, policy_chosen, policy_rejected


def test_matches_trl_loss(trainer, batch):
    with torch.no_grad():
        trl_loss = trainer.compute_loss(trainer.model, batch, return_outputs=False)

    losses, _, _ = _custom_loss(trainer, batch)

    assert torch.allclose(losses.mean(), trl_loss, atol=1e-5), (
        f"custom {losses.mean().item()} vs TRL {trl_loss.item()}"
    )


def test_matches_trl_sequence_logprobs(trainer, batch):
    """The log-prob computation is where alignment and masking bugs hide, so
    check it directly against TRL's rather than only via the final scalar."""
    from trl.trainer.utils import selective_log_softmax

    model_kwargs = {
        k: v
        for k, v in batch.items()
        if k not in {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}
    }
    input_ids = batch["input_ids"]
    completion_mask = batch["completion_mask"]

    with torch.no_grad():
        logits = trainer.model(**model_kwargs).logits

    per_token = selective_log_softmax(logits[..., :-1, :], input_ids[..., 1:])
    per_token[completion_mask[..., 1:] == 0] = 0.0
    trl_logps = per_token.sum(dim=1)

    mine = sequence_logprobs(logits, input_ids, completion_mask.float())
    assert torch.allclose(mine, trl_logps, atol=1e-5), f"{mine} vs {trl_logps}"


def test_initial_loss_is_log_two(trainer, batch):
    """Untrained: policy and reference are the same model, so every log-ratio
    cancels and the loss must be log(2) -- the same identity the smoke run hit."""
    losses, _, _ = _custom_loss(trainer, batch)
    assert torch.allclose(losses, torch.full_like(losses, torch.log(torch.tensor(2.0))), atol=1e-5)
