"""Unit tests for the custom DPO objective, against hand-computed values.

Equivalence against TRL lives in test_dpo_loss_vs_trl.py; these pin the maths
independently so a TRL change cannot quietly redefine "correct".
"""

import math

import pytest
import torch

from training.dpo_loss import dpo_loss, sequence_logprobs


# --- dpo_loss --------------------------------------------------------------


def test_identical_policy_and_reference_gives_log_two():
    """The identity worth memorizing: at initialization the policy IS the
    reference, both log-ratios vanish, sigmoid sees 0, loss is -log(0.5)."""
    logps = torch.tensor([-10.0, -20.0])
    losses, chosen_rewards, rejected_rewards = dpo_loss(logps, logps, logps, logps)

    assert torch.allclose(losses, torch.full((2,), math.log(2)))
    assert torch.allclose(chosen_rewards, torch.zeros(2))
    assert torch.allclose(rejected_rewards, torch.zeros(2))


def test_hand_computed_value():
    # margin = (-1 - -3) - (-2 - -2) = 2; beta = 0.5 -> loss = -logsigmoid(1.0)
    losses, _, _ = dpo_loss(
        torch.tensor([-1.0]),
        torch.tensor([-3.0]),
        torch.tensor([-2.0]),
        torch.tensor([-2.0]),
        beta=0.5,
    )
    expected = -math.log(1 / (1 + math.exp(-1.0)))
    assert losses.item() == pytest.approx(expected, rel=1e-6)


def test_preferring_chosen_lowers_loss():
    baseline, _, _ = dpo_loss(*[torch.tensor([-5.0])] * 4)
    better, _, _ = dpo_loss(
        torch.tensor([-4.0]),  # chosen more likely under policy
        torch.tensor([-5.0]),
        torch.tensor([-5.0]),
        torch.tensor([-5.0]),
    )
    worse, _, _ = dpo_loss(
        torch.tensor([-6.0]),  # chosen less likely under policy
        torch.tensor([-5.0]),
        torch.tensor([-5.0]),
        torch.tensor([-5.0]),
    )
    assert better.item() < baseline.item() < worse.item()


def test_rewards_are_beta_scaled_logratios():
    _, chosen_rewards, rejected_rewards = dpo_loss(
        torch.tensor([-4.0]),
        torch.tensor([-9.0]),
        torch.tensor([-5.0]),
        torch.tensor([-5.0]),
        beta=0.1,
    )
    assert chosen_rewards.item() == pytest.approx(0.1 * 1.0)
    assert rejected_rewards.item() == pytest.approx(0.1 * -4.0)


def test_numerically_stable_at_large_margins():
    huge = torch.tensor([1e4, -1e4])
    zeros = torch.zeros(2)
    losses, _, _ = dpo_loss(huge, zeros, zeros, zeros, beta=1.0)

    assert torch.isfinite(losses).all()
    # Overwhelmingly right -> ~0 loss; overwhelmingly wrong -> loss ~= |margin|.
    assert losses[0].item() == pytest.approx(0.0, abs=1e-6)
    assert losses[1].item() == pytest.approx(1e4, rel=1e-6)


def test_gradient_pushes_chosen_up_and_rejected_down():
    policy_chosen = torch.tensor([-5.0], requires_grad=True)
    policy_rejected = torch.tensor([-5.0], requires_grad=True)
    reference = torch.tensor([-5.0])

    losses, _, _ = dpo_loss(policy_chosen, policy_rejected, reference, reference)
    losses.sum().backward()

    # Descending the loss raises chosen log-prob and lowers rejected.
    assert policy_chosen.grad.item() < 0
    assert policy_rejected.grad.item() > 0


def test_beta_scales_the_margin():
    args = (
        torch.tensor([-4.0]),
        torch.tensor([-5.0]),
        torch.tensor([-5.0]),
        torch.tensor([-5.0]),
    )
    small, _, _ = dpo_loss(*args, beta=0.1)
    large, _, _ = dpo_loss(*args, beta=1.0)
    # Same correctly-ordered pair, more aggressive beta -> lower loss.
    assert large.item() < small.item()


# --- sequence_logprobs -----------------------------------------------------


def _uniform_logits(batch: int, seq: int, vocab: int) -> torch.Tensor:
    """All-zero logits give a uniform distribution: every token has log(1/vocab)."""
    return torch.zeros(batch, seq, vocab)


def test_sums_per_token_logprobs():
    vocab = 4
    logits = _uniform_logits(1, 4, vocab)
    labels = torch.tensor([[0, 1, 2, 3]])
    mask = torch.ones(1, 4)

    # 4 positions, 1 dropped by the shift -> 3 scored tokens at log(1/4) each.
    result = sequence_logprobs(logits, labels, mask)
    assert result.item() == pytest.approx(3 * math.log(1 / vocab))


def test_average_log_prob_normalizes_by_length():
    vocab = 4
    logits = _uniform_logits(1, 4, vocab)
    labels = torch.tensor([[0, 1, 2, 3]])
    mask = torch.ones(1, 4)

    result = sequence_logprobs(logits, labels, mask, average_log_prob=True)
    assert result.item() == pytest.approx(math.log(1 / vocab))


def test_mask_excludes_prompt_and_padding():
    vocab = 4
    logits = _uniform_logits(1, 5, vocab)
    labels = torch.tensor([[0, 1, 2, 3, 0]])
    # Positions 0-1 are prompt, 4 is padding; only 2 and 3 are completion.
    mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]])

    result = sequence_logprobs(logits, labels, mask)
    assert result.item() == pytest.approx(2 * math.log(1 / vocab))


def test_length_alone_changes_the_sum():
    """The length bias, in one assertion: more completion tokens means a more
    negative sequence log-prob, with no change in per-token quality."""
    vocab = 4
    logits = _uniform_logits(2, 6, vocab)
    labels = torch.zeros(2, 6, dtype=torch.long)
    mask = torch.zeros(2, 6)
    mask[0, 1:3] = 1.0  # 2 completion tokens
    mask[1, 1:6] = 1.0  # 5 completion tokens

    result = sequence_logprobs(logits, labels, mask)
    assert result[1].item() < result[0].item()
    # Averaging removes exactly this effect.
    averaged = sequence_logprobs(logits, labels, mask, average_log_prob=True)
    assert averaged[0].item() == pytest.approx(averaged[1].item())


def test_alignment_is_off_by_one_shifted():
    """Logit at position t must score the label at t+1, not at t."""
    vocab = 3
    logits = torch.zeros(1, 3, vocab)
    logits[0, 0, 2] = 10.0  # position 0 strongly predicts token 2
    logits[0, 1, 0] = 10.0  # position 1 strongly predicts token 0

    labels = torch.tensor([[9 % vocab, 2, 0]])  # tokens at t=1,2 are 2 then 0
    mask = torch.tensor([[0.0, 1.0, 1.0]])

    result = sequence_logprobs(logits, labels, mask)
    # Both scored tokens were the confidently-predicted ones, so log-probs are
    # near zero. Mis-shifted alignment would score the unlikely tokens instead.
    assert result.item() > -0.01


def test_masked_out_positions_may_hold_invalid_ids():
    """-100 is the usual ignore index and would crash gather if not handled."""
    vocab = 4
    logits = _uniform_logits(1, 3, vocab)
    labels = torch.tensor([[-100, 1, -100]])
    mask = torch.tensor([[0.0, 1.0, 0.0]])

    result = sequence_logprobs(logits, labels, mask)
    assert result.item() == pytest.approx(math.log(1 / vocab))


def test_empty_mask_does_not_divide_by_zero():
    logits = _uniform_logits(1, 3, 4)
    labels = torch.zeros(1, 3, dtype=torch.long)
    mask = torch.zeros(1, 3)

    assert sequence_logprobs(logits, labels, mask).item() == 0.0
    assert sequence_logprobs(logits, labels, mask, average_log_prob=True).item() == 0.0


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        sequence_logprobs(torch.zeros(1, 4, 5), torch.zeros(1, 3, dtype=torch.long), torch.ones(1, 3))
