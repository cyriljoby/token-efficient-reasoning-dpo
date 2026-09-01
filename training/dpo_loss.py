"""Custom DPO objective, hand-implemented for validation against TRL.

    L = -log σ( β · [ (log π(y_w|x) - log π_ref(y_w|x))
                    - (log π(y_l|x) - log π_ref(y_l|x)) ] )
"""

from __future__ import annotations

import torch

# use for pure computation
import torch.nn.functional as F

# Beta is the temperature parameter. Controls how much the model can deviate from the reference.
# Higher beta means the model is more strongly penalized for deviating from the reference.
# 0.1 is the default value and allows significant deviation from the reference, while 1.0 is a more strict penalty.
DEFAULT_BETA = 0.1

# this function computes log π(y|x) for a single model
def sequence_logprobs(
    logits: torch.Tensor, # logits[position] is a vector, where each element is a score indicating likelihood of that element being the next token 
    labels: torch.Tensor, # the true values at each position,
    loss_mask: torch.Tensor, # mask that differentiates prompt and padding from generated tokens
    *,
    average_log_prob: bool = False,
) -> torch.Tensor:
    """Per-sequence log-prob. Shapes: [B, T, V], [B, T], [B, T] -> [B].

    Summing (the default) is vanilla DPO. Averaging is a length-normalized
    objective/simpo
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)} do not match labels {tuple(labels.shape)}"
        )

    # Logits at position t predict token t+1: drop the last logit, first label.
    logits = logits[:, :-1]
    labels = labels[:, 1:]
    loss_mask = loss_mask[:, 1:]

    # Masked positions may hold pad or -100, which gather cannot index with.
    # Their values are zeroed below, so any in-range id works here.
    safe_labels = labels.masked_fill(loss_mask == 0, 0)

    # A vector for each sequence, with each position in the sequence containing the log-prob of the true label 
    per_token = torch.gather(
        logits.log_softmax(dim=-1), dim=2, index=safe_labels.unsqueeze(2)
    ).squeeze(2)
    per_token = per_token * loss_mask

    # SimPO branch: sums and length normalizes
    if average_log_prob:
        return per_token.sum(dim=-1) / loss_mask.sum(dim=-1).clamp(min=1)

    # DPO branch: sums the log-probs of the true labels for each sequence
    return per_token.sum(dim=-1) # [B]


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = DEFAULT_BETA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-example losses and implicit rewards, each [B].
    """

    # Step 1: compute the ratio (log π_θ - log π_ref) of new-to-old for both responses
    chosen_ratio = policy_chosen_logps - reference_chosen_logps      # ratio for y_w
    rejected_ratio = policy_rejected_logps - reference_rejected_logps # ratio for y_l

    # Step 2: calculate the difference between these log ratios
    ratio_diff = chosen_ratio - rejected_ratio

    # Step 3: feed into a sigmoid
    # Step 4: feed the preference probability into -log to convert it into a loss. high probabilities become low penalties, low probabilities become high penalties
    losses = -F.logsigmoid(beta * ratio_diff)

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards
