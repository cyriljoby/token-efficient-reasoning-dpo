"""Token counting for generated completions.

Splits a generation into reasoning and answer tokens at the `</think>`
delimiter. Operates on the token IDs the generator actually produced -- decoding
and re-tokenizing is not guaranteed to round-trip, which would drift the metric
away from what the model really generated.

The delimiter token itself counts toward neither bucket, so
`reasoning + answer` is `total` minus the delimiter when one is present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TokenCounts:
    reasoning: int
    answer: int
    total: int
    truncated: bool
    has_think_close: bool


def count_tokens(
    token_ids: Sequence[int],
    think_close_id: int,
    *,
    truncated: bool,
) -> TokenCounts:
    """Count reasoning and answer tokens in one generation.

    `think_close_id` is the id of `</think>`, e.g.
    `tokenizer.convert_tokens_to_ids("</think>")`. `truncated` comes from the
    sampler's finish_reason; it cannot be recovered from the tokens.

    With no `</think>`: a truncated generation never left its reasoning, while
    an untruncated one skipped reasoning entirely.
    """
    total = len(token_ids)

    index = _rindex(token_ids, think_close_id)
    if index is not None:
        return TokenCounts(
            reasoning=index,
            answer=total - index - 1,
            total=total,
            truncated=truncated,
            has_think_close=True,
        )

    if truncated:
        return TokenCounts(
            reasoning=total,
            answer=0,
            total=total,
            truncated=True,
            has_think_close=False,
        )

    return TokenCounts(
        reasoning=0,
        answer=total,
        total=total,
        truncated=False,
        has_think_close=False,
    )


def _rindex(sequence: Sequence[int], value: int) -> int | None:
    for index in range(len(sequence) - 1, -1, -1):
        if sequence[index] == value:
            return index
    return None
