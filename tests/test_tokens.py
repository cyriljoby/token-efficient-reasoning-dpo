"""Fixture tests for token counting. CLOSE stands in for the `</think>` id."""

from evaluation.tokens import count_tokens

CLOSE = 99


def test_splits_at_delimiter():
    counts = count_tokens([1, 2, 3, CLOSE, 4, 5], CLOSE, truncated=False)
    assert (counts.reasoning, counts.answer, counts.total) == (3, 2, 6)
    assert counts.has_think_close


def test_delimiter_counts_toward_neither_bucket():
    counts = count_tokens([1, 2, CLOSE, 3], CLOSE, truncated=False)
    assert counts.reasoning + counts.answer == counts.total - 1


def test_truncated_without_delimiter_is_all_reasoning():
    counts = count_tokens([1, 2, 3], CLOSE, truncated=True)
    assert (counts.reasoning, counts.answer) == (3, 0)
    assert not counts.has_think_close
    assert counts.truncated


def test_untruncated_without_delimiter_skipped_reasoning():
    counts = count_tokens([1, 2, 3], CLOSE, truncated=False)
    assert (counts.reasoning, counts.answer) == (0, 3)
    assert not counts.has_think_close


def test_uses_last_delimiter():
    counts = count_tokens([1, CLOSE, 2, CLOSE, 3], CLOSE, truncated=False)
    assert (counts.reasoning, counts.answer) == (3, 1)


def test_delimiter_first():
    counts = count_tokens([CLOSE, 1, 2], CLOSE, truncated=False)
    assert (counts.reasoning, counts.answer) == (0, 2)


def test_delimiter_last():
    counts = count_tokens([1, 2, CLOSE], CLOSE, truncated=False)
    assert (counts.reasoning, counts.answer) == (2, 0)


def test_truncated_with_delimiter_still_splits():
    # Ran out of budget while writing the answer, not the reasoning.
    counts = count_tokens([1, 2, CLOSE, 3], CLOSE, truncated=True)
    assert (counts.reasoning, counts.answer) == (2, 1)
    assert counts.truncated


def test_empty_generation():
    counts = count_tokens([], CLOSE, truncated=False)
    assert (counts.reasoning, counts.answer, counts.total) == (0, 0, 0)
