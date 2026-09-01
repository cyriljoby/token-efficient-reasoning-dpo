"""Fixture tests for the GSM8K grader.

Cases are hand-written completion strings covering the formatting variants and
failure modes the grader is expected to survive.
"""

from decimal import Decimal

import pytest

from evaluation.grade import (
    answer_segment,
    extract_gold,
    extract_prediction,
    grade,
    normalize_number,
)

GOLD_SOLUTION = "Janet sells 16 - 3 - 4 = 9 eggs.\n9 * 2 = 18\n#### 18"


# --- gold parsing ----------------------------------------------------------


def test_gold_from_full_solution():
    assert extract_gold(GOLD_SOLUTION) == Decimal(18)


def test_gold_from_bare_number():
    assert extract_gold("18") == Decimal(18)


def test_gold_with_separators():
    assert extract_gold("#### 1,000") == Decimal(1000)


# --- normalization ---------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("18", 18),
        ("18.", 18),
        ("18.00", 18),
        ("$18", 18),
        ("18%", 18),
        ("1,000", 1000),
        ("1,000,000", 1000000),
        ("-5", -5),
        ("−5", -5),  # unicode minus
        ("3.5", Decimal("3.5")),
        ("18 apples", 18),
        (r"\$18", 18),
        (r"\text{18}", 18),
    ],
)
def test_normalize_variants(text, expected):
    assert normalize_number(text) == Decimal(expected)


def test_normalize_rejects_non_numeric():
    assert normalize_number("eighteen") is None
    assert normalize_number("") is None


def test_trailing_zeros_compare_equal():
    assert normalize_number("18.00") == normalize_number("18")


# --- answer segment --------------------------------------------------------


def test_segment_is_text_after_think_block():
    assert answer_segment("<think>reasoning</think>The answer is 18.") == (
        "The answer is 18."
    )


def test_segment_uses_last_close_tag():
    completion = "<think>a</think>mid</think> final 18"
    assert answer_segment(completion) == " final 18"


def test_segment_none_when_think_never_closes():
    assert answer_segment("<think>ran out of budget mid-thought 42") is None


def test_segment_is_whole_text_when_no_think_block():
    assert answer_segment("The answer is 18.") == "The answer is 18."


# --- prediction extraction -------------------------------------------------


def test_boxed_is_preferred_over_trailing_prose():
    completion = r"</think>So we get $\boxed{18}$ eggs, or 9 dozen."
    text, value, method = extract_prediction(completion)
    assert (value, method) == (Decimal(18), "boxed")
    assert text == "18"


def test_last_boxed_wins():
    completion = r"</think>First \boxed{9}, corrected to \boxed{18}."
    _, value, _ = extract_prediction(completion)
    assert value == Decimal(18)


def test_boxed_with_nested_braces_is_brace_matched():
    completion = r"</think>\boxed{\frac{7}{2}}"
    text, value, method = extract_prediction(completion)
    assert text == r"\frac{7}{2}"
    # Not a plain number -- refuse rather than scavenge a digit from the nesting.
    assert (value, method) == (None, "none")


def test_boxed_with_units_still_parses():
    # Strictness must not reject the common `\boxed{18 \text{ apples}}` form.
    _, value, method = extract_prediction(r"</think>\boxed{18 \text{ apples}}")
    assert (value, method) == (Decimal(18), "boxed")


def test_boxed_with_two_numbers_is_ambiguous():
    _, value, method = extract_prediction(r"</think>\boxed{18, 20}")
    assert (value, method) == (None, "none")


def test_boxed_cut_off_mid_expression():
    _, value, method = extract_prediction(r"</think>the answer is \boxed{18")
    assert method == "last_number"
    assert value == Decimal(18)


def test_falls_back_to_last_number():
    completion = "</think>We had 16 eggs, ate 3, so the answer is 18."
    text, value, method = extract_prediction(completion)
    assert (value, method) == (Decimal(18), "last_number")
    assert text == "18"


def test_reasoning_numbers_are_not_read_as_the_answer():
    completion = "<think>16 - 3 - 4 = 9, times 2 = 999</think>The answer is 18."
    _, value, _ = extract_prediction(completion)
    assert value == Decimal(18)


def test_no_number_anywhere():
    assert extract_prediction("</think>I cannot determine this.") == (None, None, "none")


def test_unclosed_think_block_extracts_nothing():
    assert extract_prediction("<think>still working, maybe 18") == (None, None, "none")


# --- end-to-end grading ----------------------------------------------------


@pytest.mark.parametrize(
    "completion",
    [
        r"<think>work</think>The answer is $\boxed{18}$.",
        "<think>work</think>The answer is 18.",
        "<think>work</think>The answer is 18.00 eggs.",
        "<think>work</think>She makes $18 per day.",
        r"<think>work</think>\boxed{18.0}",
        "The answer is 18.",  # no reasoning block at all
    ],
)
def test_correct_completions(completion):
    assert grade(completion, GOLD_SOLUTION).correct


@pytest.mark.parametrize(
    "completion",
    [
        "<think>work</think>The answer is 17.",
        "<think>work</think>The answer is 180.",
        "<think>reasoning mentions 18 but never finishes",  # truncated
        "<think>work</think>I cannot determine this.",
        r"<think>work</think>\boxed{eighteen}",
    ],
)
def test_incorrect_completions(completion):
    assert not grade(completion, GOLD_SOLUTION).correct


def test_thousands_separator_end_to_end():
    assert grade("</think>The total is 1,000 dollars.", "#### 1000").correct


def test_result_carries_debug_fields():
    result = grade(r"<think>w</think>\boxed{18}", GOLD_SOLUTION)
    assert result.predicted_text == "18"
    assert result.predicted_value == Decimal(18)
    assert result.gold_value == Decimal(18)
    assert result.method == "boxed"
