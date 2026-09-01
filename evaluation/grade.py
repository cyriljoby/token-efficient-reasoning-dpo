"""Correctness grading for GSM8K completions.

Extracts one predicted number from a completion and compares it to the gold
answer. Only the post-`</think>` segment is searched, so numbers mentioned
mid-reasoning are never mistaken for the answer.

Prefers extracting nothing over guessing: grader failures should be countable,
not silent. GSM8K gold answers are always plain numbers, so symbolic forms are
not compared.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

_GOLD_MARKER = "####"

# Permissive on the way in; normalization tightens it later.
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

_LATEX_NOISE = (r"\left", r"\right", r"\!", r"\,", r"\;", r"\:", r"\ ", r"\$", r"\%")


@dataclass(frozen=True)
class GradeResult:
    """`method` is "boxed", "last_number", or "none" -- tracked so the share of
    completions graded by the reliable path vs. the fallback is measurable."""

    correct: bool
    predicted_text: str | None
    predicted_value: Decimal | None
    gold_value: Decimal | None
    method: str


def answer_segment(completion: str) -> str | None:
    """Part of `completion` that should contain the final answer.

    None means an unclosed reasoning block: cut off before committing to an
    answer, which is incorrect rather than unparseable.
    """
    if THINK_CLOSE in completion:
        return completion.rsplit(THINK_CLOSE, 1)[1]
    if THINK_OPEN in completion:
        return None
    # No reasoning block at all. A real case to grade, not a malformed one --
    # a length-optimized model skipping reasoning is a result we expect.
    return completion


def _strip_latex(text: str) -> str:
    for token in _LATEX_NOISE:
        text = text.replace(token, "")
    # \text{...} and friends wrap units and stray words around the number.
    return re.sub(r"\\(?:text|mathrm|mbox)\{([^{}]*)\}", r"\1", text)


def _extract_boxed(text: str) -> str | None:
    r"""Contents of the last \boxed{...}. Brace-matched so nested braces
    (`\boxed{\frac{1}{2}}`) don't truncate at the first closing brace."""
    marker = r"\boxed"
    start = text.rfind(marker)
    if start == -1:
        return None

    cursor = start + len(marker)
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text) or text[cursor] != "{":
        return None

    depth = 0
    for index in range(cursor, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[cursor + 1 : index]
    return None  # cut off inside \boxed{


def normalize_number(text: str) -> Decimal | None:
    """Parse a number, tolerating separators, currency/percent, unicode minus,
    trailing punctuation, and trailing zeros (`18.00` == `18`)."""
    cleaned = _strip_latex(text)
    cleaned = cleaned.replace("\u2212", "-")
    cleaned = cleaned.replace("$", "").replace("%", "").replace(" ", "")
    cleaned = cleaned.replace("\u00a0", "")

    match = _NUMBER_RE.search(cleaned)
    if match is None:
        return None

    try:
        value = Decimal(match.group(0).replace(",", ""))
    except InvalidOperation:
        return None
    # normalize() equates 18.00 and 18; the +0 avoids exponent form (1E+3).
    return value.normalize() + Decimal(0)


def _parse_boxed_number(text: str) -> Decimal | None:
    r"""Strict parse of `\boxed{...}` contents.

    Looser parsing would read `\boxed{\frac{7}{2}}` as 7. A surviving backslash
    means unhandled LaTeX structure, and multiple numbers mean the box is
    ambiguous; both are refused rather than guessed at.
    """
    cleaned = _strip_latex(text)
    if "\\" in cleaned:
        return None
    if len(_NUMBER_RE.findall(cleaned.replace(",", ""))) != 1:
        return None
    return normalize_number(cleaned)


def extract_gold(gold: str) -> Decimal | None:
    """Accepts a full GSM8K solution (`... #### 18`) or a bare number."""
    if _GOLD_MARKER in gold:
        gold = gold.rsplit(_GOLD_MARKER, 1)[1]
    return normalize_number(gold)


def extract_prediction(completion: str) -> tuple[str | None, Decimal | None, str]:
    r"""Returns (raw_text, value, method). Prefers `\boxed{...}` as an explicit
    commitment; falls back to the last number in the answer segment."""
    segment = answer_segment(completion)
    if segment is None:
        return None, None, "none"

    boxed = _extract_boxed(segment)
    if boxed is not None:
        value = _parse_boxed_number(boxed)
        if value is not None:
            return boxed, value, "boxed"
        # A \boxed{} holding a non-number is a deliberate answer that happens
        # not to be numeric. Don't scavenge an unrelated number from earlier.
        return boxed, None, "none"

    matches = _NUMBER_RE.findall(_strip_latex(segment).replace(",", ""))
    if not matches:
        return None, None, "none"

    raw = matches[-1]
    return raw, normalize_number(raw), "last_number"


def grade(completion: str, gold: str) -> GradeResult:
    gold_value = extract_gold(gold)
    predicted_text, predicted_value, method = extract_prediction(completion)

    correct = (
        gold_value is not None
        and predicted_value is not None
        and predicted_value == gold_value
    )
    return GradeResult(
        correct=correct,
        predicted_text=predicted_text,
        predicted_value=predicted_value,
        gold_value=gold_value,
        method=method,
    )


def is_correct(completion: str, gold: str) -> bool:
    return grade(completion, gold).correct
