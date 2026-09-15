"""Phase 36e frozen span-failure taxonomy.

Pure functions that classify how a model's per-byte typed-BIO predictions
relate to one gold span, and locate predicted spans with no gold counterpart
at all. No training, decoding policy, or model code lives here — this module
only interprets already-computed predictions.
"""

from __future__ import annotations

from story_model.explicit_offset_candidate_proposer import (
    OUTSIDE_TAG,
    EvidenceSpan,
    tag_is_begin,
)


SPAN_ERROR_TYPES = (
    "correct",
    "missing_entirely",
    "misplaced_start",
    "early_ending",
    "fragmentation",
    "late_ending",
)


def classify_gold_span_prediction(
    gold_byte_start: int,
    gold_byte_end: int,
    gold_begin_tag: int,
    gold_inside_tag: int,
    predicted_tags: tuple[int, ...],
) -> str:
    """Classify one gold span's outcome against the full predicted-tag row.

    ``predicted_tags`` is the complete per-byte predicted tag sequence for
    the row (same length and alignment as the row's gold byte tags).
    """
    if not 0 <= gold_byte_start < gold_byte_end <= len(predicted_tags):
        raise ValueError("gold span is out of range for the predicted tags")
    start_tag = predicted_tags[gold_byte_start]
    if start_tag != gold_begin_tag:
        region_has_positive = any(
            predicted_tags[position] != OUTSIDE_TAG
            for position in range(gold_byte_start, gold_byte_end)
        )
        return "misplaced_start" if region_has_positive else "missing_entirely"
    for position in range(gold_byte_start + 1, gold_byte_end):
        tag = predicted_tags[position]
        if tag == OUTSIDE_TAG:
            return "early_ending"
        if tag_is_begin(tag):
            return "fragmentation"
        if tag != gold_inside_tag:
            return "fragmentation"
    if gold_byte_end < len(predicted_tags):
        after_tag = predicted_tags[gold_byte_end]
        if after_tag == gold_inside_tag:
            return "late_ending"
    return "correct"


def spurious_predicted_spans(
    predicted_spans: tuple[EvidenceSpan, ...],
    gold_spans: tuple[EvidenceSpan, ...],
) -> tuple[EvidenceSpan, ...]:
    """Predicted spans that share zero bytes with any gold span."""

    def overlaps(a: EvidenceSpan, b: EvidenceSpan) -> bool:
        return a.byte_start < b.byte_end and b.byte_start < a.byte_end

    return tuple(
        predicted
        for predicted in predicted_spans
        if not any(overlaps(predicted, gold) for gold in gold_spans)
    )


def positive_vs_outside_confusion(
    gold_tags: tuple[int, ...],
    predicted_tags: tuple[int, ...],
) -> dict:
    """Collapsed O-vs-positive confusion, independent of B/I identity.

    The conditional B/I consistency objective never directly constrains the
    O logit, so correct B-vs-I discrimination can coexist with the model
    deciding the wrong bytes belong to a span at all. This measures that
    axis on its own.
    """
    if len(gold_tags) != len(predicted_tags):
        raise ValueError("gold and predicted tag sequences must be the same length")
    gold_outside_total = 0
    gold_outside_as_positive = 0
    gold_positive_total = 0
    gold_positive_as_outside = 0
    for gold, predicted in zip(gold_tags, predicted_tags):
        if gold == OUTSIDE_TAG:
            gold_outside_total += 1
            if predicted != OUTSIDE_TAG:
                gold_outside_as_positive += 1
        else:
            gold_positive_total += 1
            if predicted == OUTSIDE_TAG:
                gold_positive_as_outside += 1
    return {
        "gold_outside_total": gold_outside_total,
        "gold_outside_as_positive_count": gold_outside_as_positive,
        "gold_outside_as_positive_rate": (
            gold_outside_as_positive / gold_outside_total
            if gold_outside_total
            else 0.0
        ),
        "gold_positive_total": gold_positive_total,
        "gold_positive_as_outside_count": gold_positive_as_outside,
        "gold_positive_as_outside_rate": (
            gold_positive_as_outside / gold_positive_total
            if gold_positive_total
            else 0.0
        ),
    }


def span_length_bucket(byte_length: int) -> str:
    """A small set of fixed, human-readable span-length buckets."""
    if byte_length <= 8:
        return "1-8_bytes"
    if byte_length <= 16:
        return "9-16_bytes"
    if byte_length <= 24:
        return "17-24_bytes"
    return "25plus_bytes"
