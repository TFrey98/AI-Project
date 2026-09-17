"""Tests for the Phase 36e span-failure taxonomy.

Documented in docs/phase36e_frozen_span_failure.md. These cover the pure
classification functions in `story_model.span_failure_analysis` — every
branch of `classify_gold_span_prediction` (correct, missing_entirely,
misplaced_start, early_ending, fragmentation, late_ending), the
zero-byte-overlap definition of a spurious span, and the collapsed
positive-vs-outside confusion that is deliberately independent of B/I
identity. No model is involved: these are hand-built tag sequences, so a
taxonomy regression fails here rather than silently changing an audit's
reported numbers.
"""

import pytest

from story_model.explicit_offset_candidate_proposer import (
    OUTSIDE_TAG,
    EvidenceSpan,
    begin_tag,
    inside_tag,
)
from story_model.span_failure_analysis import (
    classify_gold_span_prediction,
    positive_vs_outside_confusion,
    span_length_bucket,
    spurious_predicted_spans,
)

B = begin_tag("route")
I = inside_tag("route")
O = OUTSIDE_TAG


def test_classifies_exact_match_as_correct():
    tags = (O, B, I, I, O)
    assert classify_gold_span_prediction(1, 4, B, I, tags) == "correct"


def test_classifies_missing_span_as_missing_entirely():
    tags = (O, O, O, O, O)
    assert classify_gold_span_prediction(1, 4, B, I, tags) == "missing_entirely"


def test_classifies_wrong_start_position_as_misplaced_start():
    # Gold begins at 1, but the model's positive run begins at 2.
    tags = (O, O, B, I, O)
    assert classify_gold_span_prediction(1, 4, B, I, tags) == "misplaced_start"


def test_classifies_premature_outside_as_early_ending():
    tags = (O, B, I, O, O)
    assert classify_gold_span_prediction(1, 4, B, I, tags) == "early_ending"


def test_classifies_mid_span_restart_as_fragmentation():
    tags = (O, B, B, I, O)
    assert classify_gold_span_prediction(1, 4, B, I, tags) == "fragmentation"


def test_classifies_mid_span_type_switch_as_fragmentation():
    other_inside = inside_tag("container")
    tags = (O, B, other_inside, I, O)
    assert classify_gold_span_prediction(1, 4, B, I, tags) == "fragmentation"


def test_classifies_overextension_immediately_after_as_late_ending():
    tags = (O, B, I, I, I)
    assert classify_gold_span_prediction(1, 4, B, I, tags) == "late_ending"


def test_rejects_out_of_range_span():
    with pytest.raises(ValueError):
        classify_gold_span_prediction(1, 10, B, I, (O, B, I))


def test_spurious_spans_have_zero_byte_overlap_with_gold():
    gold = (EvidenceSpan(0, 5, "abcde", "route"),)
    overlapping = EvidenceSpan(3, 8, "de-xyz", "route")
    disjoint = EvidenceSpan(10, 15, "hijkl", "route")
    result = spurious_predicted_spans((overlapping, disjoint), gold)
    assert result == (disjoint,)


def test_positive_vs_outside_confusion_is_independent_of_begin_inside_identity():
    gold = (O, B, I, I, O)
    # Perfect B/I discrimination but every positive byte predicted as O.
    predicted = (O, O, O, O, O)
    result = positive_vs_outside_confusion(gold, predicted)
    assert result["gold_positive_as_outside_rate"] == 1.0
    assert result["gold_outside_as_positive_rate"] == 0.0


def test_positive_vs_outside_confusion_requires_matching_lengths():
    with pytest.raises(ValueError):
        positive_vs_outside_confusion((O, B), (O,))


@pytest.mark.parametrize(
    "length,bucket",
    [
        (1, "1-8_bytes"),
        (8, "1-8_bytes"),
        (9, "9-16_bytes"),
        (16, "9-16_bytes"),
        (17, "17-24_bytes"),
        (25, "25plus_bytes"),
    ],
)
def test_span_length_bucket_boundaries(length, bucket):
    assert span_length_bucket(length) == bucket
