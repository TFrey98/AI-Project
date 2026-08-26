"""Structured semantic scoring for counterfactual free-running output."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


SEMANTIC_TRANSFER_THRESHOLDS = {
    "train": {
        "semantic_correct_rate": 0.90,
        "counterfactual_pair_semantic_rate": 0.80,
        "end_stop_rate": 0.95,
        "degenerate_loop_rate_max": 0.05,
        "context_helped_rate": 0.75,
        "mean_context_advantage": 0.12,
    },
    "val": {
        "semantic_correct_rate": 0.75,
        "counterfactual_pair_semantic_rate": 0.60,
        "end_stop_rate": 0.95,
        "degenerate_loop_rate_max": 0.05,
        "context_helped_rate": 0.75,
        "mean_context_advantage": 0.12,
    },
    "lexical": {
        "semantic_correct_rate": 0.60,
        "counterfactual_pair_semantic_rate": 0.40,
        "end_stop_rate": 0.95,
        "degenerate_loop_rate_max": 0.05,
        "context_helped_rate": 0.60,
        "mean_context_advantage": 0.08,
    },
    "paraphrase": {
        "semantic_correct_rate": 0.60,
        "counterfactual_pair_semantic_rate": 0.40,
        "end_stop_rate": 0.95,
        "degenerate_loop_rate_max": 0.05,
        "context_helped_rate": 0.60,
        "mean_context_advantage": 0.08,
    },
    "transfer": {
        "semantic_correct_rate": 0.40,
        "counterfactual_pair_semantic_rate": 0.25,
        "end_stop_rate": 0.95,
        "degenerate_loop_rate_max": 0.05,
        "context_helped_rate": 0.50,
        "mean_context_advantage": 0.05,
    },
}

_NEGATIVE_CUES = (
    "avoid",
    "barred",
    "blocked",
    "closed",
    "collapsed",
    "cut off",
    "damaged",
    "do not",
    "don't",
    "flooded",
    "guarded",
    "impassable",
    "not passable",
    "not traversable",
    "obstructed",
    "sealed",
    "unavailable",
    "unsafe",
    "washed away",
)
_POSITIVE_CUES = (
    "available",
    "choose",
    "clear",
    "continue",
    "follow",
    "go by",
    "open",
    "pass by",
    "passable",
    "proceed",
    "remains usable",
    "take",
    "traversable",
    "unblocked",
    "use",
    "viable",
)


def _normalized_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _phrase_position(text: str, phrase: str) -> tuple[int, int] | None:
    normalized = _normalized_text(text)
    words = re.split(r"\s+", _normalized_text(phrase).strip())
    pattern = r"(?<!\w)" + r"\s+".join(
        re.escape(word) for word in words
    ) + r"(?!\w)"
    match = re.search(pattern, normalized)
    return match.span() if match is not None else None


def _cue_associations(
    text: str,
    expected_span: tuple[int, int],
    alternative_span: tuple[int, int],
    cues: tuple[str, ...],
) -> tuple[bool, bool]:
    normalized = _normalized_text(text)
    expected_center = sum(expected_span) / 2
    alternative_center = sum(alternative_span) / 2
    expected_found = False
    alternative_found = False

    for cue in cues:
        start = 0

        while True:
            position = normalized.find(cue, start)

            if position < 0:
                break

            center = position + len(cue) / 2

            if abs(center - expected_center) < abs(
                center - alternative_center
            ):
                expected_found = True
            else:
                alternative_found = True

            start = position + len(cue)

    return expected_found, alternative_found


def score_semantic_value(
    response: str,
    expected_value: str,
    alternative_value: str,
    skill: str,
) -> dict:
    """Score which member of a counterfactual value pair was selected."""

    expected_span = _phrase_position(response, expected_value)
    alternative_span = _phrase_position(response, alternative_value)

    if expected_span is None:
        return {
            "semantic_correct": False,
            "semantic_status": (
                "alternative_only"
                if alternative_span is not None
                else "value_missing"
            ),
            "expected_value_mentioned": False,
            "alternative_value_mentioned": alternative_span is not None,
        }

    if alternative_span is None:
        return {
            "semantic_correct": True,
            "semantic_status": "expected_only",
            "expected_value_mentioned": True,
            "alternative_value_mentioned": False,
        }

    expected_negative, alternative_negative = _cue_associations(
        response,
        expected_span,
        alternative_span,
        _NEGATIVE_CUES,
    )
    expected_positive, alternative_positive = _cue_associations(
        response,
        expected_span,
        alternative_span,
        _POSITIVE_CUES,
    )

    if alternative_negative and not expected_negative:
        correct = True
        status = "expected_selected_alternative_rejected"
    elif expected_negative and not alternative_negative:
        correct = False
        status = "expected_rejected"
    elif skill == "scene_route" and (
        expected_positive != alternative_positive
    ):
        correct = expected_positive
        status = (
            "expected_selected" if correct else "alternative_selected"
        )
    else:
        correct = expected_span[0] < alternative_span[0]
        status = "expected_first" if correct else "alternative_first"

    return {
        "semantic_correct": correct,
        "semantic_status": status,
        "expected_value_mentioned": True,
        "alternative_value_mentioned": True,
    }


def load_semantic_answer_keys(path: str | Path) -> dict[str, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    if payload.get("schema_version") != 1:
        raise ValueError("unsupported semantic answer-key schema")

    entries = payload.get("entries")

    if not isinstance(entries, dict) or not entries:
        raise ValueError("semantic answer keys must contain entries")

    return entries


def score_diagnostic_rows(
    rows: Iterable[dict],
    answer_keys: dict[str, dict],
) -> tuple[dict, ...]:
    """Join diagnostic rows to answer keys and attach semantic fields."""

    scored = []

    for row in rows:
        context_id = str(row["scenario_id"])

        if context_id not in answer_keys:
            raise ValueError(f"missing semantic answer key: {context_id}")

        key = answer_keys[context_id]

        for field in ("split", "conversation_id", "skill"):
            if str(row[field]) != str(key[field]):
                raise ValueError(
                    f"semantic answer key {field} mismatch for {context_id}"
                )

        response = row.get("full_context", {}).get("response")

        if not isinstance(response, str):
            raise ValueError(
                f"diagnostic row {context_id} has no generated response"
            )

        metrics = score_semantic_value(
            response=response,
            expected_value=str(key["expected_value"]),
            alternative_value=str(key["alternative_value"]),
            skill=str(key["skill"]),
        )
        scored.append(
            {
                **row,
                "expected_value": key["expected_value"],
                "alternative_value": key["alternative_value"],
                **metrics,
            }
        )

    return tuple(scored)


def summarize_semantic_rows(rows: Iterable[dict]) -> dict[str, dict]:
    """Aggregate semantic value selection by split and complete pair."""

    grouped: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        grouped[str(row["split"])].append(row)

    if not grouped:
        raise ValueError("semantic diagnostic rows cannot be empty")

    summaries = {}

    for split, split_rows in grouped.items():
        conversations: dict[str, list[dict]] = defaultdict(list)

        for row in split_rows:
            conversations[str(row["conversation_id"])].append(row)

        incomplete = [
            conversation_id
            for conversation_id, pair in conversations.items()
            if len(pair) != 2
        ]

        if incomplete:
            raise ValueError(
                f"semantic scoring requires complete pairs in {split}: "
                f"{incomplete[0]}"
            )

        count = len(split_rows)
        summaries[split] = {
            "examples": count,
            "pairs": len(conversations),
            "semantic_correct_rate": sum(
                bool(row["semantic_correct"]) for row in split_rows
            )
            / count,
            "counterfactual_pair_semantic_rate": sum(
                all(bool(row["semantic_correct"]) for row in pair)
                for pair in conversations.values()
            )
            / len(conversations),
            "expected_value_mention_rate": sum(
                bool(row["expected_value_mentioned"])
                for row in split_rows
            )
            / count,
            "alternative_value_mention_rate": sum(
                bool(row["alternative_value_mentioned"])
                for row in split_rows
            )
            / count,
            "statuses": dict(
                sorted(
                    Counter(
                        str(row["semantic_status"])
                        for row in split_rows
                    ).items()
                )
            ),
        }

    return dict(sorted(summaries.items()))


def semantic_transfer_acceptance_failures(
    summaries: dict[str, dict],
) -> tuple[str, ...]:
    """Apply Phase 28 semantic, stability, and grounding thresholds."""

    failures = []

    for split, thresholds in SEMANTIC_TRANSFER_THRESHOLDS.items():
        summary = summaries.get(split)

        if not isinstance(summary, dict):
            failures.append(f"missing diagnostic split: {split}")
            continue

        for metric, threshold in thresholds.items():
            if metric.endswith("_max"):
                source_metric = metric.removesuffix("_max")
                value = float(summary.get(source_metric, float("inf")))

                if value > threshold:
                    failures.append(
                        f"{split} {source_metric} {value:.3f} exceeds "
                        f"{threshold:.3f}"
                    )
            else:
                value = float(summary.get(metric, float("-inf")))

                if value < threshold:
                    failures.append(
                        f"{split} {metric} {value:.3f} is below "
                        f"{threshold:.3f}"
                    )

    return tuple(failures)
