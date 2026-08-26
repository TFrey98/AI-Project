"""Score generated counterfactual answers by selected semantic value."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from story_model.semantic_evaluation import (
    load_semantic_answer_keys,
    score_diagnostic_rows,
    summarize_semantic_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--answer-keys", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    diagnostic_path = Path(args.diagnostics)
    rows = tuple(
        json.loads(line)
        for line in diagnostic_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    answer_keys = load_semantic_answer_keys(args.answer_keys)
    scored = score_diagnostic_rows(rows, answer_keys)
    summaries = summarize_semantic_rows(scored)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in scored
        ),
        encoding="utf-8",
    )
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for split, summary in summaries.items():
        print(
            f"{split}: semantic {summary['semantic_correct_rate']:.1%}, "
            "complete pairs "
            f"{summary['counterfactual_pair_semantic_rate']:.1%}, "
            "expected value mentioned "
            f"{summary['expected_value_mention_rate']:.1%}"
        )

    print(f"scored details: {output_path}")
    print(f"semantic summary: {summary_path}")


if __name__ == "__main__":
    main()
