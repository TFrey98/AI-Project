"""Gate a checkpoint on free-running reconstruction of known responses."""

from __future__ import annotations

import argparse
from dataclasses import replace

from story_model.character_chat import (
    generate_character_response,
    load_character_runtime,
)
from story_model.character_evaluate import (
    evaluate_character_generations,
)
from story_model.character_training import load_character_training_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--min-exact-fraction",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--min-end-fraction",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--min-mean-similarity",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help=(
            "Exit unsuccessfully unless the configured exact, end, and "
            "similarity thresholds are reached."
        ),
    )
    args = parser.parse_args()

    for value, label in (
        (args.min_exact_fraction, "min_exact_fraction"),
        (args.min_end_fraction, "min_end_fraction"),
        (args.min_mean_similarity, "min_mean_similarity"),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{label} must be between 0 and 1")

    runtime = load_character_runtime(
        args.checkpoint,
        device=args.device,
    )
    records = load_character_training_records(args.data)

    def generate(record, index):
        return generate_character_response(
            model=runtime.model,
            tokenizer=runtime.tokenizer,
            context=replace(record.context, target_response=None),
            block_size=runtime.block_size,
            device=runtime.device,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed + index,
            greedy=True,
        )

    report = evaluate_character_generations(records, generate)

    print(f"checkpoint: {args.checkpoint}")
    print(f"completed updates: {runtime.checkpoint_step}")
    print(f"data: {args.data}")
    print(f"device: {runtime.device}")

    for result in report["results"]:
        print(
            f"{result['context_id']}: "
            f"exact={result['exact_response']}, "
            f"end={result['end_stop']}, "
            f"prefix={result['prefix_fraction']:.1%}, "
            f"similarity={result['similarity']:.1%}, "
            f"stop={result['stop_reason']}"
        )

    exact_fraction = report["exact_responses"] / report["examples"]
    end_fraction = report["end_stops"] / report["examples"]
    gate_passed = (
        exact_fraction >= args.min_exact_fraction
        and end_fraction >= args.min_end_fraction
        and report["mean_similarity"] >= args.min_mean_similarity
    )
    print(
        "summary: "
        f"exact {report['exact_responses']}/{report['examples']}, "
        f"end {report['end_stops']}/{report['examples']}, "
        f"passed {report['passed']}/{report['examples']}, "
        f"mean prefix {report['mean_prefix_fraction']:.1%}, "
        f"mean similarity {report['mean_similarity']:.1%}"
    )

    if args.require_pass and not gate_passed:
        raise SystemExit("free-running character generation gate: failed")

    print(
        "free-running character generation gate: "
        + ("passed" if gate_passed else "not passed")
    )


if __name__ == "__main__":
    main()
